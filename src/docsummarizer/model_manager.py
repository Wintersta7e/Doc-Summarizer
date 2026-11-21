# SPDX-License-Identifier: MIT
"""Download, load and run the local LLM."""

import ctypes
import functools
import importlib
import json
import os
import time
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
    NoReturn,
    NotRequired,
    Protocol,
    TypedDict,
    cast,
    final,
    overload,
)

from huggingface_hub import hf_hub_download
from typing_extensions import override

from docsummarizer.logger import (
    get_logger,
    get_memory_usage_mb,
    log_debug,
    log_error,
    log_info,
)
from docsummarizer.paths import app_data_dir
from docsummarizer.provenance import SourceSpan, locate_quote, split_sentences

if TYPE_CHECKING:
    from tqdm import tqdm

# Supported summary types. Kept as module-level string constants (rather than
# enum.StrEnum, which requires Python 3.11+) so callers can keep passing plain
# strings without an extra import.
SUMMARY_TYPE_BRIEF = "brief"
SUMMARY_TYPE_DETAILED = "detailed"
SUMMARY_TYPE_STRUCTURED = "structured"
SUMMARY_TYPES = (SUMMARY_TYPE_BRIEF, SUMMARY_TYPE_DETAILED, SUMMARY_TYPE_STRUCTURED)


@functools.lru_cache(maxsize=1)
def gpu_offload_supported() -> bool:
    """Whether the installed llama-cpp build can actually offload to a GPU.

    A CPU-only wheel silently ignores ``n_gpu_layers``, so the UI surfaces this
    rather than letting the GPU toggle pretend to work. Cached: the answer is
    fixed for a given build, and importing llama-cpp is expensive.
    """
    try:
        return bool(_load_llama_cpp().llama_supports_gpu_offload())
    except Exception:  # no runtime extra, or a library that fails to load
        logger = get_logger()
        logger.debug("GPU offload probe failed", exc_info=True)
        return False


class ChatCompletionResponseMessage(TypedDict):
    """The answer of a blocking completion."""

    role: str
    content: str | None


class ChatCompletionResponseChoice(TypedDict):
    """One choice of a blocking completion."""

    message: ChatCompletionResponseMessage
    finish_reason: str | None


class CompletionUsage(TypedDict):
    """The token counts llama-cpp reports for a blocking completion."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CreateChatCompletionResponse(TypedDict):
    """The fields of llama-cpp's blocking completion response that callers read."""

    choices: list[ChatCompletionResponseChoice]
    usage: NotRequired[CompletionUsage | None]


class ChatCompletionStreamResponseDelta(TypedDict, total=False):
    """The text a streamed chunk adds."""

    role: str
    content: str | None


class ChatCompletionStreamResponseChoice(TypedDict):
    """One choice of a streamed chunk."""

    delta: ChatCompletionStreamResponseDelta
    finish_reason: str | None


class CreateChatCompletionStreamResponse(TypedDict):
    """The fields of llama-cpp's streamed completion chunk that callers read."""

    choices: list[ChatCompletionStreamResponseChoice]


class ChatModel(Protocol):
    """The part of ``llama_cpp.Llama`` the summarizer calls; the eval runner wraps it."""

    @overload
    def create_chat_completion(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        stream: Literal[False] = False,
        **kwargs: object,
    ) -> "CreateChatCompletionResponse": ...

    @overload
    def create_chat_completion(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        stream: Literal[True],
        **kwargs: object,
    ) -> "Iterator[CreateChatCompletionStreamResponse]": ...

    def tokenize(self, text: bytes, *, add_bos: bool = True) -> list[int]:
        """The token ids of ``text``."""
        ...


class _LlamaFactory(Protocol):
    """``llama_cpp.Llama``'s constructor: the model path, then its integer and flag options."""

    def __call__(self, *, model_path: str, **options: int) -> ChatModel: ...


class _LlamaCpp(Protocol):
    """The members of the ``llama_cpp`` package this module uses."""

    Llama: _LlamaFactory
    # The low-level ctypes binding. Its functions vary by build, so ``_binding``
    # probes each one by name.
    llama_cpp: object

    def llama_supports_gpu_offload(self) -> bool:
        """Whether this build can offload layers to a GPU."""
        ...


def _load_llama_cpp() -> _LlamaCpp:
    """The optional ``llama_cpp`` runtime extra; ``ImportError`` when it is not installed.

    Loaded by name so that the type checker reads the protocol above, not the
    package, which the gate environment does not install.
    """
    # Through ``object``: a module and a protocol share no declared members.
    module = cast("object", importlib.import_module("llama_cpp"))
    return cast("_LlamaCpp", module)


class _PerfCounters(Protocol):
    """The fields of llama.cpp's ``llama_perf_context_data`` the stats read."""

    n_p_eval: int
    n_eval: int
    t_p_eval_ms: float
    t_eval_ms: float


class SummarizationCancelledError(Exception):
    """Raised between chunks when the caller requests a clean stop."""


def _binding(module: object, name: str) -> Callable[..., object] | None:
    """A function of llama-cpp's low-level binding, or ``None`` on a build without it."""
    found: object = getattr(module, name, None)
    return found if callable(found) else None


def _raise_if_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel and should_cancel():
        raise SummarizationCancelledError


@dataclass(frozen=True)
class ModelConfig:
    """Static metadata for a GGUF model on HuggingFace."""

    repo_id: str
    filename: str
    name: str
    size_gb: float


# Qwen3 4B Instruct (2507 refresh): a non-thinking instruct model that, in
# side-by-side testing, produced markedly more structured and faithful
# summaries than the previous Mistral 7B v0.2 default while being smaller
# (2.5 GB vs 4.4 GB) and faster. GGUFs are sourced from Unsloth, since the
# old TheBloke repos are no longer maintained.
DEFAULT_MODEL = ModelConfig(
    repo_id="unsloth/Qwen3-4B-Instruct-2507-GGUF",
    filename="Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    name="Qwen3 4B Instruct 2507",
    size_gb=2.5,
)

# Summarization tuning -------------------------------------------------------
# Chunks are cut by characters at the document's own measured chars-per-token
# (~3.4 for English, ~1.3 for Chinese, on the model's real tokenizer); this
# headroom absorbs density varying within a document, so few pieces need a
# second cut.
_CHUNK_HEADROOM = 0.9
# Tokens reserved for everything in a request that isn't the document itself
# (system prompt, instruction, the language directive stated twice, chat-template
# overhead).
_SCAFFOLD_TOKENS = 550
# Floor on the per-chunk token budget, so a tiny n_ctx can't yield degenerate
# one-character chunks.
_MIN_CHUNK_TOKENS = 512

_CLOSED_MESSAGE = "Summarizer has been closed; create a new instance."

_SYSTEM_PROMPT = (
    "You are a precise document-summarization assistant. Produce only the "
    "requested summary, based solely on the provided document. Do not add a "
    "preamble, a sign-off, or follow-up questions."
)

_SUMMARY_INSTRUCTIONS = {
    SUMMARY_TYPE_BRIEF: (
        "Summarize the document below in one concise paragraph (3-5 sentences), "
        "focusing on the main topic, key findings, and conclusions."
    ),
    SUMMARY_TYPE_DETAILED: (
        "Provide a detailed summary of the document below. Include the main topic "
        "and purpose, the key points and arguments, important findings or "
        "conclusions, and any significant methods or approaches mentioned."
    ),
    SUMMARY_TYPE_STRUCTURED: (
        "Analyze the document below and provide a structured summary with these "
        "sections, each on its own line:\n"
        "**Title/Topic:** What is this document about?\n"
        "**Purpose:** Why was this written?\n"
        "**Key Points:** Main arguments or findings, as bullet points.\n"
        "**Methods:** If applicable, how the work was conducted.\n"
        "**Conclusions:** The main takeaways.\n"
        "**Significance:** Why it matters."
    ),
}


# Output language ------------------------------------------------------------
# Every prompt here is written in English, which biases a multilingual model
# toward answering in English: a Chinese document carrying a few English
# acronyms was reported coming back summarized entirely in English, while the
# same document without them summarized in Chinese. Nothing previously told the
# model what language to answer in, so it inferred one from the prompt plus the
# document. Every request now states the language explicitly.
LANGUAGE_AUTO = "auto"

# Offered by the GUI picker and named in the CLI's --language help. Any other
# non-empty name is still accepted and passed to the model verbatim, so a
# language missing from this list is a one-word CLI flag away.
OUTPUT_LANGUAGES = (
    LANGUAGE_AUTO,
    "English",
    "Chinese",
    "Spanish",
    "French",
    "German",
    "Portuguese",
    "Italian",
    "Dutch",
    "Polish",
    "Russian",
    "Ukrainian",
    "Turkish",
    "Arabic",
    "Hindi",
    "Japanese",
    "Korean",
    "Vietnamese",
)

# A language name is interpolated straight into the prompt, so it is collapsed
# to one short line first — a pasted paragraph must not become instructions.
_MAX_LANGUAGE_LEN = 40

_AUTO_LANGUAGE_DIRECTIVE = (
    "Write the summary in the same language as the document. Isolated "
    "foreign-language names, acronyms, and technical terms do not change that: "
    "follow the language the body of the document is written in."
)


def normalize_language(language: str | None) -> str:
    """Coerce a user-supplied output language to a safe single-line value.

    Returns ``LANGUAGE_AUTO`` for anything empty or unset, so every caller can
    pass whatever it has (a CLI flag, a hand-edited settings file) without
    pre-validating.
    """
    if not language:
        return LANGUAGE_AUTO
    cleaned = " ".join(language.split())[:_MAX_LANGUAGE_LEN].strip()
    if not cleaned or cleaned.lower() == LANGUAGE_AUTO:
        return LANGUAGE_AUTO
    return cleaned


def _language_directive(language: str) -> str:
    """The sentence appended to a prompt to pin the summary's language."""
    language = normalize_language(language)
    if language == LANGUAGE_AUTO:
        return _AUTO_LANGUAGE_DIRECTIVE
    return (
        f"Write the summary in {language}, even when the document is written in another language."
    )


def _split_into_chunks(text: str, max_chars: int) -> list[str]:
    """Split `text` into chunks no larger than `max_chars`.

    Prefers paragraph boundaries (blank-line separated), keeping chunks
    coherent. A single paragraph longer than `max_chars` is hard-split as a
    last resort. Pure function (no model state) so it can be unit-tested
    directly.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        if len(para) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(para[i : i + max_chars] for i in range(0, len(para), max_chars))
            continue
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks


def _paragraph_chunks(text: str, max_chars: int) -> list[tuple[str, int]]:
    """``_split_into_chunks`` in ``Summarizer.chunks_within``'s pair shape.

    The prose path never reads the offsets, so they are all 0.
    """
    return [(chunk, 0) for chunk in _split_into_chunks(text, max_chars)]


# Structured, source-grounded summaries -------------------------------------- #
# These power the GUI's provenance feature: discrete points, each grounded in a
# source sentence. The plain ``summarize() -> str`` path above is untouched (the
# CLI depends on it); everything here is additive.

# Reasoning/grounding is more reliable with a strict JSON contract and
# near-deterministic sampling, so the structured path uses its own system
# prompt, a low temperature, and a fixed seed (distinct from the prose path).
_STRUCTURED_SYSTEM_PROMPT = (
    "You are a precise document-summarization assistant. Output ONLY valid JSON "
    "matching the requested shape, with no preamble or commentary. Ground every "
    'point in the document: each "quote" must be a sentence copied verbatim from '
    "the document that supports the point."
)
_STRUCTURED_SEED = 0
_STRUCTURED_TEMPERATURE = 0.1
# JSON output is denser than prose, so reserve more of the context window for
# the response than the prose path's _SCAFFOLD_TOKENS does.
_STRUCTURED_SCAFFOLD_TOKENS = 850
_DETAILED_POINT_COUNT = 3
# Structured-summary sections, in render order. CONCLUSIONS is a synthesis with
# no single supporting sentence, so it carries no quote.
_STRUCTURED_SECTIONS = ("PURPOSE", "METHOD", "RESULTS", "CONCLUSIONS")

_DETAILED_JSON_INSTRUCTION = (
    "Summarize the document as a JSON object of the form "
    '{"lead": "<one-sentence overview>", "points": [{"text": "<key point>", '
    '"quote": "<verbatim supporting sentence from the document>"}]}. '
    f"Provide exactly {_DETAILED_POINT_COUNT} points, each with a verbatim quote."
)
_STRUCTURED_JSON_INSTRUCTION = (
    "Summarize the document as a JSON object of the form "
    '{"sections": {"PURPOSE": [{"text": "...", "quote": "<verbatim sentence>"}], '
    '"METHOD": [...], "RESULTS": [...], "CONCLUSIONS": [{"text": "..."}]}}. '
    "Every PURPOSE, METHOD, and RESULTS point must include a verbatim quote from "
    "the document; CONCLUSIONS is a synthesis and needs no quote."
)


@dataclass(frozen=True)
class SummaryPoint:
    """One summary claim, optionally grounded in a source sentence."""

    text: str
    citation: SourceSpan | None = None


@dataclass(frozen=True)
class RunStats:
    """One summary request's cost, summed over every completion it issued.

    The three counter fields are ``None`` when the backend exposes no counters.
    """

    calls: int  # completions the request issued, every step
    truncated_calls: int  # completions ending finish_reason == "length"
    prompt_tokens: int | None  # prompt length summed over calls
    prompt_evaluated: int | None  # n_p_eval deltas; < prompt_tokens on a prefix hit
    prompt_seconds: float | None  # t_p_eval_ms deltas / 1000
    generated_tokens: int  # n_eval deltas; without counters, _count_tokens(content)
    generation_seconds: float | None  # t_eval_ms deltas / 1000
    wall_seconds: float  # request start -> return


@dataclass(frozen=True)
class _PerfSnapshot:
    """llama.cpp's cumulative perf counters at one instant."""

    n_p_eval: int
    n_eval: int
    t_p_eval_ms: float
    t_eval_ms: float


def _perf_delta(before: _PerfSnapshot, after: _PerfSnapshot) -> _PerfSnapshot:
    """What one completion added to the counters."""
    return _PerfSnapshot(
        after.n_p_eval - before.n_p_eval,
        after.n_eval - before.n_eval,
        after.t_p_eval_ms - before.t_p_eval_ms,
        after.t_eval_ms - before.t_eval_ms,
    )


@final
class _RunStatsAccumulator:
    """Sums one request's completions; ``finish`` freezes the result once."""

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._calls = 0
        self._truncated = 0
        self._prompt_tokens: int | None = 0
        self._counters = True
        self._prompt_evaluated = 0
        self._prompt_ms = 0.0
        self._generated = 0
        self._generation_ms = 0.0
        self._final: RunStats | None = None

    def add_call(
        self,
        *,
        finish_reason: object,
        perf: _PerfSnapshot | None,
        usage_prompt_tokens: int | None,
        n_tokens_after: int | None,
        count_generated: Callable[[], int],
    ) -> None:
        """Add one completion, aborted or not."""
        self._calls += 1
        if finish_reason == "length":
            self._truncated += 1
        if perf is not None:
            self._prompt_evaluated += perf.n_p_eval
            self._prompt_ms += perf.t_p_eval_ms
            self._generated += perf.n_eval
            self._generation_ms += perf.t_eval_ms
        else:
            self._counters = False
            self._generated += count_generated()
        prompt: int | None = None
        if usage_prompt_tokens is not None:
            prompt = usage_prompt_tokens
        elif n_tokens_after is not None and perf is not None:
            prompt = n_tokens_after - perf.n_eval
        if prompt is None or self._prompt_tokens is None:
            self._prompt_tokens = None
        else:
            self._prompt_tokens += prompt

    def finish(self) -> RunStats:
        """The request's ``RunStats``, built on the first call and reused after."""
        if self._final is None:
            counters = self._counters
            self._final = RunStats(
                calls=self._calls,
                truncated_calls=self._truncated,
                prompt_tokens=self._prompt_tokens,
                prompt_evaluated=self._prompt_evaluated if counters else None,
                prompt_seconds=self._prompt_ms / 1000 if counters else None,
                generated_tokens=self._generated,
                generation_seconds=self._generation_ms / 1000 if counters else None,
                wall_seconds=time.perf_counter() - self._start,
            )
        return self._final


@dataclass(frozen=True)
class StructuredSummary:
    """A structured summary the GUI can render with per-point provenance.

    - Brief: ``lead`` holds the single paragraph; ``points`` is empty.
    - Detailed: ``lead`` overview plus grounded ``points``.
    - Structured: ``sections`` keyed by ``_STRUCTURED_SECTIONS``.

    ``text`` is the plain-text rendering, kept so the CLI and the Save path can
    write a structured summary the same way they write a prose one.
    ``run_stats`` is what the request cost, set by ``Summarizer.summarize_structured``.
    """

    summary_type: str
    lead: str | None
    points: list[SummaryPoint]
    sections: dict[str, list[SummaryPoint]] | None
    text: str
    run_stats: RunStats | None = None


def _split_into_chunks_with_offsets(text: str, max_chars: int) -> list[tuple[str, int]]:
    """Like ``_split_into_chunks``, but each chunk carries its base offset.

    Returns ``(chunk_text, base_offset)`` pairs where
    ``text[base_offset : base_offset + len(chunk_text)] == chunk_text`` exactly,
    so a quote located at a position within a chunk maps back to the full
    document. Splits on sentence boundaries; a single sentence longer than
    ``max_chars`` is hard-sliced.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return [(text, 0)]

    sentences = split_sentences(text)
    if not sentences:
        return [(text[i : i + max_chars], i) for i in range(0, len(text), max_chars)]

    chunks: list[tuple[str, int]] = []
    chunk_start: int | None = None
    chunk_end = 0
    for sent_start, sent_end in sentences:
        if sent_end - sent_start > max_chars:
            if chunk_start is not None:
                chunks.append((text[chunk_start:chunk_end], chunk_start))
                chunk_start = None
            for i in range(sent_start, sent_end, max_chars):
                end = min(i + max_chars, sent_end)
                chunks.append((text[i:end], i))
            continue
        if chunk_start is None:
            chunk_start, chunk_end = sent_start, sent_end
        elif sent_end - chunk_start <= max_chars:
            chunk_end = sent_end
        else:
            chunks.append((text[chunk_start:chunk_end], chunk_start))
            chunk_start, chunk_end = sent_start, sent_end
    if chunk_start is not None:
        chunks.append((text[chunk_start:chunk_end], chunk_start))
    return chunks


def parse_structured_json(raw: str) -> dict[str, object] | None:
    """Parse a JSON object from a model response, tolerantly.

    Tries a direct parse, then decodes the first ``{...}`` object even when the
    model wraps it in prose (a common small-model failure mode), ignoring any
    trailing text. Returns ``None`` if nothing parses to a JSON object — the
    caller then degrades to a prose summary.
    """
    raw = raw.strip()
    if not raw:
        return None
    data: object
    try:
        data = cast("object", json.loads(raw))
    except (json.JSONDecodeError, ValueError):
        start = raw.find("{")
        if start == -1:
            return None
        try:
            data, _ = cast("tuple[object, int]", json.JSONDecoder().raw_decode(raw, start))
        except (json.JSONDecodeError, ValueError):
            return None
    return cast("dict[str, object]", data) if isinstance(data, dict) else None


def read_raw_points(value: object) -> list[tuple[str, str]]:
    """Coerce a parsed ``points`` array into ``(text, quote)`` pairs.

    Defensive against a small local model's malformed entries: non-string or
    empty ``text`` is dropped, and a non-string/``null`` ``quote`` is treated as
    absent (so ``null`` is never searched as the literal string ``"None"``).
    """
    if not isinstance(value, list):
        return []
    points: list[tuple[str, str]] = []
    for item in cast("list[object]", value):
        if not isinstance(item, dict):
            continue
        fields = cast("dict[str, object]", item)
        text = fields.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        quote = fields.get("quote")
        quote = quote.strip() if isinstance(quote, str) else ""
        points.append((text.strip(), quote))
    return points


def _ground_points(
    raw_points: list[tuple[str, str]],
    text: str,
    base: int,
    sentences: list[tuple[int, int]],
) -> list[SummaryPoint]:
    """Attach a source citation to each point by locating its quote in ``text``.

    Citations are shifted by ``base`` so their offsets index the full document
    even when ``text`` is a chunk. An unlocatable quote yields no citation.
    """
    points: list[SummaryPoint] = []
    for point_text, quote in raw_points:
        citation = locate_quote(quote, text, sentences) if quote else None
        if citation is not None and base:
            citation = SourceSpan(
                citation.start + base, citation.end + base, citation.quote, citation.score
            )
        points.append(SummaryPoint(text=point_text, citation=citation))
    return points


def _build_structured(
    parsed: dict[str, object], summary_type: str, text: str, base: int
) -> StructuredSummary:
    """Build a ``StructuredSummary`` from one parsed JSON response over ``text``.

    Points (and each section) are capped to ``_DETAILED_POINT_COUNT`` so a
    single pass and a map-reduce pass agree on the per-section bound.
    """
    sentences = split_sentences(text)
    if summary_type == SUMMARY_TYPE_DETAILED:
        lead = str(parsed.get("lead", "")).strip() or None
        points = _ground_points(read_raw_points(parsed.get("points")), text, base, sentences)[
            :_DETAILED_POINT_COUNT
        ]
        return _make_structured(SUMMARY_TYPE_DETAILED, lead, points, None)
    found = parsed.get("sections")
    raw_sections = cast("dict[str, object]", found) if isinstance(found, dict) else {}
    sections = {
        name: _ground_points(read_raw_points(raw_sections.get(name)), text, base, sentences)[
            :_DETAILED_POINT_COUNT
        ]
        for name in _STRUCTURED_SECTIONS
    }
    return _make_structured(SUMMARY_TYPE_STRUCTURED, None, [], sections)


def _log_run_stats(stats: RunStats) -> None:
    """Log one request's counts and times on a single line, never its content."""

    def count(value: int | None) -> str:
        return "n/a" if value is None else str(value)

    def seconds(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    def rate(tokens: int | None, elapsed: float | None) -> str:
        return "n/a" if tokens is None or not elapsed else f"{tokens / elapsed:.1f}"

    log_info(
        f"run_stats calls={stats.calls} truncated={stats.truncated_calls} "
        f"prompt_tokens={count(stats.prompt_tokens)} "
        f"prompt_evaluated={count(stats.prompt_evaluated)} "
        f"prompt_s={seconds(stats.prompt_seconds)} "
        f"prompt_tok_s={rate(stats.prompt_evaluated, stats.prompt_seconds)} "
        f"generated={stats.generated_tokens} "
        f"generation_s={seconds(stats.generation_seconds)} "
        f"gen_tok_s={rate(stats.generated_tokens, stats.generation_seconds)} "
        f"wall_s={stats.wall_seconds:.3f}"
    )
    log_debug(f"Memory usage: {get_memory_usage_mb()} MB")


def _render_structured(
    summary_type: str,
    lead: str | None,
    points: list[SummaryPoint],
    sections: dict[str, list[SummaryPoint]] | None,
) -> str:
    """Render a structured summary as plain text (for the CLI / Save path)."""
    lines: list[str] = []
    if lead:
        lines.append(lead)
    if summary_type == SUMMARY_TYPE_STRUCTURED and sections:
        for name in _STRUCTURED_SECTIONS:
            section_points = sections.get(name) or []
            if not section_points:
                continue
            if lines:
                lines.append("")
            lines.append(f"{name.title()}:")
            lines.extend(f"- {point.text}" for point in section_points)
    else:
        if points and lines:
            lines.append("")
        lines.extend(f"- {point.text}" for point in points)
    return "\n".join(lines).strip()


def _make_structured(
    summary_type: str,
    lead: str | None,
    points: list[SummaryPoint],
    sections: dict[str, list[SummaryPoint]] | None,
) -> StructuredSummary:
    """Assemble a ``StructuredSummary``, rendering its ``text`` form once.

    Keeps the plain-text ``text`` field in lockstep with the structured fields
    at every build site, so the two can't drift apart.
    """
    return StructuredSummary(
        summary_type,
        lead,
        points,
        sections,
        _render_structured(summary_type, lead, points, sections),
    )


def _select_round_robin(groups: list[list[SummaryPoint]], limit: int) -> list[SummaryPoint]:
    """Pick up to ``limit`` points, drawing across ``groups`` (chunks) in turn.

    Round-robin (first of each group, then second, ...) so a long document's
    summary is represented by all its chunks rather than only the earliest.
    """
    selected: list[SummaryPoint] = []
    depth = max((len(group) for group in groups), default=0)
    for i in range(depth):
        for group in groups:
            if i < len(group):
                selected.append(group[i])
                if len(selected) >= limit:
                    return selected
    return selected


def _is_empty_structured(summary: StructuredSummary) -> bool:
    """True when a built summary has no usable content (triggers fallback)."""
    if summary.summary_type == SUMMARY_TYPE_STRUCTURED:
        return not (summary.sections and any(summary.sections.values()))
    return not summary.points


def get_models_directory() -> Path:
    """Get the directory where models are stored."""
    return app_data_dir("models")


def is_model_downloaded(model_config: ModelConfig = DEFAULT_MODEL) -> bool:
    """Check if the model file exists locally."""
    return (get_models_directory() / model_config.filename).exists()


def get_model_path(model_config: ModelConfig = DEFAULT_MODEL) -> Path:
    """Get the full path to the model file."""
    return get_models_directory() / model_config.filename


def _build_progress_tqdm(callback: Callable[[float, str], None]) -> "type[tqdm[NoReturn]]":
    """Build a tqdm subclass that fires `callback` on each whole-MB step.

    huggingface_hub instantiates this for `hf_hub_download`. tqdm calls
    `update()` once per HTTP chunk (~thousands over a multi-GB download); we
    coalesce to one callback per integer megabyte so the GUI doesn't burn
    Tk main-loop wakeups on no-op text changes.
    """
    from tqdm.auto import tqdm

    # tqdm is generic only in its type stubs; at runtime it is not subscriptable.
    # NoReturn: the stubs' type for a bar built without an iterable, as
    # huggingface_hub builds it.
    base = cast("type[tqdm[NoReturn]]", tqdm)

    class _ProgressTqdm(base):
        _last_reported_mb: int = -1

        @override
        def update(self, n: float | None = 1) -> bool | None:
            ret = super().update(n)
            try:
                total = cast("float | None", self.total)
                done = cast("float", self.n)
                if not total or total <= 0:
                    return ret
                mb_done = int(done / (1024 * 1024))
                if mb_done == self._last_reported_mb:
                    return ret
                self._last_reported_mb = mb_done
                pct = (done / total) * 100.0
                mb_total = total / (1024 * 1024)
                callback(pct, f"Downloading {mb_done} / {mb_total:.0f} MB")
            except Exception:
                # Progress reporting must never break the download itself.
                logger = get_logger()
                logger.debug("Progress callback raised", exc_info=True)
            return ret

    return _ProgressTqdm


def download_model(
    model_config: ModelConfig = DEFAULT_MODEL,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[Path, str | None]:
    """Download the model from HuggingFace.

    Returns ``(model_path, error_message)``; ``error_message`` is None on
    success. ``progress_callback`` (if given) receives ``(percent, message)``
    updates as bytes arrive, plus a final 100% / "Download complete" call.
    """
    models_dir = get_models_directory()
    model_path = models_dir / model_config.filename

    log_info(f"Model download requested: {model_config.name}")
    log_debug(f"Model path: {model_path}")
    log_debug(f"Models directory: {models_dir}")

    if model_path.exists():
        file_size_gb = model_path.stat().st_size / (1024**3)
        log_info(f"Model already exists: {model_path.name} ({file_size_gb:.2f} GB)")
        if progress_callback:
            progress_callback(100.0, "Model already downloaded")
        return model_path, None

    try:
        log_info(f"Starting download: {model_config.repo_id}/{model_config.filename}")
        log_info(f"Expected size: ~{model_config.size_gb} GB")

        if progress_callback:
            progress_callback(
                0.0, f"Downloading {model_config.name} (~{model_config.size_gb} GB)..."
            )

        start_time = time.time()

        tqdm_class = _build_progress_tqdm(progress_callback) if progress_callback else None

        downloaded_path = hf_hub_download(
            repo_id=model_config.repo_id,
            filename=model_config.filename,
            local_dir=models_dir,
            tqdm_class=tqdm_class,
        )

        elapsed = time.time() - start_time
        file_size_gb = Path(downloaded_path).stat().st_size / (1024**3)
        log_info(f"Download complete: {file_size_gb:.2f} GB in {elapsed:.1f}s")

        if progress_callback:
            progress_callback(100.0, "Download complete")

        return Path(downloaded_path), None

    except Exception as e:
        error_msg = f"Failed to download model: {e!s}"
        logger = get_logger()
        logger.exception(error_msg)
        log_error(f"Exception type: {type(e).__name__}")
        if progress_callback:
            progress_callback(0.0, error_msg)
        return model_path, error_msg


def _reduce_structured(
    summary_type: str, partials: list[StructuredSummary]
) -> StructuredSummary | None:
    """Merge per-chunk summaries, drawing round-robin so every chunk is represented.

    ``None`` when no chunk produced a point, which tells the caller to fall back.
    """
    if summary_type == SUMMARY_TYPE_DETAILED:
        lead = next((partial.lead for partial in partials if partial.lead), None)
        groups = [partial.points for partial in partials if partial.points]
        points = _select_round_robin(groups, _DETAILED_POINT_COUNT)
        if not points:
            return None
        return _make_structured(SUMMARY_TYPE_DETAILED, lead, points, None)
    sections = {
        name: _select_round_robin(
            [
                partial.sections[name]
                for partial in partials
                if partial.sections and partial.sections.get(name)
            ],
            _DETAILED_POINT_COUNT,
        )
        for name in _STRUCTURED_SECTIONS
    }
    if not any(sections.values()):
        return None
    return _make_structured(SUMMARY_TYPE_STRUCTURED, None, [], sections)


@final
class Summarizer:
    """Handles text summarization using the local LLM."""

    # The open request's accumulator; a class default so every instance starts
    # without one.
    _run_stats: _RunStatsAccumulator | None = None

    def __init__(
        self,
        model_path: Path,
        n_ctx: int = 8192,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
    ) -> None:
        """Initialize the summarizer with a model.

        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context window size (default 8192). Documents longer than
                this are summarized via map-reduce chunking.
            n_threads: Number of CPU threads. `None` = auto (half of available
                cores). `0` means "let llama.cpp decide" and is passed through.
            n_gpu_layers: Model layers to offload to the GPU. `0` (default)
                keeps inference fully on the CPU for portability; `-1` offloads
                all layers. Ignored by CPU-only llama-cpp builds.
        """
        llama = _load_llama_cpp().Llama

        self.n_ctx = n_ctx
        cpu_count = os.cpu_count() or 8
        default_threads = max(4, cpu_count // 2)
        self.n_threads = default_threads if n_threads is None else n_threads
        self.n_gpu_layers = n_gpu_layers

        log_info("Initializing Summarizer")
        log_info(f"Model path: {model_path}")
        log_info(f"Context window: {n_ctx} tokens")
        log_info(f"CPU threads: {self.n_threads} of {cpu_count} available")
        log_info(
            f"GPU layers: {n_gpu_layers} ({'GPU offload enabled' if n_gpu_layers else 'CPU only'})"
        )
        log_debug(f"Memory before loading: {get_memory_usage_mb()} MB")

        start_time = time.time()

        self.llm: ChatModel | None = llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=self.n_threads,
            n_threads_batch=self.n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

        load_time = time.time() - start_time
        log_info(f"Model loaded successfully in {load_time:.2f}s")
        log_debug(f"Memory after loading: {get_memory_usage_mb()} MB")
        # Set during summarize(); read per-token to interrupt generation on Stop.
        self._cancel_check: Callable[[], bool] | None = None
        # Set during summarize(); read by every prompt builder. Instance state
        # rather than a threaded-through argument, mirroring _cancel_check.
        self._language: str = LANGUAGE_AUTO
        # ctypes callback objects must stay strongly referenced while llama.cpp
        # owns the callback pointer. These are cleared after each completion.
        self._llama_abort_callback: object | None = None
        self._llama_abort_callback_user_data: ctypes.c_void_p | None = None

    def _llama_context(self) -> object | None:
        """Return the low-level llama_context pointer, if this backend exposes it."""
        llm = self.llm
        if llm is None:
            return None
        try:
            ctx = cast("object | None", getattr(llm, "ctx", None))
        except Exception as exc:  # the optional binding varies by build
            logger = get_logger()
            logger.debug("Unable to read llama ctx property: %s", exc, exc_info=True)
            ctx = None
        if ctx is not None:
            return ctx
        try:
            internal_ctx: object = getattr(llm, "_ctx", None)
            if internal_ctx is not None:
                ctx = cast("object | None", getattr(internal_ctx, "ctx", None))
        except Exception as exc:  # the optional binding varies by build
            logger = get_logger()
            logger.debug("Unable to read llama _ctx property: %s", exc, exc_info=True)
            return None
        return ctx

    def _install_llama_abort_callback(self, check: Callable[[], bool]) -> bool:
        """Ask llama.cpp to abort the active decode when ``check`` turns true.

        This reaches into llama-cpp-python's optional low-level binding. If the
        package, callback type, setter, or context pointer is unavailable, the
        existing streaming cancellation path remains in force.
        """
        try:
            llama_cpp = _load_llama_cpp().llama_cpp
        except Exception as exc:  # the optional binding varies by build
            logger = get_logger()
            logger.debug("llama abort callback unavailable: %s", exc, exc_info=True)
            return False

        set_abort_callback = _binding(llama_cpp, "llama_set_abort_callback")
        callback_type = _binding(llama_cpp, "ggml_abort_callback")
        ctx = self._llama_context()
        if not callable(set_abort_callback) or not callable(callback_type) or ctx is None:
            return False

        def abort_callback(_: object) -> bool:
            try:
                return bool(check())
            except Exception:
                # Runs inside llama.cpp's decode loop: an exception must not reach C.
                logger = get_logger()
                logger.debug("Cancel check raised inside llama abort callback", exc_info=True)
                return False

        c_callback = callback_type(abort_callback)
        user_data = ctypes.c_void_p()
        self._llama_abort_callback = c_callback
        self._llama_abort_callback_user_data = user_data
        try:
            _ = set_abort_callback(ctx, c_callback, user_data)
        except Exception as exc:  # the optional binding varies by build
            self._llama_abort_callback = None
            self._llama_abort_callback_user_data = None
            logger = get_logger()
            logger.debug("Unable to install llama abort callback: %s", exc, exc_info=True)
            return False
        return True

    def _clear_llama_abort_callback(self) -> None:
        """Remove any callback installed by ``_install_llama_abort_callback``."""
        if getattr(self, "_llama_abort_callback", None) is None:
            return
        try:
            llama_cpp = _load_llama_cpp().llama_cpp
            set_abort_callback = _binding(llama_cpp, "llama_set_abort_callback")
            callback_type = _binding(llama_cpp, "ggml_abort_callback")
            ctx = self._llama_context()
            if callable(set_abort_callback) and callable(callback_type) and ctx is not None:
                _ = set_abort_callback(ctx, callback_type(), ctypes.c_void_p())
        except Exception as exc:  # the optional binding varies by build
            logger = get_logger()
            logger.debug("Unable to clear llama abort callback: %s", exc, exc_info=True)
        finally:
            self._llama_abort_callback = None
            self._llama_abort_callback_user_data = None

    @contextmanager
    def _llama_abort_callback_guard(self, check: Callable[[], bool]) -> Generator[None, None, None]:
        """Temporarily install llama.cpp decode cancellation for one completion."""
        installed = self._install_llama_abort_callback(check)
        try:
            yield
        finally:
            if installed:
                self._clear_llama_abort_callback()

    def _complete(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        """One chat completion.

        When a cancel-check is active (the GUI Stop path) the response is
        streamed and a llama.cpp abort callback is installed so prompt
        processing and generation can both stop soon after the flag is set.
        ``create_chat_completion`` has no stopping_criteria parameter, so these
        hooks are layered around the chat call instead.
        Otherwise a single blocking call is used.
        """
        llm = self.llm
        if llm is None:
            raise RuntimeError(_CLOSED_MESSAGE)
        check = self._cancel_check
        if check is not None:
            _raise_if_cancelled(check)
        before = self.perf_snapshot()
        finish_reason: str | None = None
        usage: CompletionUsage | None = None
        content = ""
        try:
            if check is None:
                response = llm.create_chat_completion(messages=messages, stream=False, **kwargs)
                choice = response["choices"][0]
                finish_reason = choice.get("finish_reason")
                usage = response.get("usage")
                content = str(choice["message"].get("content") or "")
                return content.strip()
            try:
                with self._llama_abort_callback_guard(check):
                    for chunk in llm.create_chat_completion(
                        messages=messages, stream=True, **kwargs
                    ):
                        _raise_if_cancelled(check)
                        choice = chunk["choices"][0]
                        finish_reason = choice.get("finish_reason") or finish_reason
                        delta = (choice.get("delta") or {}).get("content")
                        if delta:
                            content += delta
                    _raise_if_cancelled(check)
            except SummarizationCancelledError:
                raise
            except Exception as exc:
                if check():
                    raise SummarizationCancelledError from exc
                raise
            return content.strip()
        finally:
            self._record_call(before, finish_reason, usage, content)

    @contextmanager
    def _request_stats(self) -> Generator[_RunStatsAccumulator, None, None]:
        """The request's accumulator: the outermost entry point opens it, nested calls share it."""
        current = self._run_stats
        if current is not None:
            yield current
            return
        stats = _RunStatsAccumulator()
        self._run_stats = stats
        try:
            yield stats
        finally:
            self._run_stats = None
            _log_run_stats(stats.finish())

    def perf_snapshot(self) -> _PerfSnapshot | None:
        """llama.cpp's perf counters now, or ``None`` when the backend has none."""
        ctx = self._llama_context()
        if ctx is None:
            return None
        try:
            llama_cpp = _load_llama_cpp().llama_cpp
            perf_context = _binding(llama_cpp, "llama_perf_context")
            if not callable(perf_context):
                return None
            data = cast("_PerfCounters", perf_context(ctx))
            return _PerfSnapshot(
                int(data.n_p_eval),
                int(data.n_eval),
                float(data.t_p_eval_ms),
                float(data.t_eval_ms),
            )
        except Exception as exc:  # the optional binding varies by build
            logger = get_logger()
            logger.debug("llama perf counters unavailable: %s", exc, exc_info=True)
            return None

    def _record_call(
        self,
        before: _PerfSnapshot | None,
        finish_reason: str | None,
        usage: "Mapping[str, object] | None",
        content: str,
    ) -> None:
        """Add one completion to the open request's stats, if a request is open."""
        stats = self._run_stats
        if stats is None:
            return
        # Runs in _complete's finally: raising here would mask the call's own error.
        try:
            after = self.perf_snapshot() if before is not None else None
            perf = _perf_delta(before, after) if before is not None and after is not None else None
            usage_prompt = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            n_tokens: object = getattr(self.llm, "n_tokens", None)
            stats.add_call(
                finish_reason=finish_reason,
                perf=perf,
                usage_prompt_tokens=usage_prompt if isinstance(usage_prompt, int) else None,
                n_tokens_after=n_tokens if isinstance(n_tokens, int) else None,
                count_generated=lambda: self._count_tokens(content) if content else 0,
            )
        except Exception:
            logger = get_logger()
            logger.debug("Unable to record run stats", exc_info=True)

    def _count_tokens(self, text: str) -> int:
        """Tokens ``text`` occupies in this model's context window."""
        if self.llm is None:
            raise RuntimeError(_CLOSED_MESSAGE)
        return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False))

    def chunks_within(
        self,
        text: str,
        budget_tokens: int,
        split: Callable[[str, int], list[tuple[str, int]]],
    ) -> list[tuple[str, int]]:
        """Cut ``text`` into ``(chunk, offset)`` pieces of at most ``budget_tokens`` each.

        Characters per token vary ~2.6x between English and Chinese, so no
        fixed ratio fits both: the character budget handed to ``split`` comes
        from this text's measured density, and any piece still over budget is
        cut again at its own density. Each cut is below ``len(text)``, so this
        terminates.
        """
        tokens = self._count_tokens(text)
        if tokens <= budget_tokens:
            return [(text, 0)]
        max_chars = max(int(len(text) * budget_tokens / tokens * _CHUNK_HEADROOM), 1)
        pieces: list[tuple[str, int]] = []
        for chunk, base in split(text, max_chars):
            pieces.extend(
                (piece, base + offset)
                for piece, offset in self.chunks_within(chunk, budget_tokens, split)
            )
        return pieces

    def summarize(
        self,
        text: str,
        summary_type: str = SUMMARY_TYPE_DETAILED,
        max_tokens: int = 1024,
        should_cancel: Callable[[], bool] | None = None,
        language: str = LANGUAGE_AUTO,
    ) -> str:
        """Generate a summary of the given text.

        Documents that fit the context window are summarized in a single pass.
        Longer documents are handled with a map-reduce strategy: the text is
        split into context-sized chunks, each is summarized, and the partial
        summaries are synthesized into one — so no content is silently dropped.

        Args:
            text: The text to summarize
            summary_type: one of `SUMMARY_TYPES`. Unknown values fall back to
                "detailed" to preserve the prior tolerant behavior.
            max_tokens: Maximum tokens in the response
            should_cancel: Polled between and during model calls; returning
                true stops the run with ``SummarizationCancelledError``.
            language: Output language. ``LANGUAGE_AUTO`` (the default) tells the
                model to match the document; any other name pins the summary to
                that language.

        Returns:
            The generated summary
        """
        if self.llm is None:
            raise RuntimeError(_CLOSED_MESSAGE)

        with self._request_stats():
            self._cancel_check = should_cancel
            self._language = normalize_language(language)
            text = text.strip()
            log_info(
                f"Starting summarization: type={summary_type}, "
                f"language={self._language}, input_chars={len(text)}"
            )

            budget_tokens = max(self.n_ctx - max_tokens - _SCAFFOLD_TOKENS, _MIN_CHUNK_TOKENS)
            chunks = [
                chunk for chunk, _ in self.chunks_within(text, budget_tokens, _paragraph_chunks)
            ]

            if len(chunks) == 1:
                return self._summarize_once(text, summary_type, max_tokens)

            # Document exceeds the context budget: map-reduce.
            log_info(f"Document exceeds context budget; summarizing in {len(chunks)} chunks")

            partials: list[str] = []
            for i, chunk in enumerate(chunks, 1):
                _raise_if_cancelled(should_cancel)
                log_info(f"Summarizing chunk {i}/{len(chunks)} ({len(chunk)} chars)")
                partials.append(self._summarize_once(chunk, summary_type, max_tokens))
            combined = "\n\n".join(partials)

            # Reduce. If the combined partials still overflow (a very long
            # document with many chunks), recurse — each pass strictly shrinks the
            # text, so this terminates.
            if self._count_tokens(combined) > budget_tokens:
                log_info("Combined section summaries still exceed budget; reducing again")
                return self.summarize(combined, summary_type, max_tokens, should_cancel, language)

            return self._synthesize(combined, summary_type, max_tokens)

    def summarize_structured(
        self,
        text: str,
        summary_type: str = SUMMARY_TYPE_DETAILED,
        max_tokens: int = 1024,
        should_cancel: Callable[[], bool] | None = None,
        language: str = LANGUAGE_AUTO,
    ) -> StructuredSummary:
        """Summarize into discrete, source-grounded points for the GUI.

        Each point carries a ``SourceSpan`` citation when its model-emitted
        quote can be re-located in the source (offsets into the full document),
        otherwise ``None`` — citations are never fabricated. Brief is a single
        plain paragraph (no provenance). Any JSON-parse failure degrades to the
        plain ``summarize`` output, so this never hard-fails. ``summarize`` (the
        ``-> str`` API the CLI uses) is unchanged.
        """
        if self.llm is None:
            raise RuntimeError(_CLOSED_MESSAGE)

        with self._request_stats() as stats:
            # NB: do not strip — citation offsets must index the caller's exact text
            # (extracted documents commonly start with whitespace). split_sentences
            # already ignores leading/trailing whitespace for grounding.
            if summary_type == SUMMARY_TYPE_BRIEF:
                lead = self.summarize(text, SUMMARY_TYPE_BRIEF, max_tokens, should_cancel, language)
                brief = StructuredSummary(SUMMARY_TYPE_BRIEF, lead or None, [], None, lead)
                return replace(brief, run_stats=stats.finish())

            if summary_type not in (SUMMARY_TYPE_DETAILED, SUMMARY_TYPE_STRUCTURED):
                summary_type = SUMMARY_TYPE_DETAILED

            log_info(
                f"Starting structured summarization: type={summary_type}, "
                f"language={normalize_language(language)}, input_chars={len(text)}"
            )
            try:
                result = self._summarize_structured(
                    text, summary_type, max_tokens, should_cancel, language
                )
            except SummarizationCancelledError:
                raise  # a clean stop must not fall back to more inference
            except Exception:
                logger = get_logger()
                logger.warning(
                    "Structured summarization failed; falling back to prose", exc_info=True
                )
                result = None
            if result is None:
                fallback = self._fallback_structured(text, summary_type, max_tokens, language)
                return replace(fallback, run_stats=stats.finish())
            return replace(result, run_stats=stats.finish())

    def _summarize_structured(
        self,
        text: str,
        summary_type: str,
        max_tokens: int,
        should_cancel: Callable[[], bool] | None = None,
        language: str = LANGUAGE_AUTO,
    ) -> StructuredSummary | None:
        """Run the structured path; ``None`` signals the caller to fall back."""
        self._cancel_check = should_cancel
        self._language = normalize_language(language)
        budget_tokens = max(
            self.n_ctx - max_tokens - _STRUCTURED_SCAFFOLD_TOKENS, _MIN_CHUNK_TOKENS
        )
        chunks = self.chunks_within(text, budget_tokens, _split_into_chunks_with_offsets)
        instruction = (
            _DETAILED_JSON_INSTRUCTION
            if summary_type == SUMMARY_TYPE_DETAILED
            else _STRUCTURED_JSON_INSTRUCTION
        )

        if len(chunks) == 1:
            parsed = parse_structured_json(
                self._chat_json(f"{instruction}\n\nDocument:\n{text}", max_tokens)
            )
            if parsed is None:
                return None
            built = _build_structured(parsed, summary_type, text, 0)
            return None if _is_empty_structured(built) else built

        # Map each chunk (citations already at global offsets), keeping per-chunk
        # groups. Reduce by drawing across chunks round-robin, so a long document
        # draws from its whole body rather than only the first chunk.
        partials: list[StructuredSummary] = []
        for chunk_text, base in chunks:
            _raise_if_cancelled(should_cancel)
            parsed = parse_structured_json(
                self._chat_json(f"{instruction}\n\nDocument:\n{chunk_text}", max_tokens)
            )
            if parsed is not None:
                partials.append(_build_structured(parsed, summary_type, chunk_text, base))
        return _reduce_structured(summary_type, partials)

    def _chat_json(self, user_content: str, max_tokens: int) -> str:
        """One chat completion constrained to a JSON object, low-temperature."""
        directive = _language_directive(self._language)
        return self._complete(
            [
                {"role": "system", "content": f"{_STRUCTURED_SYSTEM_PROMPT} {directive}"},
                {"role": "user", "content": f"{user_content}\n\n{directive}"},
            ],
            max_tokens=max_tokens,
            temperature=_STRUCTURED_TEMPERATURE,
            top_p=0.9,
            seed=_STRUCTURED_SEED,
            response_format={"type": "json_object"},
        )

    def _fallback_structured(
        self, text: str, summary_type: str, max_tokens: int, language: str = LANGUAGE_AUTO
    ) -> StructuredSummary:
        """Degrade to a plain prose summary wrapped as a ``StructuredSummary``."""
        summary = self.summarize(text, summary_type, max_tokens, language=language)
        return StructuredSummary(summary_type, summary or None, [], None, summary)

    def _summarize_once(self, text: str, summary_type: str, max_tokens: int) -> str:
        """Summarize text that fits within the context window in one call."""
        instruction = _SUMMARY_INSTRUCTIONS.get(
            summary_type, _SUMMARY_INSTRUCTIONS[SUMMARY_TYPE_DETAILED]
        )
        return self._chat(f"{instruction}\n\nDocument:\n{text}", max_tokens)

    def _synthesize(self, partials_text: str, summary_type: str, max_tokens: int) -> str:
        """Combine per-chunk summaries into one coherent summary."""
        instruction = _SUMMARY_INSTRUCTIONS.get(
            summary_type, _SUMMARY_INSTRUCTIONS[SUMMARY_TYPE_DETAILED]
        )
        prompt = (
            "Below are summaries of consecutive sections of a single document. "
            "Combine them into one coherent summary, removing redundancy.\n\n"
            f"{instruction}\n\nSection summaries:\n{partials_text}"
        )
        return self._chat(prompt, max_tokens)

    def _chat(self, user_content: str, max_tokens: int) -> str:
        """Run one chat completion, applying the model's own chat template.

        Using ``create_chat_completion`` (rather than a raw prompt string) lets
        llama.cpp wrap the message in whatever instruction format the loaded
        model expects, so swapping models doesn't require hand-editing prompts.

        The language directive is stated twice — in the system turn and as the
        last line of the user turn. A 4B model reliably honours the instruction
        nearest the end of the prompt; the system copy survives a long document
        pushing that line far from the start.
        """
        directive = _language_directive(self._language)
        return self._complete(
            [
                {"role": "system", "content": f"{_SYSTEM_PROMPT} {directive}"},
                {"role": "user", "content": f"{user_content}\n\n{directive}"},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9,
        )

    def close(self) -> None:
        """Release the underlying llama.cpp model.

        Safe to call multiple times. After close(), `summarize()` raises;
        instantiate a new Summarizer to reload. Prefer this to relying on
        `__del__`, which is unreliable during interpreter shutdown.
        """
        self._clear_llama_abort_callback()
        if self.llm is not None:
            del self.llm
            self.llm = None

    def __enter__(self) -> "Summarizer":
        """Return the summarizer; leaving the block closes it."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Close the summarizer, freeing the model."""
        self.close()
