"""
Model Manager Module
Handles downloading, loading, and running the local LLM.
"""

import functools
import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from .logger import get_memory_usage_mb, log_debug, log_error, log_info, log_warning
from .paths import app_data_dir
from .provenance import SourceSpan, locate_quote, split_sentences

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
        from llama_cpp import llama_supports_gpu_offload

        return bool(llama_supports_gpu_offload())
    except Exception:
        return False


class SummarizationCancelledError(Exception):
    """Raised between chunks when the caller requests a clean stop."""


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
# Conservative chars-per-token estimate, used to keep prompts within the
# context window without taking on a tokenizer dependency.
_CHARS_PER_TOKEN = 3
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
class StructuredSummary:
    """A structured summary the GUI can render with per-point provenance.

    - Brief: ``lead`` holds the single paragraph; ``points`` is empty.
    - Detailed: ``lead`` overview plus grounded ``points``.
    - Structured: ``sections`` keyed by ``_STRUCTURED_SECTIONS``.

    ``text`` is the plain-text rendering, kept so the CLI and the Save path can
    write a structured summary the same way they write a prose one.
    """

    summary_type: str
    lead: str | None
    points: list[SummaryPoint]
    sections: dict[str, list[SummaryPoint]] | None
    text: str


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


def _parse_structured_json(raw: str) -> dict[str, Any] | None:
    """Parse a JSON object from a model response, tolerantly.

    Tries a direct parse, then decodes the first ``{...}`` object even when the
    model wraps it in prose (a common small-model failure mode), ignoring any
    trailing text. Returns ``None`` if nothing parses to a JSON object — the
    caller then degrades to a prose summary.
    """
    raw = raw.strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        start = raw.find("{")
        if start == -1:
            return None
        try:
            data, _ = json.JSONDecoder().raw_decode(raw, start)
        except (json.JSONDecodeError, ValueError):
            return None
    return data if isinstance(data, dict) else None


def _read_raw_points(value: Any) -> list[tuple[str, str]]:
    """Coerce a parsed ``points`` array into ``(text, quote)`` pairs.

    Defensive against a small local model's malformed entries: non-string or
    empty ``text`` is dropped, and a non-string/``null`` ``quote`` is treated as
    absent (so ``null`` is never searched as the literal string ``"None"``).
    """
    if not isinstance(value, list):
        return []
    points: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        quote = item.get("quote")
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
    parsed: dict[str, Any], summary_type: str, text: str, base: int
) -> StructuredSummary:
    """Build a ``StructuredSummary`` from one parsed JSON response over ``text``.

    Points (and each section) are capped to ``_DETAILED_POINT_COUNT`` so a
    single pass and a map-reduce pass agree on the per-section bound.
    """
    sentences = split_sentences(text)
    if summary_type == SUMMARY_TYPE_DETAILED:
        lead = str(parsed.get("lead", "")).strip() or None
        points = _ground_points(_read_raw_points(parsed.get("points")), text, base, sentences)[
            :_DETAILED_POINT_COUNT
        ]
        return _make_structured(SUMMARY_TYPE_DETAILED, lead, points, None)
    raw_sections = parsed.get("sections")
    raw_sections = raw_sections if isinstance(raw_sections, dict) else {}
    sections = {
        name: _ground_points(_read_raw_points(raw_sections.get(name)), text, base, sentences)[
            :_DETAILED_POINT_COUNT
        ]
        for name in _STRUCTURED_SECTIONS
    }
    return _make_structured(SUMMARY_TYPE_STRUCTURED, None, [], sections)


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


def _build_progress_tqdm(callback: Callable[[float, str], None]) -> type[Any]:
    """Build a tqdm subclass that fires `callback` on each whole-MB step.

    huggingface_hub instantiates this for `hf_hub_download`. tqdm calls
    `update()` once per HTTP chunk (~thousands over a multi-GB download); we
    coalesce to one callback per integer megabyte so the GUI doesn't burn
    Tk main-loop wakeups on no-op text changes.
    """
    from tqdm.auto import tqdm as _BaseTqdm  # noqa: N812

    class _ProgressTqdm(_BaseTqdm):  # type: ignore[misc]
        _last_reported_mb = -1

        def update(self, n: int = 1) -> Any:
            ret = super().update(n)
            try:
                if not self.total or self.total <= 0:
                    return ret
                mb_done = int(self.n / (1024 * 1024))
                if mb_done == self._last_reported_mb:
                    return ret
                self._last_reported_mb = mb_done
                pct = (self.n / self.total) * 100.0
                mb_total = self.total / (1024 * 1024)
                callback(pct, f"Downloading {mb_done} / {mb_total:.0f} MB")
            except Exception as exc:
                # Progress reporting must never break the download itself.
                log_debug(f"Progress callback raised: {exc!s}")
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
        log_error(error_msg)
        log_error(f"Exception type: {type(e).__name__}")
        if progress_callback:
            progress_callback(0.0, error_msg)
        return model_path, error_msg


class Summarizer:
    """Handles text summarization using the local LLM."""

    def __init__(
        self,
        model_path: Path,
        n_ctx: int = 8192,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
    ):
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
        from llama_cpp import Llama

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

        self.llm: Llama | None = Llama(
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
        self._llama_abort_callback: Any | None = None
        self._llama_abort_callback_user_data: Any | None = None

    def _llama_context(self) -> Any | None:
        """Return the low-level llama_context pointer, if this backend exposes it."""
        llm = self.llm
        if llm is None:
            return None
        try:
            ctx = getattr(llm, "ctx", None)
        except Exception as exc:
            log_debug(f"Unable to read llama ctx property: {exc!s}")
            ctx = None
        if ctx is not None:
            return ctx
        try:
            internal_ctx = getattr(llm, "_ctx", None)
            return getattr(internal_ctx, "ctx", None) if internal_ctx is not None else None
        except Exception as exc:
            log_debug(f"Unable to read llama _ctx property: {exc!s}")
            return None

    def _install_llama_abort_callback(self, check: Callable[[], bool]) -> bool:
        """Ask llama.cpp to abort the active decode when ``check`` turns true.

        This reaches into llama-cpp-python's optional low-level binding. If the
        package, callback type, setter, or context pointer is unavailable, the
        existing streaming cancellation path remains in force.
        """
        try:
            import ctypes

            from llama_cpp import llama_cpp
        except Exception as exc:
            log_debug(f"llama abort callback unavailable: {exc!s}")
            return False

        set_abort_callback = getattr(llama_cpp, "llama_set_abort_callback", None)
        callback_type = getattr(llama_cpp, "ggml_abort_callback", None)
        ctx = self._llama_context()
        if not callable(set_abort_callback) or not callable(callback_type) or ctx is None:
            return False

        def abort_callback(_: Any) -> bool:
            try:
                return bool(check())
            except Exception as exc:
                log_debug(f"Cancel check raised inside llama abort callback: {exc!s}")
                return False

        c_callback = callback_type(abort_callback)
        user_data = ctypes.c_void_p()
        self._llama_abort_callback = c_callback
        self._llama_abort_callback_user_data = user_data
        try:
            set_abort_callback(ctx, c_callback, user_data)
        except Exception as exc:
            self._llama_abort_callback = None
            self._llama_abort_callback_user_data = None
            log_debug(f"Unable to install llama abort callback: {exc!s}")
            return False
        return True

    def _clear_llama_abort_callback(self) -> None:
        """Remove any callback installed by ``_install_llama_abort_callback``."""
        if getattr(self, "_llama_abort_callback", None) is None:
            return
        try:
            import ctypes

            from llama_cpp import llama_cpp

            set_abort_callback = getattr(llama_cpp, "llama_set_abort_callback", None)
            callback_type = getattr(llama_cpp, "ggml_abort_callback", None)
            ctx = self._llama_context()
            if callable(set_abort_callback) and callable(callback_type) and ctx is not None:
                set_abort_callback(ctx, callback_type(), ctypes.c_void_p())
        except Exception as exc:
            log_debug(f"Unable to clear llama abort callback: {exc!s}")
        finally:
            self._llama_abort_callback = None
            self._llama_abort_callback_user_data = None

    @contextmanager
    def _llama_abort_callback_guard(self, check: Callable[[], bool]) -> Iterator[None]:
        """Temporarily install llama.cpp decode cancellation for one completion."""
        installed = self._install_llama_abort_callback(check)
        try:
            yield
        finally:
            if installed:
                self._clear_llama_abort_callback()

    def _complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
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
        if check is None:
            response = llm.create_chat_completion(messages=messages, **kwargs)
            return str(response["choices"][0]["message"].get("content") or "").strip()
        _raise_if_cancelled(check)
        content = ""
        try:
            with self._llama_abort_callback_guard(check):
                for chunk in llm.create_chat_completion(messages=messages, stream=True, **kwargs):
                    _raise_if_cancelled(check)
                    delta = (chunk["choices"][0].get("delta") or {}).get("content")
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
            language: Output language. ``LANGUAGE_AUTO`` (the default) tells the
                model to match the document; any other name pins the summary to
                that language.

        Returns:
            The generated summary
        """
        if self.llm is None:
            raise RuntimeError(_CLOSED_MESSAGE)

        self._cancel_check = should_cancel
        self._language = normalize_language(language)
        text = text.strip()
        log_info(
            f"Starting summarization: type={summary_type}, "
            f"language={self._language}, input_chars={len(text)}"
        )

        budget_tokens = max(self.n_ctx - max_tokens - _SCAFFOLD_TOKENS, _MIN_CHUNK_TOKENS)
        budget_chars = budget_tokens * _CHARS_PER_TOKEN

        if len(text) <= budget_chars:
            start_time = time.time()
            summary = self._summarize_once(text, summary_type, max_tokens)
            self._log_speed(summary, time.time() - start_time)
            return summary

        # Document exceeds the context budget: map-reduce.
        chunks = _split_into_chunks(text, budget_chars)
        log_info(f"Document exceeds context budget; summarizing in {len(chunks)} chunks")
        start_time = time.time()

        partials = []
        for i, chunk in enumerate(chunks, 1):
            _raise_if_cancelled(should_cancel)
            log_info(f"Summarizing chunk {i}/{len(chunks)} ({len(chunk)} chars)")
            partials.append(self._summarize_once(chunk, summary_type, max_tokens))
        combined = "\n\n".join(partials)

        # Reduce. If the combined partials still overflow (a very long
        # document with many chunks), recurse — each pass strictly shrinks the
        # text, so this terminates.
        if len(combined) > budget_chars:
            log_info("Combined section summaries still exceed budget; reducing again")
            return self.summarize(combined, summary_type, max_tokens, should_cancel, language)

        summary = self._synthesize(combined, summary_type, max_tokens)
        self._log_speed(summary, time.time() - start_time)
        return summary

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

        # NB: do not strip — citation offsets must index the caller's exact text
        # (extracted documents commonly start with whitespace). split_sentences
        # already ignores leading/trailing whitespace for grounding.
        if summary_type == SUMMARY_TYPE_BRIEF:
            lead = self.summarize(text, SUMMARY_TYPE_BRIEF, max_tokens, should_cancel, language)
            return StructuredSummary(SUMMARY_TYPE_BRIEF, lead or None, [], None, lead)

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
        except Exception as exc:
            log_warning(f"Structured summarization failed ({exc!s}); falling back to prose")
            result = None
        if result is None:
            return self._fallback_structured(text, summary_type, max_tokens, language)
        return result

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
        budget_chars = (
            max(self.n_ctx - max_tokens - _STRUCTURED_SCAFFOLD_TOKENS, _MIN_CHUNK_TOKENS)
            * _CHARS_PER_TOKEN
        )
        instruction = (
            _DETAILED_JSON_INSTRUCTION
            if summary_type == SUMMARY_TYPE_DETAILED
            else _STRUCTURED_JSON_INSTRUCTION
        )

        if len(text) <= budget_chars:
            parsed = _parse_structured_json(
                self._chat_json(f"{instruction}\n\nDocument:\n{text}", max_tokens)
            )
            if parsed is None:
                return None
            built = _build_structured(parsed, summary_type, text, 0)
            return None if _is_empty_structured(built) else built

        # Map each chunk (citations already at global offsets), keeping per-chunk
        # groups. Reduce by drawing across chunks round-robin, so a long document
        # draws from its whole body rather than only the first chunk.
        lead: str | None = None
        detailed_groups: list[list[SummaryPoint]] = []
        section_groups: dict[str, list[list[SummaryPoint]]] = {
            name: [] for name in _STRUCTURED_SECTIONS
        }
        for chunk_text, base in _split_into_chunks_with_offsets(text, budget_chars):
            _raise_if_cancelled(should_cancel)
            parsed = _parse_structured_json(
                self._chat_json(f"{instruction}\n\nDocument:\n{chunk_text}", max_tokens)
            )
            if parsed is None:
                continue
            partial = _build_structured(parsed, summary_type, chunk_text, base)
            if lead is None and partial.lead:
                lead = partial.lead
            if partial.points:
                detailed_groups.append(partial.points)
            if partial.sections:
                for name in _STRUCTURED_SECTIONS:
                    section = partial.sections.get(name)
                    if section:
                        section_groups[name].append(section)

        if summary_type == SUMMARY_TYPE_DETAILED:
            points = _select_round_robin(detailed_groups, _DETAILED_POINT_COUNT)
            if not points:
                return None
            return _make_structured(SUMMARY_TYPE_DETAILED, lead, points, None)
        sections = {
            name: _select_round_robin(groups, _DETAILED_POINT_COUNT)
            for name, groups in section_groups.items()
        }
        if not any(sections.values()):
            return None
        return _make_structured(SUMMARY_TYPE_STRUCTURED, None, [], sections)

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

    def _log_speed(self, summary: str, elapsed: float) -> None:
        approx_tokens = max(len(summary) // 4, 1)
        tokens_per_sec = approx_tokens / elapsed if elapsed > 0 else 0
        log_info(
            f"Summary generated in {elapsed:.2f}s "
            f"(~{approx_tokens} tokens, {tokens_per_sec:.1f} tok/s)"
        )
        log_debug(f"Memory usage: {get_memory_usage_mb()} MB")

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
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
