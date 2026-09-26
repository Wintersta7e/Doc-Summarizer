"""Tests for structured, source-grounded summaries.

`Summarizer.summarize_structured` returns discrete points, each grounded in a
source sentence (model-cited quote, verified by re-locating it in the source).
Exercised against a fake ``llm`` that returns canned JSON, so no llama-cpp.
"""

from __future__ import annotations

import json

import pytest

from docsummarizer import model_manager
from docsummarizer.model_manager import (
    SUMMARY_TYPE_BRIEF,
    SUMMARY_TYPE_DETAILED,
    SUMMARY_TYPE_STRUCTURED,
    StructuredSummary,
    Summarizer,
    SummaryPoint,
)
from docsummarizer.provenance import SourceSpan

_SOURCE = (
    "The Transformer is a model architecture eschewing recurrence entirely. "
    "It relies on self-attention to draw global dependencies between tokens. "
    "On English-to-German translation it reaches 28.4 BLEU, a new state of the art. "
    "Training took three and a half days on eight GPUs."
)


class _JsonLLM:
    """Fake llm returning queued payloads (last repeats) from create_chat_completion."""

    def __init__(self, *payloads: str) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def tokenize(self, text: bytes, add_bos: bool = True) -> list[int]:
        return [0] * (len(text) // 3)  # ~3 UTF-8 bytes per token, like the real one

    def create_chat_completion(self, *, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        idx = min(len(self.calls) - 1, len(self.payloads) - 1)
        return {"choices": [{"message": {"content": self.payloads[idx]}}]}


def _shell(fake_llm, n_ctx: int = 8192) -> Summarizer:
    s = object.__new__(Summarizer)
    s.llm = fake_llm
    s.n_ctx = n_ctx
    return s


def _detailed_payload() -> str:
    return json.dumps(
        {
            "lead": "The Transformer is an attention-only architecture.",
            "points": [
                {
                    "text": "It drops recurrence.",
                    "quote": "The Transformer is a model architecture eschewing recurrence entirely.",
                },
                {
                    "text": "It uses self-attention.",
                    "quote": "It relies on self-attention to draw global dependencies between tokens.",
                },
                {
                    "text": "It sets a translation record.",
                    "quote": "On English-to-German translation it reaches 28.4 BLEU, a new state of the art.",
                },
            ],
        }
    )


# --------------------------------------------------------------------------- #
# Data contract
# --------------------------------------------------------------------------- #
def test_summary_point_and_structured_summary_are_frozen() -> None:
    from dataclasses import FrozenInstanceError

    pt = SummaryPoint(text="x", citation=None)
    with pytest.raises(FrozenInstanceError):
        pt.text = "y"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Detailed — single pass
# --------------------------------------------------------------------------- #
def test_detailed_returns_three_grounded_points() -> None:
    fake = _JsonLLM(_detailed_payload())
    s = _shell(fake)

    result = s.summarize_structured(_SOURCE, SUMMARY_TYPE_DETAILED)

    assert isinstance(result, StructuredSummary)
    assert result.summary_type == SUMMARY_TYPE_DETAILED
    assert result.lead == "The Transformer is an attention-only architecture."
    assert len(result.points) == 3
    for point in result.points:
        assert isinstance(point.citation, SourceSpan)
        # The citation offsets index the *real* source text.
        assert _SOURCE[point.citation.start : point.citation.end] == point.citation.quote


def test_detailed_requests_json_at_low_temperature() -> None:
    fake = _JsonLLM(_detailed_payload())
    s = _shell(fake)
    s.summarize_structured(_SOURCE, SUMMARY_TYPE_DETAILED)
    kwargs = fake.calls[0]["kwargs"]
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["temperature"] <= 0.15


def test_detailed_unmatched_quote_yields_no_citation() -> None:
    payload = json.dumps(
        {
            "lead": "Overview.",
            "points": [{"text": "A claim.", "quote": "This sentence is nowhere in the source."}],
        }
    )
    s = _shell(_JsonLLM(payload))
    result = s.summarize_structured(_SOURCE, SUMMARY_TYPE_DETAILED)
    assert result.points[0].citation is None


def test_detailed_render_text_contains_points() -> None:
    s = _shell(_JsonLLM(_detailed_payload()))
    result = s.summarize_structured(_SOURCE, SUMMARY_TYPE_DETAILED)
    assert "It drops recurrence." in result.text
    assert result.text.strip() != ""


# --------------------------------------------------------------------------- #
# Structured — sections
# --------------------------------------------------------------------------- #
def test_structured_returns_sections() -> None:
    payload = json.dumps(
        {
            "sections": {
                "PURPOSE": [
                    {
                        "text": "Replace recurrence with attention.",
                        "quote": "The Transformer is a model architecture eschewing recurrence entirely.",
                    }
                ],
                "METHOD": [
                    {
                        "text": "Self-attention over tokens.",
                        "quote": "It relies on self-attention to draw global dependencies between tokens.",
                    }
                ],
                "RESULTS": [
                    {
                        "text": "28.4 BLEU.",
                        "quote": "On English-to-German translation it reaches 28.4 BLEU, a new state of the art.",
                    }
                ],
                "CONCLUSIONS": [{"text": "Attention suffices."}],
            }
        }
    )
    s = _shell(_JsonLLM(payload))
    result = s.summarize_structured(_SOURCE, SUMMARY_TYPE_STRUCTURED)

    assert result.summary_type == SUMMARY_TYPE_STRUCTURED
    assert result.sections is not None
    assert set(result.sections) >= {"PURPOSE", "METHOD", "RESULTS", "CONCLUSIONS"}
    assert result.sections["PURPOSE"][0].citation is not None
    # CONCLUSIONS is a synthesis: no quote, so no citation.
    assert result.sections["CONCLUSIONS"][0].citation is None


# --------------------------------------------------------------------------- #
# Brief — provenance off, plain paragraph
# --------------------------------------------------------------------------- #
def test_brief_is_plain_paragraph_without_points() -> None:
    fake = _JsonLLM("A concise one-paragraph summary.")
    s = _shell(fake)
    result = s.summarize_structured(_SOURCE, SUMMARY_TYPE_BRIEF)
    assert result.summary_type == SUMMARY_TYPE_BRIEF
    assert result.lead == "A concise one-paragraph summary."
    assert result.points == []
    assert result.sections is None
    # Brief uses the plain chat path (not JSON mode).
    assert "response_format" not in fake.calls[0]["kwargs"]


# --------------------------------------------------------------------------- #
# Fallback — never hard-fail on bad JSON
# --------------------------------------------------------------------------- #
def test_invalid_json_falls_back_to_plain_summary() -> None:
    # First (structured) call returns junk; fallback calls plain summarize.
    fake = _JsonLLM("not json {{{ at all", "PLAIN FALLBACK SUMMARY")
    s = _shell(fake)
    result = s.summarize_structured(_SOURCE, SUMMARY_TYPE_DETAILED)
    assert result.lead == "PLAIN FALLBACK SUMMARY"
    assert result.points == []
    assert result.text == "PLAIN FALLBACK SUMMARY"


def test_summarize_structured_raises_after_close() -> None:
    s = _shell(_JsonLLM("{}"))
    s.llm = None
    with pytest.raises(RuntimeError):
        s.summarize_structured(_SOURCE)


# --------------------------------------------------------------------------- #
# Offset-aware chunker (pure function)
# --------------------------------------------------------------------------- #
def test_chunk_offsets_short_text_single_chunk() -> None:
    assert model_manager._split_into_chunks_with_offsets("hello world", 100) == [("hello world", 0)]


def test_chunk_offsets_nonpositive_max_returns_whole() -> None:
    assert model_manager._split_into_chunks_with_offsets("abc", 0) == [("abc", 0)]


def test_chunk_offsets_are_exact_substrings_of_source() -> None:
    text = " ".join(f"Sentence number {i} with some filler words here." for i in range(40))
    chunks = model_manager._split_into_chunks_with_offsets(text, 200)
    assert len(chunks) >= 2
    bases = [base for _, base in chunks]
    assert bases == sorted(bases)  # increasing
    for chunk_text, base in chunks:
        # Every chunk is a verbatim slice at its reported offset.
        assert text[base : base + len(chunk_text)] == chunk_text


def test_chunk_offsets_hard_split_oversized_sentence() -> None:
    big = "x" * 250  # no sentence boundary, longer than max
    chunks = model_manager._split_into_chunks_with_offsets(big, 100)
    assert "".join(c for c, _ in chunks) == big
    assert all(len(c) <= 100 for c, _ in chunks)


# --------------------------------------------------------------------------- #
# Map-reduce — long document, citations resolve to GLOBAL offsets
# --------------------------------------------------------------------------- #
def test_long_document_grounds_points_at_global_offsets() -> None:
    # A sentence that appears in every chunk; small ctx forces chunking.
    marker = "The attention mechanism is the central idea of this work."
    text = (
        (" ".join([f"Filler sentence {i} padding padding padding." for i in range(60)]))
        + " "
        + marker
    )
    text = "\n\n".join([text] * 3)
    payload = json.dumps({"lead": "x", "points": [{"text": "central idea", "quote": marker}]})
    fake = _JsonLLM(payload)
    s = _shell(fake, n_ctx=1024)  # forces multiple chunks

    result = s.summarize_structured(text, SUMMARY_TYPE_DETAILED)

    assert len(fake.calls) >= 2  # mapped over chunks
    grounded = [p for p in result.points if p.citation is not None]
    assert grounded, "expected at least one grounded point"
    for point in grounded:
        assert text[point.citation.start : point.citation.end] == point.citation.quote
    assert len(result.points) <= 3


def _multi_chunk_text(marker: str, blocks: int = 3) -> str:
    """A document large enough to force chunking, with ``marker`` in each block."""
    block = " ".join(f"Padding sentence {i} adds length to this block." for i in range(45))
    return "\n\n".join(f"{block} {marker}" for _ in range(blocks))


# --------------------------------------------------------------------------- #
# Citation offsets must index the caller's text, including leading whitespace
# --------------------------------------------------------------------------- #
def test_citations_index_source_with_leading_whitespace() -> None:
    # Extraction commonly yields leading whitespace; offsets must still index it.
    source = "\n\n   " + _SOURCE
    payload = json.dumps(
        {
            "lead": "x",
            "points": [
                {
                    "text": "drops recurrence",
                    "quote": "The Transformer is a model architecture eschewing recurrence entirely.",
                }
            ],
        }
    )
    s = _shell(_JsonLLM(payload))
    result = s.summarize_structured(source, SUMMARY_TYPE_DETAILED)
    cited = [p for p in result.points if p.citation is not None]
    assert cited
    for point in cited:
        assert source[point.citation.start : point.citation.end] == point.citation.quote


# --------------------------------------------------------------------------- #
# Map-reduce must not silently drop later chunks
# --------------------------------------------------------------------------- #
def test_detailed_map_reduce_covers_multiple_chunks() -> None:
    marker = "The central claim recurs verbatim across every part of the body."
    text = _multi_chunk_text(marker)

    class _IndexedLLM(_JsonLLM):
        def __init__(self) -> None:
            self.calls: list[int] = []

        def create_chat_completion(self, *, messages, **kwargs):
            n = len(self.calls)
            self.calls.append(n)
            content = json.dumps(
                {
                    "lead": f"lead{n}",
                    "points": [{"text": f"c{n}p{j}", "quote": marker} for j in range(3)],
                }
            )
            return {"choices": [{"message": {"content": content}}]}

    fake = _IndexedLLM()
    s = _shell(fake, n_ctx=1024)
    result = s.summarize_structured(text, SUMMARY_TYPE_DETAILED)

    assert len(fake.calls) >= 2
    assert len(result.points) <= 3
    # Surviving points must come from more than just the first chunk.
    survivors = {p.text[:2] for p in result.points}  # "c0", "c1", ...
    assert len(survivors) >= 2, [p.text for p in result.points]


def test_structured_map_reduce_grounds_sections_at_global_offsets() -> None:
    marker = "Self-attention connects all positions with a constant path length."
    text = _multi_chunk_text(marker)
    payload = json.dumps(
        {
            "sections": {
                "PURPOSE": [{"text": "purpose pt", "quote": marker}],
                "METHOD": [{"text": "method pt", "quote": marker}],
                "RESULTS": [{"text": "results pt", "quote": marker}],
                "CONCLUSIONS": [{"text": "synthesis only"}],
            }
        }
    )
    fake = _JsonLLM(payload)
    s = _shell(fake, n_ctx=1024)
    result = s.summarize_structured(text, SUMMARY_TYPE_STRUCTURED)

    assert len(fake.calls) >= 2
    assert result.sections is not None
    for name in ("PURPOSE", "METHOD", "RESULTS"):
        points = result.sections[name]
        assert len(points) <= 3
        for point in points:
            if point.citation is not None:
                assert text[point.citation.start : point.citation.end] == point.citation.quote
    for point in result.sections["CONCLUSIONS"]:
        assert point.citation is None


# --------------------------------------------------------------------------- #
# Fallback on well-formed-but-empty JSON
# --------------------------------------------------------------------------- #
def test_empty_detailed_json_falls_back() -> None:
    fake = _JsonLLM(json.dumps({"lead": "x", "points": []}), "PLAIN FALLBACK")
    s = _shell(fake)
    result = s.summarize_structured(_SOURCE, SUMMARY_TYPE_DETAILED)
    assert result.lead == "PLAIN FALLBACK"
    assert result.points == []
    assert len(fake.calls) == 2


def test_empty_structured_json_falls_back() -> None:
    fake = _JsonLLM(json.dumps({"sections": {}}), "PLAIN FALLBACK")
    s = _shell(fake)
    result = s.summarize_structured(_SOURCE, SUMMARY_TYPE_STRUCTURED)
    assert result.lead == "PLAIN FALLBACK"
    assert result.sections is None
    assert len(fake.calls) == 2


# --------------------------------------------------------------------------- #
# Prose-wrapped / trailing-brace JSON extraction
# --------------------------------------------------------------------------- #
def test_prose_wrapped_json_is_extracted() -> None:
    inner = json.dumps(
        {
            "lead": "ov",
            "points": [
                {
                    "text": "a",
                    "quote": "The Transformer is a model architecture eschewing recurrence entirely.",
                }
            ],
        }
    )
    fake = _JsonLLM(f"Sure! Here is the summary:\n{inner}\nHope that helps!")
    s = _shell(fake)
    result = s.summarize_structured(_SOURCE, SUMMARY_TYPE_DETAILED)
    assert result.lead == "ov"
    assert result.points[0].citation is not None


def test_json_with_trailing_prose_braces_still_parses() -> None:
    inner = json.dumps({"lead": "ov", "points": [{"text": "a", "quote": "x"}]})
    fake = _JsonLLM(f"{inner} (note: {{extra}})", "PLAIN FALLBACK")
    s = _shell(fake)
    result = s.summarize_structured(_SOURCE, SUMMARY_TYPE_DETAILED)
    # Leading JSON object is parsed despite the trailing prose braces.
    assert result.lead == "ov"


# --------------------------------------------------------------------------- #
# Point-count cap is consistent between single-pass and map-reduce
# --------------------------------------------------------------------------- #
def test_single_pass_caps_points_to_three() -> None:
    quote = "The Transformer is a model architecture eschewing recurrence entirely."
    points = [{"text": f"pt{i}", "quote": quote} for i in range(5)]
    fake = _JsonLLM(json.dumps({"lead": "x", "points": points}))
    s = _shell(fake)
    result = s.summarize_structured(_SOURCE, SUMMARY_TYPE_DETAILED)
    assert len(result.points) == 3


# --------------------------------------------------------------------------- #
# Defensive coercion of malformed point fields
# --------------------------------------------------------------------------- #
def test_malformed_point_fields_are_rejected() -> None:
    payload = json.dumps(
        {
            "lead": "x",
            "points": [
                {"text": {"nested": "obj"}},  # non-string text → dropped
                {"text": "kept", "quote": None},  # null quote → no citation, no "None" search
            ],
        }
    )
    s = _shell(_JsonLLM(payload))
    result = s.summarize_structured(_SOURCE, SUMMARY_TYPE_DETAILED)
    assert [p.text for p in result.points] == ["kept"]
    assert result.points[0].citation is None


# --------------------------------------------------------------------------- #
# Render shape + immutability
# --------------------------------------------------------------------------- #
def test_structured_render_text_has_ordered_section_headers() -> None:
    payload = json.dumps(
        {
            "sections": {
                "PURPOSE": [{"text": "purpose text", "quote": "x"}],
                "METHOD": [{"text": "method text", "quote": "x"}],
                "RESULTS": [{"text": "results text", "quote": "x"}],
                "CONCLUSIONS": [{"text": "conclusion text"}],
            }
        }
    )
    s = _shell(_JsonLLM(payload))
    rendered = s.summarize_structured(_SOURCE, SUMMARY_TYPE_STRUCTURED).text
    for header in ("Purpose:", "Method:", "Results:", "Conclusions:"):
        assert header in rendered
    assert (
        rendered.index("Purpose:")
        < rendered.index("Method:")
        < rendered.index("Results:")
        < rendered.index("Conclusions:")
    )
    assert "- conclusion text" in rendered


def test_structured_summary_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    summary = StructuredSummary("brief", "x", [], None, "x")
    with pytest.raises(FrozenInstanceError):
        summary.lead = "y"  # type: ignore[misc]
