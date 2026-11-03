"""Tests for `docsummarizer.provenance` — sentence splitting + quote location.

Pure functions, no model: given a source string and a (model-emitted) quote,
locate the source sentence it came from, by char offsets into the full text.
"""

from __future__ import annotations

from difflib import SequenceMatcher

import pytest

from docsummarizer.provenance import SourceSpan, locate_quote, split_sentences


# --------------------------------------------------------------------------- #
# split_sentences — offsets must be exact slices of the source
# --------------------------------------------------------------------------- #
def test_split_sentences_returns_exact_offset_slices() -> None:
    text = "Foo bar. Baz qux! Quux?"
    spans = split_sentences(text)
    assert [text[s:e] for s, e in spans] == ["Foo bar.", "Baz qux!", "Quux?"]


def test_split_sentences_empty_text_is_empty() -> None:
    assert split_sentences("") == []
    assert split_sentences("   \n\n  ") == []


def test_split_sentences_no_terminal_punctuation_is_one_span() -> None:
    text = "a heading with no full stop"
    spans = split_sentences(text)
    assert len(spans) == 1
    assert text[spans[0][0] : spans[0][1]] == text


def test_split_sentences_offsets_are_into_full_text() -> None:
    text = "First sentence here. Second one follows."
    spans = split_sentences(text)
    # The second span must start partway through the document, not at 0.
    assert spans[1][0] > 0
    assert text[spans[1][0] : spans[1][1]] == "Second one follows."


def test_split_sentences_splits_on_newlines_after_punctuation() -> None:
    text = "Line one ends.\nLine two ends."
    spans = split_sentences(text)
    assert [text[s:e] for s, e in spans] == ["Line one ends.", "Line two ends."]


# --------------------------------------------------------------------------- #
# locate_quote — fuzzy-match a quote to the best source sentence
# --------------------------------------------------------------------------- #
_SOURCE = (
    "The Transformer is a model architecture eschewing recurrence. "
    "It relies entirely on an attention mechanism to draw global dependencies. "
    "Experiments show it is superior in quality while being more parallelizable."
)

_WATER = (
    "The pumping station draws raw water from the reservoir through two intake pipes. "
    "Sand filters remove most of the sediment before the water reaches the treatment hall. "
    "Operators test the chlorine level every four hours and record it in the daily log."
)


def test_locate_quote_verbatim_match_returns_that_sentence() -> None:
    quote = "It relies entirely on an attention mechanism to draw global dependencies."
    span = locate_quote(quote, _SOURCE)
    assert span is not None
    assert _SOURCE[span.start : span.end] == quote
    assert span.score >= 0.99


def test_locate_quote_offsets_map_into_full_source() -> None:
    quote = "The Transformer is a model architecture eschewing recurrence."
    span = locate_quote(quote, _SOURCE)
    assert span is not None
    assert span.start == 0
    assert _SOURCE[span.start : span.end] == quote


def test_locate_quote_unrelated_text_returns_none() -> None:
    assert locate_quote("The mitochondria is the powerhouse of the cell.", _SOURCE) is None


def test_locate_quote_empty_quote_returns_none() -> None:
    assert locate_quote("", _SOURCE) is None
    assert locate_quote("   ", _SOURCE) is None


def test_locate_quote_short_verbatim_fragment_grounds() -> None:
    # A 4-bit model often cites a phrase, not a whole sentence. A verbatim
    # fragment must still ground to its source sentence (coverage, not the
    # symmetric whole-sentence ratio).
    for fragment in ("eschewing recurrence", "attention mechanism", "superior in quality"):
        span = locate_quote(fragment, _SOURCE)
        assert span is not None, fragment
        assert fragment in _SOURCE[span.start : span.end]


def test_locate_quote_threshold_governs_fuzzy_match() -> None:
    # A threshold can only raise the coverage bar: a strict one rejects a loose
    # paraphrase, and one below the default is refused rather than loosening it.
    quote = "the model avoids recurrent and convolutional layers"
    assert locate_quote(quote, _SOURCE, threshold=0.99) is None
    with pytest.raises(ValueError, match="threshold"):
        locate_quote(quote, _SOURCE, threshold=0.1)


def test_paraphrase_below_contiguous_bar_is_not_located() -> None:
    """A paraphrase with high summed coverage but no long contiguous run is not cited.

    Catches dropping the contiguous-run condition, so coverage summed over
    scattered blocks alone admits a paraphrase.
    """
    from docsummarizer.provenance import _normalize

    quote = "Operators check chlorine every four hours and note it in the log."
    start, end = split_sentences(_WATER)[2]
    sentence = _WATER[start:end]
    q = _normalize(quote)
    matcher = SequenceMatcher(None, q, _normalize(sentence), autojunk=False)
    sizes = [block.size for block in matcher.get_matching_blocks()]
    coverage = sum(sizes) / len(q)
    longest = max(sizes) / len(q)
    print(f"coverage={coverage:.3f} longest={longest:.3f}")
    assert coverage >= 0.6
    assert longest < 0.5
    assert locate_quote(quote, _WATER) is None


def test_locate_quote_returns_best_verified_sentence() -> None:
    """The best-covering sentence that verifies wins, not the best-covering one.

    Catches ranking first and testing only the top-coverage sentence (rank,
    then filter), which returns ``None`` or the unverified sentence.
    """
    source = (
        "The quiet river once flooded half of the lower old town each spring. "
        "Every year the river flooded the lower fields."
    )
    quote = "the river flooded the lower town in spring"
    span = locate_quote(quote, source)
    assert span is not None
    assert span.quote == "Every year the river flooded the lower fields."
    assert source[span.start : span.end] == span.quote


def test_verified_predicate_cases() -> None:
    from docsummarizer.provenance import is_verified_match

    s1, s2, s3 = (_WATER[s:e] for s, e in split_sentences(_WATER))
    messy = "Sand filters remove   most of the\n   sediment before the water\nreaches the treatment hall."
    wrapped = f"{s1} {messy} {s3}"
    paraphrase = "Operators check chlorine every four hours and note it in the log."
    near = "Operators test the chlorine level every four hours and write it in the daily log."
    # (label, quote, text, threshold, expected span text or None, expected score or None)
    cases: list[tuple[str, str, str, float, str | None, float | None]] = [
        ("a verbatim sentence", s2, _WATER, 0.6, s2, 1.0),
        ("b wrapped source", s2, wrapped, 0.6, messy, 1.0),
        (
            "c wrapped quote",
            "Sand filters  remove most of the\nsediment before  the water reaches the treatment hall.",
            _WATER,
            0.6,
            s2,
            None,
        ),
        ("d paraphrase", paraphrase, _WATER, 0.6, None, None),
        ("e too short", "pipes", _WATER, 0.6, None, None),
        ("f near-verbatim at default", near, _WATER, 0.6, s3, 0.963),
        ("f near-verbatim at 0.99", near, _WATER, 0.99, None, None),
    ]
    for label, quote, text, threshold, want, want_score in cases:
        span = locate_quote(quote, text, threshold=threshold)
        got = None if span is None else span.quote
        print(f"{label}: expected={want!r} got={got!r}")
        assert got == want, label
        if span is not None:
            assert is_verified_match(quote, span.quote), label
            assert text[span.start : span.end] == span.quote, label
            if want_score is not None:
                assert span.score == pytest.approx(want_score, abs=5e-4), label
    assert is_verified_match("pipes", _WATER) is False
    print("g threshold below default: expected=ValueError")
    with pytest.raises(ValueError, match="threshold"):
        locate_quote(s2, _WATER, threshold=0.1)


def test_locate_quote_consults_is_verified_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """``locate_quote`` grounds only through the public predicate.

    Catches inlining the conditions in ``locate_quote``, so a patched
    predicate no longer governs production grounding.
    """
    from docsummarizer import provenance

    monkeypatch.setattr(provenance, "is_verified_match", lambda quote, span_text: False)
    s1 = _WATER[: split_sentences(_WATER)[0][1]]
    assert locate_quote(s1, _WATER) is None


def test_locate_quote_rejects_trivially_short_quote() -> None:
    # A 1-2 char "quote" would match almost anything; reject it.
    assert locate_quote("the", _SOURCE) is None


def test_split_sentences_keeps_abbreviations_whole() -> None:
    text = "Dr. Smith left at noon. He went home."
    spans = split_sentences(text)
    assert [text[s:e] for s, e in spans] == ["Dr. Smith left at noon.", "He went home."]


def test_split_sentences_keeps_initials_whole() -> None:
    text = "The U.S. economy grew. Inflation fell."
    spans = split_sentences(text)
    assert [text[s:e] for s, e in spans] == ["The U.S. economy grew.", "Inflation fell."]


def test_locate_quote_grounds_sentence_with_abbreviation() -> None:
    source = "Dr. Smith reported a forty percent gain. The study ended early."
    quote = "Dr. Smith reported a forty percent gain."
    span = locate_quote(quote, source)
    assert span is not None
    assert source[span.start : span.end] == quote


def test_source_span_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    import pytest

    span = SourceSpan(start=0, end=5, quote="hello", score=1.0)
    with pytest.raises(FrozenInstanceError):
        span.start = 3  # type: ignore[misc]
