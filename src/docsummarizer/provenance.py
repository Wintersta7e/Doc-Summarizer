"""Provenance: locate the source sentence a summary point came from.

Pure, dependency-free (stdlib only) so it is fully unit-testable without the
LLM. The model emits a ``quote`` alongside each summary point; we re-locate
that quote in the *full* source document by fuzzy-matching it against the
document's sentences, and record the char offsets of the best sentence that
verifiably contains the quote (:func:`is_verified_match`). A quote no sentence
verifiably contains yields ``None`` — we never fabricate an offset.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

# Minimum *quote coverage* (0..1): how much of the quote is present in a source
# sentence, so a short verbatim phrase inside a long sentence still grounds.
# Coverage is one of three conditions; contiguity (below) is the second, and a
# minimum quote length the third.
DEFAULT_THRESHOLD = 0.6

# Quotes shorter than this (after normalization) are rejected: a 1-2 char quote
# would match almost any sentence and produce meaningless provenance.
_MIN_QUOTE_CHARS = 6

# Coverage sums disjoint matching blocks, so a paraphrase that shares scattered
# words reaches it too; one contiguous run must also cover this much of a quote.
VERIFIED_LONGEST_FRACTION = 0.5

# Whitespace next to CJK text is ignored: a model copying it drops wrap spaces.
# Han, kana, Hangul, CJK symbols and punctuation, fullwidth forms (escaped, so
# Unicode normalization of this file cannot fold a range endpoint).
_CJK = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef\u3000-\u303f\uac00-\ud7af"
_SPACE_BY_CJK = re.compile(rf"(?<=[{_CJK}])\s+|\s+(?=[{_CJK}])")

# Split on whitespace that follows sentence-terminal punctuation. Keeps the
# terminator with the sentence it ends. Newlines count as whitespace, so this
# also breaks on line boundaries after a full stop.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

# A period after one of these (or after a single uppercase initial) is an
# abbreviation, not a sentence end — splitting there would fragment a sentence
# and break verbatim-quote grounding.
_ABBREVIATIONS = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "sr",
        "jr",
        "st",
        "vs",
        "etc",
        "al",
        "fig",
        "no",
        "vol",
        "pp",
        "inc",
        "ltd",
        "co",
        "eg",
        "ie",
        "cf",
        "approx",
        "dept",
        "est",
        "min",
        "max",
    }
)
_TOKEN_BEFORE_DOT = re.compile(r"([A-Za-z]+)\.\Z")


@dataclass(frozen=True)
class SourceSpan:
    """A located source sentence: char offsets into the full document.

    ``start``/``end`` index the full source text such that ``text[start:end]``
    is ``quote``. ``score`` is the quote-coverage confidence (0..1).
    """

    start: int
    end: int
    quote: str
    score: float


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink ``[start, end)`` to exclude leading/trailing whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _is_abbreviation_boundary(text: str, boundary_start: int) -> bool:
    """True if the period just before ``boundary_start`` is an abbreviation dot."""
    if boundary_start == 0 or text[boundary_start - 1] != ".":
        return False
    match = _TOKEN_BEFORE_DOT.search(text[:boundary_start])
    if match is None:
        return False
    token = match.group(1)
    return (len(token) == 1 and token.isupper()) or token.lower() in _ABBREVIATIONS


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` char-offset spans of the sentences in ``text``.

    Each span covers a sentence's non-whitespace extent, so ``text[start:end]``
    is the verbatim sentence. Abbreviations and initials (``Dr.``, ``U.S.``)
    do not split. Whitespace-only input yields ``[]``.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    for boundary in _SENTENCE_BOUNDARY.finditer(text):
        if _is_abbreviation_boundary(text, boundary.start()):
            continue
        seg_start, seg_end = _trim(text, pos, boundary.start())
        if seg_end > seg_start:
            spans.append((seg_start, seg_end))
        pos = boundary.end()
    seg_start, seg_end = _trim(text, pos, len(text))
    if seg_end > seg_start:
        spans.append((seg_start, seg_end))
    return spans


def _normalize(s: str) -> str:
    """Lowercase, collapse whitespace runs, drop whitespace touching CJK text."""
    return _SPACE_BY_CJK.sub("", " ".join(s.lower().split()))


def is_verified_match(quote: str, span_text: str) -> bool:
    """True if ``span_text`` verifiably contains ``quote``.

    After normalization the quote must be at least ``_MIN_QUOTE_CHARS`` long,
    at least ``DEFAULT_THRESHOLD`` of it must be present in ``span_text``, and
    one contiguous run must cover at least ``VERIFIED_LONGEST_FRACTION`` of it.
    """
    q = _normalize(quote)
    if len(q) < _MIN_QUOTE_CHARS:
        return False
    matcher = SequenceMatcher(None, q, _normalize(span_text), autojunk=False)
    sizes = [block.size for block in matcher.get_matching_blocks()]
    coverage = sum(sizes) / len(q)
    return coverage >= DEFAULT_THRESHOLD and max(sizes) >= VERIFIED_LONGEST_FRACTION * len(q)


def locate_quote(
    quote: str,
    text: str,
    sentences: list[tuple[int, int]] | None = None,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> SourceSpan | None:
    """Find the best source sentence in ``text`` that verifiably contains ``quote``.

    Sentences whose *quote coverage* (the fraction of the quote present in the
    sentence) reaches ``threshold`` are ranked by coverage, then by how tightly
    they fit; the earlier sentence wins a full tie. The first one that
    :func:`is_verified_match` accepts is returned as a :class:`SourceSpan`
    (offsets into ``text``), otherwise ``None``. ``threshold`` can only raise
    the coverage bar: below ``DEFAULT_THRESHOLD`` it raises ``ValueError``.
    Pass precomputed ``sentences`` to avoid re-splitting when locating many
    quotes against the same source.
    """
    if threshold < DEFAULT_THRESHOLD:
        raise ValueError(f"threshold must be at least {DEFAULT_THRESHOLD}, got {threshold}")
    normalized_quote = _normalize(quote)
    if len(normalized_quote) < _MIN_QUOTE_CHARS:
        return None
    if sentences is None:
        sentences = split_sentences(text)

    candidates: list[tuple[float, float, int, int]] = []
    matcher = SequenceMatcher(autojunk=False)
    matcher.set_seq1(normalized_quote)
    for start, end in sentences:
        matcher.set_seq2(_normalize(text[start:end]))
        matched = sum(block.size for block in matcher.get_matching_blocks())
        coverage = matched / len(normalized_quote)
        if coverage >= threshold:
            candidates.append((coverage, matcher.ratio(), start, end))

    # Prefer higher coverage; tie-break toward the tighter-fitting sentence. The
    # sort is stable, so the earlier sentence wins a full tie.
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    for coverage, _ratio, start, end in candidates:
        if is_verified_match(quote, text[start:end]):
            return SourceSpan(start=start, end=end, quote=text[start:end], score=round(coverage, 4))
    return None
