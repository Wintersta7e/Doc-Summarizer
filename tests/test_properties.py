# SPDX-License-Identifier: MIT
"""Properties of the parsers, chunkers, provenance and language normalisation.

Hypothesis generates the inputs: unicode text with CJK, emoji and mixed
newlines, and hostile bytes for every extractor. ``conftest.py`` registers the
profiles: ``ci`` (deterministic, 200 examples) by default, ``fuzz`` (10 000)
for the weekly run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st

from docsummarizer.document_parser import _EXTRACTORS, extract_from_txt, extract_text
from docsummarizer.io_helpers import summary_filename, write_summary_docx, write_text
from docsummarizer.model_manager import (
    _MAX_LANGUAGE_LEN,
    SUMMARY_TYPES,
    _split_into_chunks,
    _split_into_chunks_with_offsets,
    normalize_language,
)
from docsummarizer.provenance import (
    _MIN_QUOTE_CHARS,
    is_verified_match,
    locate_quote,
    split_sentences,
)
from tests.test_document_parser import _write_pdf

if TYPE_CHECKING:
    from collections.abc import Callable

_LOG = logging.getLogger(__name__)

# Document text: letters, marks, digits, punctuation and symbols from every
# script, mixed with newline styles, sentence ends, CJK, emoji and abbreviations.
_PIECES = st.sampled_from(
    [
        "\n",
        "\r\n",
        "\r",
        "\n\n",
        "\t",
        " ",
        ". ",
        "! ",
        "? ",
        "。",
        "😀",
        "👩‍💻",
        "日本語の文書です。",
        "这是一个句子。",
        "Dr. ",
        "e.g. ",
    ]
)
_TEXT = st.lists(st.one_of(st.characters(categories=("L", "M", "N", "P", "S", "Zs")), _PIECES)).map(
    "".join
)
# Paragraphs of that text, so a chunk bound lands on paragraph and sentence
# boundaries as often as inside them.
_DOCUMENT = st.lists(_TEXT, max_size=8).map("\n\n".join)
_BOUNDS = st.integers(min_value=1, max_value=60)
# A run of the text with no whitespace, so no sentence boundary can split it,
# long enough to be quoted.
_WORD = st.text(st.characters(categories=("L", "N")), min_size=_MIN_QUOTE_CHARS)

# Two alphabets with no character, upper or lower case, in common: a quote drawn
# from one shares nothing with a document drawn from the other.
_DOCUMENT_ALPHABET = "abcdefghijklmABCDEFGHIJKLM一二三四五 .!?\n"
_FOREIGN_ALPHABET = "nopqrstuvwxyzNOPQRSTUVWXYZ六七八九十 "

# The Unicode encodings, each of which the detector reads exactly; legacy code
# pages are detected heuristically and decode on a best-effort basis.
_UNICODE_ENCODINGS = ("utf-8", "utf-8-sig", "utf-16", "utf-32")


def _ink(text: str) -> str:
    """``text`` without its whitespace: what a chunker may not drop or reorder."""
    return "".join(text.split())


@given(text=_DOCUMENT, max_chars=_BOUNDS)
def test_paragraph_chunks_keep_every_character_in_order_within_the_bound(
    text: str, max_chars: int
) -> None:
    chunks = _split_into_chunks(text, max_chars)

    assert all(len(chunk) <= max_chars for chunk in chunks)
    assert _ink("".join(chunks)) == _ink(text)


@given(text=_DOCUMENT, max_chars=_BOUNDS)
def test_offset_chunks_are_slices_of_the_text_that_skip_only_whitespace(
    text: str, max_chars: int
) -> None:
    position = 0
    for chunk, offset in _split_into_chunks_with_offsets(text, max_chars):
        assert len(chunk) <= max_chars
        assert offset >= position
        assert text[offset : offset + len(chunk)] == chunk
        assert not text[position:offset].strip()
        position = offset + len(chunk)
    assert not text[position:].strip()


@given(before=_TEXT, quote=_WORD, after=_TEXT)
def test_a_quote_taken_from_the_text_locates_a_span_that_verifies(
    before: str, quote: str, after: str
) -> None:
    text = before + quote + after

    span = locate_quote(quote, text)

    assert span is not None
    assert text[span.start : span.end] == span.quote
    assert is_verified_match(quote, span.quote)


@given(
    text=st.text(alphabet=_DOCUMENT_ALPHABET),
    quote=st.text(alphabet=_FOREIGN_ALPHABET, min_size=1),
)
def test_a_quote_sharing_nothing_with_the_text_never_verifies(text: str, quote: str) -> None:
    assert locate_quote(quote, text) is None
    assert not any(is_verified_match(quote, text[s:e]) for s, e in split_sentences(text))


@given(language=st.one_of(st.none(), st.text()))
def test_normalize_language_is_one_bounded_line_and_idempotent(language: str | None) -> None:
    once = normalize_language(language)

    assert once.splitlines() == [once]
    assert len(once) <= _MAX_LANGUAGE_LEN
    assert normalize_language(once) == once


@pytest.mark.parametrize("encoding", _UNICODE_ENCODINGS)
@given(text=_TEXT)
def test_txt_text_round_trips_through_each_unicode_encoding(
    encoding: str, text: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    path = tmp_path_factory.mktemp("txt") / "document.txt"
    _ = path.write_bytes(text.encode(encoding))

    assert extract_from_txt(str(path)) == text


@given(
    name=st.text(),
    summary_type=st.sampled_from(SUMMARY_TYPES),
    fmt=st.sampled_from(["txt", "docx"]),
)
def test_a_summary_filename_is_one_entry_of_the_output_dir(
    name: str, summary_type: str, fmt: str
) -> None:
    out = Path("summaries")
    filename = summary_filename(Path(name), summary_type, fmt)

    target = out / filename

    assert target.parent == out
    assert target.name == filename


@pytest.mark.parametrize("fmt", ["txt", "docx"])
@given(existing=st.binary())
def test_an_export_without_replace_never_overwrites_a_file(
    fmt: str, existing: bytes, tmp_path_factory: pytest.TempPathFactory
) -> None:
    out = tmp_path_factory.mktemp("export")
    target = out / f"document.pdf.detailed.{fmt}"
    _ = target.write_bytes(existing)

    if fmt == "txt":
        written = write_text(target, "a new summary", replace=False)
    else:
        written = write_summary_docx(
            target, source_name="document.pdf", summary="a new summary", replace=False
        )

    assert written is False
    assert target.read_bytes() == existing
    assert list(out.iterdir()) == [target]


def _docx_seed(path: Path) -> None:
    from docx import Document

    document = Document()
    _ = document.add_paragraph("A seed paragraph.")
    _ = document.add_table(rows=1, cols=2).rows[0].cells[0].add_paragraph("cell")
    document.save(str(path))


def _pdf_seed(path: Path) -> None:
    _write_pdf(path, "A seed page.")


def _rtf_seed(path: Path) -> None:
    _ = path.write_text(r"{\rtf1\ansi{\fonttbl\f0 Arial;}\f0 A seed \b paragraph\b0.\par}")


def _text_seed(path: Path) -> None:
    _ = path.write_text("A seed paragraph.\n\n日本語の文書です。\n", encoding="utf-8")


_SEEDS: dict[str, Callable[[Path], None]] = {
    ".pdf": _pdf_seed,
    ".docx": _docx_seed,
    ".rtf": _rtf_seed,
    ".txt": _text_seed,
    ".md": _text_seed,
    ".text": _text_seed,
}


@pytest.fixture(scope="session")
def seeds(tmp_path_factory: pytest.TempPathFactory) -> dict[str, bytes]:
    """A valid file of each supported extension, as bytes.

    Returns:
        The bytes of each extension's seed file.
    """
    out = tmp_path_factory.mktemp("seeds")
    found: dict[str, bytes] = {}
    for extension, write in _SEEDS.items():
        path = out / f"seed{extension}"
        write(path)
        found[extension] = path.read_bytes()
    return found


@st.composite
def _mutated(draw: st.DrawFn, seed: bytes) -> bytes:
    """``seed`` with some bytes overwritten, then cut short: input a parser gets deep into."""
    data = bytearray(seed)
    for _ in range(draw(st.integers(min_value=1, max_value=16))):
        data[draw(st.integers(min_value=0, max_value=len(data) - 1))] = draw(
            st.integers(min_value=0, max_value=255)
        )
    return bytes(data[: draw(st.integers(min_value=0, max_value=len(data)))])


def test_every_extractor_has_a_seed() -> None:
    assert set(_SEEDS) == set(_EXTRACTORS)


@pytest.mark.parametrize("extension", sorted(_EXTRACTORS))
@given(data=st.data())
def test_hostile_bytes_give_text_or_an_error_message(
    extension: str,
    seeds: dict[str, bytes],
    tmp_path_factory: pytest.TempPathFactory,
    data: st.DataObject,
) -> None:
    seed = seeds[extension]
    raw = data.draw(st.one_of(st.binary(max_size=4096), _mutated(seed)))
    path = tmp_path_factory.mktemp("hostile") / f"document{extension}"
    _ = path.write_bytes(raw)

    text, error = extract_text(str(path))

    if error is None:
        assert text.strip()
    else:
        assert text == ""
        assert error
