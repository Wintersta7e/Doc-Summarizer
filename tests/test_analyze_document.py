# SPDX-License-Identifier: MIT
"""Tests for `document_parser.analyze_document` — the metadata strip data.

The Summary/Extract screens show "N PAGES · N WORDS · N CHARS · ENC · parser".
None of that existed before; analyze_document derives it. Pages are PDF-only,
encoding is plain-text-only, words/chars come from the extracted text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from docsummarizer import document_parser
from docsummarizer.document_parser import analyze_document

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_txt_stats_from_real_file(tmp_path: Path) -> None:
    f = tmp_path / "note.txt"
    _ = f.write_text("Hello world. Foo bar baz.", encoding="utf-8")

    stats = analyze_document(str(f))

    assert stats["name"] == "note.txt"
    assert stats["words"] == 5
    assert stats["chars"] == len("Hello world. Foo bar baz.")
    assert stats["pages"] is None
    assert stats["parser"] == "chardet"
    assert isinstance(stats["encoding"], str)
    assert stats["encoding"]


def test_md_uses_text_parser_label(tmp_path: Path) -> None:
    f = tmp_path / "readme.md"
    _ = f.write_text("# Title\n\nSome body text here.", encoding="utf-8")
    stats = analyze_document(str(f))
    assert stats["parser"] == "chardet"
    assert stats["pages"] is None


def test_words_and_chars_use_provided_text_not_file() -> None:
    # When text is supplied, the file is not re-read for word/char counts.
    stats = analyze_document("ghost.docx", text="one two three")
    assert stats["words"] == 3
    assert stats["chars"] == len("one two three")
    assert stats["name"] == "ghost.docx"
    assert stats["parser"] == "python-docx"
    assert stats["pages"] is None
    assert stats["encoding"] is None


def test_rtf_parser_label() -> None:
    stats = analyze_document("doc.rtf", text="body")
    assert stats["parser"] == "striprtf"


def _twelve_pages(_file_path: str) -> int:
    return 12


def test_pdf_reports_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(document_parser, "_pdf_page_count", _twelve_pages)
    stats = analyze_document("paper.pdf", text="extracted body text")
    assert stats["pages"] == 12
    assert stats["parser"] == "pypdf"
    assert stats["encoding"] is None


def test_analyze_extracts_when_text_not_supplied(tmp_path: Path) -> None:
    f = tmp_path / "doc.txt"
    _ = f.write_text("alpha beta gamma delta", encoding="utf-8")
    stats = analyze_document(str(f))  # no text= → extracts internally
    assert stats["words"] == 4


def test_pdf_page_count_handles_unreadable_file() -> None:
    # Missing/corrupt PDF must not raise — analyze should degrade to None pages.
    assert analyze_document("does-not-exist.pdf", text="body")["pages"] is None
