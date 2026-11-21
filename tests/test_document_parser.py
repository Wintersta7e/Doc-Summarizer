# SPDX-License-Identifier: MIT
"""Tests for `docsummarizer.document_parser`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from docsummarizer.document_parser import (
    SUPPORTED_EXTENSION_SET,
    extract_text,
    find_documents,
    get_document_info,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_extract_unsupported_extension(tmp_path: Path) -> None:
    target = tmp_path / "thing.xyz"
    _ = target.write_text("doesn't matter")
    text, error = extract_text(str(target))
    assert text == ""
    assert error is not None
    assert "Unsupported" in error


def test_extract_missing_file(tmp_path: Path) -> None:
    text, error = extract_text(str(tmp_path / "nope.txt"))
    assert text == ""
    assert error is not None
    assert "not found" in error.lower()


def test_extract_txt_utf8(tmp_path: Path) -> None:
    target = tmp_path / "hello.txt"
    _ = target.write_text("hello world\n", encoding="utf-8")
    text, error = extract_text(str(target))
    assert error is None
    assert "hello world" in text


def test_extract_txt_latin1_encoding_detection(tmp_path: Path) -> None:
    target = tmp_path / "latin.txt"
    _ = target.write_bytes("caf\xe9 m\xfcnchen\n".encode("latin-1"))
    text, error = extract_text(str(target))
    assert error is None
    # We don't promise exact decoding — only that we got *something*
    # printable and didn't choke on the encoding mismatch.
    assert "caf" in text.lower()


def test_extract_md_routes_to_txt(tmp_path: Path) -> None:
    target = tmp_path / "notes.md"
    _ = target.write_text("# heading\n\nbody\n", encoding="utf-8")
    text, error = extract_text(str(target))
    assert error is None
    assert "heading" in text


def test_extract_docx_basic(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    target = tmp_path / "doc.docx"
    document = docx.Document()
    document.add_paragraph("first paragraph")
    document.add_paragraph("second paragraph")
    document.save(str(target))

    text, error = extract_text(str(target))
    assert error is None
    assert "first paragraph" in text
    assert "second paragraph" in text


def test_extract_rtf_basic(tmp_path: Path) -> None:
    pytest.importorskip("striprtf")
    target = tmp_path / "doc.rtf"
    _ = target.write_text(r"{\rtf1\ansi hello rtf body}", encoding="utf-8")
    text, error = extract_text(str(target))
    assert error is None
    assert "hello rtf body" in text


def test_legacy_doc_not_supported(tmp_path: Path) -> None:
    """`.doc` (binary OLE) was dropped: python-docx can't read it."""
    target = tmp_path / "old.doc"
    _ = target.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")  # OLE magic
    text, error = extract_text(str(target))
    assert text == ""
    assert error is not None
    assert "Unsupported" in error


def test_supported_extension_set_canonical() -> None:
    """The canonical set is the single source of truth — assert against it
    directly rather than parsing the dialog-filter strings."""
    assert ".doc" not in SUPPORTED_EXTENSION_SET
    assert ".docx" in SUPPORTED_EXTENSION_SET
    assert {".pdf", ".docx", ".rtf", ".txt", ".md", ".text"} == SUPPORTED_EXTENSION_SET


def test_get_document_info_existing(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    _ = target.write_bytes(b"x" * 2048)

    info = get_document_info(str(target))
    assert info["name"] == "f.txt"
    assert info["extension"] == ".txt"
    assert info["size_bytes"] == 2048
    assert info["size_mb"] == round(2048 / (1024 * 1024), 2)


def test_get_document_info_missing(tmp_path: Path) -> None:
    info = get_document_info(str(tmp_path / "missing.txt"))
    assert info["size_bytes"] == 0
    assert info["size_mb"] == 0


def test_find_documents_on_single_supported_file(tmp_path: Path) -> None:
    target = tmp_path / "report.pdf"
    _ = target.write_bytes(b"%PDF-")
    assert find_documents(target) == [target]


def test_find_documents_on_single_unsupported_file(tmp_path: Path) -> None:
    target = tmp_path / "thing.xyz"
    _ = target.write_text("nope")
    assert find_documents(target) == []


def test_find_documents_in_directory_filters_to_supported(tmp_path: Path) -> None:
    _ = (tmp_path / "a.pdf").write_bytes(b"")
    _ = (tmp_path / "b.docx").write_bytes(b"")
    _ = (tmp_path / "c.exe").write_bytes(b"")
    (tmp_path / "subdir").mkdir()
    _ = (tmp_path / "subdir" / "ignored.pdf").write_bytes(b"")  # non-recursive

    found = {p.name for p in find_documents(tmp_path)}
    assert found == {"a.pdf", "b.docx"}
