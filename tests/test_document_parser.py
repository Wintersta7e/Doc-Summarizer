# SPDX-License-Identifier: MIT
"""Tests for `docsummarizer.document_parser`."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import chardet
import pytest

from docsummarizer.document_parser import (
    SUPPORTED_EXTENSION_SET,
    analyze_document,
    extract_text,
    find_documents,
    get_document_info,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("detected", [None, "ascii"], ids=["no-guess", "too-narrow"])
def test_extract_txt_replaces_bytes_the_detected_encoding_cannot_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, detected: str | None
) -> None:
    """Not UTF-8, and the detector's guess (none means UTF-8) cannot read every byte."""

    def detect(_raw: bytes) -> dict[str, str | None]:
        return {"encoding": detected}

    monkeypatch.setattr(chardet, "detect", detect)
    target = tmp_path / "latin.txt"
    _ = target.write_bytes(b"caf\xe9")

    assert extract_text(str(target)) == ("caf�", None)
    assert analyze_document(str(target))["encoding"] == (detected or "utf-8")


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


# Valid UTF-8 holding a NUL, C1 controls and unassigned astral code points: the
# encoding detector reads these bytes as UTF-16-BE.
_UTF8_WITH_CONTROLS = "\xe4\x00\U00091647q\xb4\x93u\xd6\xf8\U000cef81\xa8\x9435Z\x7f"


@pytest.mark.parametrize(("encoding", "label"), [("utf-8", "utf-8"), ("utf-8-sig", "UTF-8-SIG")])
def test_extract_txt_reads_valid_utf8_with_control_characters_as_utf8(
    tmp_path: Path, encoding: str, label: str
) -> None:
    target = tmp_path / "controls.txt"
    _ = target.write_bytes(_UTF8_WITH_CONTROLS.encode(encoding))

    assert extract_text(str(target)) == (_UTF8_WITH_CONTROLS, None)
    assert analyze_document(str(target))["encoding"] == label


def test_extract_md_routes_to_txt(tmp_path: Path) -> None:
    target = tmp_path / "notes.md"
    _ = target.write_text("# heading\n\nbody\n", encoding="utf-8")
    text, error = extract_text(str(target))
    assert error is None
    assert "heading" in text


def test_extract_docx_basic(tmp_path: Path) -> None:
    if importlib.util.find_spec("docx") is None:
        pytest.skip("python-docx is not installed")
    from docx import Document

    target = tmp_path / "doc.docx"
    document = Document()
    _ = document.add_paragraph("first paragraph")
    _ = document.add_paragraph("second paragraph")
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
    """The canonical set is the single source of truth.

    Assert against it directly rather than parsing the dialog-filter strings.
    """
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


def _write_pdf(target: Path, *pages: str) -> None:
    """Write a minimal PDF whose pages each draw one line of text."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    font = 3 + 2 * len(pages)
    for i, line in enumerate(pages):
        stream = f"BT /F1 12 Tf 72 720 Td ({line}) Tj ET".encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {4 + 2 * i} 0 R >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    _ = target.write_bytes(bytes(out))


def test_extract_pdf_joins_the_text_of_every_page(tmp_path: Path) -> None:
    target = tmp_path / "doc.pdf"
    _write_pdf(target, "first page", "second page")

    text, error = extract_text(str(target))

    assert error is None
    assert text == "first page\n\nsecond page"
    assert analyze_document(str(target), text)["pages"] == 2


def test_a_corrupt_pdf_is_an_error_message_not_a_crash(tmp_path: Path) -> None:
    target = tmp_path / "broken.pdf"
    _ = target.write_bytes(b"not a pdf at all")

    text, error = extract_text(str(target))

    assert text == ""
    assert error is not None
    assert error.startswith("Error extracting text: ")
    assert analyze_document(str(target), "")["pages"] is None


def test_extract_docx_reads_table_rows(tmp_path: Path) -> None:
    from docx import Document

    target = tmp_path / "table.docx"
    document = Document()
    table = document.add_table(rows=2, cols=2)
    for row, cells in zip(table.rows, (("name", "size"), ("report", "")), strict=True):
        for cell, value in zip(row.cells, cells, strict=True):
            cell.text = value
    _ = document.add_table(rows=1, cols=1)  # an empty row adds no line
    document.save(str(target))

    text, error = extract_text(str(target))

    assert error is None
    assert text == "name | size\n\nreport"


def test_analyze_document_without_a_readable_file_has_no_encoding(tmp_path: Path) -> None:
    stats = analyze_document(str(tmp_path / "gone.txt"), "")

    assert stats["encoding"] is None
