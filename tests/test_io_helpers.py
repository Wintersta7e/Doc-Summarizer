"""Tests for `docsummarizer.io_helpers`."""

from __future__ import annotations

from pathlib import Path

import pytest

from docsummarizer.io_helpers import write_summary_txt


def test_write_summary_txt_minimal(tmp_path: Path) -> None:
    out = tmp_path / "s.txt"
    write_summary_txt(out, source_name="doc.pdf", summary="the summary body", replace=True)
    content = out.read_text(encoding="utf-8")
    assert content.startswith("Summary of: doc.pdf\n")
    assert "Type:" not in content  # not provided -> not written
    assert "=" * 50 in content
    assert "the summary body" in content


def test_write_summary_txt_with_type(tmp_path: Path) -> None:
    out = tmp_path / "s.txt"
    write_summary_txt(
        out,
        source_name="doc.pdf",
        summary="body",
        summary_type="brief",
        replace=True,
    )
    content = out.read_text(encoding="utf-8")
    assert "Summary of: doc.pdf\n" in content
    assert "Type: brief\n" in content


def test_write_summary_txt_custom_separator(tmp_path: Path) -> None:
    out = tmp_path / "s.txt"
    write_summary_txt(out, source_name="x", summary="y", separator_width=60, replace=True)
    content = out.read_text(encoding="utf-8")
    assert "=" * 60 in content
    assert "=" * 61 not in content


def test_write_summary_txt_accepts_pathlike(tmp_path: Path) -> None:
    out = tmp_path / "s.txt"
    write_summary_txt(out, source_name="x", summary="y", replace=True)
    assert out.exists()


def test_write_summary_txt_accepts_string_path(tmp_path: Path) -> None:
    out = tmp_path / "s.txt"
    write_summary_txt(str(out), source_name="x", summary="y", replace=True)
    assert out.exists()


def test_write_summary_docx_creates_file(tmp_path: Path) -> None:
    pytest.importorskip("docx")
    from docsummarizer.io_helpers import write_summary_docx

    out = tmp_path / "s.docx"
    write_summary_docx(out, source_name="doc.pdf", summary="body text 42", replace=True)
    assert out.exists()

    # Round-trip: read back and check that both the heading (which uses
    # style "Title" at level 0, not "Heading N") and the body paragraph
    # made it into the file.
    import docx as python_docx

    document = python_docx.Document(str(out))
    all_text = "\n".join(p.text for p in document.paragraphs)
    assert "doc.pdf" in all_text
    assert "body text 42" in all_text
