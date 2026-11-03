"""Tests for publishing summary files: temp file, create-if-absent, path identity.

Every summary write renders into a temp file beside its target and hands it to
``io_helpers.publish``. These tests pin that a create-if-absent publish never
replaces a file that exists, that no temp file outlives a write, and that
``same_file`` recognises every spelling of one path.
"""

from __future__ import annotations

import errno
import os
import stat
import sys
from pathlib import Path
from typing import NoReturn

import pytest

from docsummarizer import io_helpers
from docsummarizer.io_helpers import publish, same_file, write_summary_txt

SENTINEL = b"a file the user wrote\n"

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX filesystem semantics")


def _partials(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.partial"))


def _temp(folder: Path, content: bytes) -> Path:
    tmp = folder / "out.txt.0123456789abcdef.partial"
    tmp.write_bytes(content)
    return tmp


@pytest.mark.windows
def test_publish_create_if_absent_writes_absent_target(tmp_path: Path) -> None:
    """Catches a create-if-absent publish that never moves the temp into place."""
    target = tmp_path / "out.txt"
    tmp = _temp(tmp_path, b"new summary")

    assert publish(tmp, target, replace=False) is True

    assert target.read_bytes() == b"new summary"
    assert _partials(tmp_path) == []


@pytest.mark.windows
def test_publish_create_if_absent_keeps_existing_target(tmp_path: Path) -> None:
    """Catches create-if-absent publishing done with a replacing rename."""
    target = tmp_path / "out.txt"
    target.write_bytes(SENTINEL)
    before = target.stat().st_mtime_ns
    tmp = _temp(tmp_path, b"new summary")

    assert publish(tmp, target, replace=False) is False

    assert target.read_bytes() == SENTINEL
    assert target.stat().st_mtime_ns == before
    assert _partials(tmp_path) == []


@pytest.mark.windows
def test_publish_replace_swaps_existing_target(tmp_path: Path) -> None:
    """Catches a replacing publish that keeps the old file or leaves the temp."""
    target = tmp_path / "out.txt"
    target.write_bytes(SENTINEL)
    tmp = _temp(tmp_path, b"new summary")

    assert publish(tmp, target, replace=True) is True

    assert target.read_bytes() == b"new summary"
    assert _partials(tmp_path) == []


@posix_only
def test_publish_without_hard_links_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches an ``except OSError`` fallback that writes the target directly."""

    def no_links(src: object, dst: object) -> NoReturn:
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "link", no_links)
    target = tmp_path / "out.txt"
    tmp = _temp(tmp_path, b"new summary")

    with pytest.raises(PermissionError):
        publish(tmp, target, replace=False)

    assert not target.exists()
    assert _partials(tmp_path) == []


@pytest.mark.windows
def test_interrupted_write_leaves_no_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a write path without a ``finally`` that removes its temp file."""

    def interrupted(tmp: Path, target: Path, *, replace: bool) -> NoReturn:
        raise KeyboardInterrupt

    monkeypatch.setattr(io_helpers, "publish", interrupted)
    target = tmp_path / "out.txt"

    with pytest.raises(KeyboardInterrupt):
        write_summary_txt(target, source_name="a.txt", summary="body", replace=False)

    assert not target.exists()
    assert _partials(tmp_path) == []


@posix_only
def test_published_file_mode_matches_plain_open(tmp_path: Path) -> None:
    """Catches a temp created owner-only (``tempfile.mkstemp`` is 0o600)."""
    plain = tmp_path / "plain.txt"
    plain.write_text("x", encoding="utf-8")
    target = tmp_path / "out.txt"

    write_summary_txt(target, source_name="a.txt", summary="body", replace=False)

    assert stat.S_IMODE(target.stat().st_mode) == stat.S_IMODE(plain.stat().st_mode)


@posix_only
def test_same_file_matches_every_spelling_of_one_path(tmp_path: Path) -> None:
    """Catches a plain string comparison: each spelling below names one file."""
    real = tmp_path / "doc.txt"
    real.write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    symlink = tmp_path / "link.txt"
    symlink.symlink_to(real)
    hardlink = tmp_path / "hard.txt"
    os.link(real, hardlink)
    sibling = tmp_path / "other.txt"
    sibling.write_text("x", encoding="utf-8")

    positives = {
        "same path": (real, real),
        "dot-dot segment": (tmp_path / "sub" / ".." / "doc.txt", real),
        "symlink": (symlink, real),
        "hard link": (hardlink, real),
    }
    for label, (a, b) in positives.items():
        print(f"same_file {label}: {same_file(a, b)}")
        assert same_file(a, b), label

    assert not same_file(sibling, real)
    missing = tmp_path / "missing"
    assert same_file(tmp_path / "sub" / ".." / "missing", missing)


@pytest.mark.windows
@pytest.mark.skipif(sys.platform != "win32", reason="case-insensitive paths are Windows-only")
def test_same_file_ignores_case_on_windows(tmp_path: Path) -> None:
    """Catches a comparison without ``normcase`` on a case-insensitive filesystem."""
    p = tmp_path / "Doc.txt"
    p.write_text("x", encoding="utf-8")

    assert same_file(str(p).upper(), p)
