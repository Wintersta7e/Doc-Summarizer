"""Published claims the repo must keep true.

Each test reads tracked files as they stand in the working tree, prints what it
scanned and fails when it scanned nothing. Phrases match whitespace-insensitively,
so re-wrapping a line cannot hide one.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SELF = Path(__file__).resolve().relative_to(_ROOT).as_posix()

_REMOVED_FILES = ("setup_and_run.sh", "setup_and_run.bat", "requirements.lock")
_REMOVED_FILE_REF = re.compile(rb"setup_and_run|requirements\.lock", re.IGNORECASE)
# Phrases of the removed lockfile section only: a hash-locked requirements file
# for release builds may be documented with the same tool and the same word.
_LOCKFILE_PROCEDURE = (
    "Reproducible installs (lockfile)",
    "pip install -r requirements.lock",
    "-o requirements.lock",
    "Dependabot will refresh the lockfile",
)
# A comment may wrap mid-phrase ("the setup" / "# scripts"), so these span a "#".
_PYPROJECT_STALE = (
    re.compile(r"setup\s*(#\s*)?scripts", re.IGNORECASE),
    re.compile(r"narrow\s*(#\s*)?matrix", re.IGNORECASE),
)


def _read(rel: str) -> str:
    """Return a repo file's text, printing it as scanned; an empty file fails."""
    text = (_ROOT / rel).read_text(encoding="utf-8")
    print(f"scanned {rel}: {len(text.splitlines())} lines")
    assert text.strip(), f"{rel} is empty"
    return text


def _flat(text: str) -> str:
    """Collapse whitespace runs to one space."""
    return " ".join(text.split())


def _hits(text: str, phrases: Sequence[str]) -> list[str]:
    """Return the phrases that occur in *text*, ignoring case and line wraps."""
    flat = _flat(text).casefold()
    return [phrase for phrase in phrases if _flat(phrase).casefold() in flat]


def _tracked_files() -> list[str]:
    """Return every path git tracks, printing the count; none listed fails."""
    listing = subprocess.run(
        ["git", "ls-files", "-z"],  # noqa: S607 - the checkout's git, found on PATH
        cwd=_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    paths = [rel for rel in listing.decode("utf-8").split("\0") if rel]
    print(f"tracked files: {len(paths)}")
    assert paths, "git ls-files listed 0 files"
    return paths


def test_setup_scripts_and_lockfile_are_gone() -> None:
    """The setup scripts and the lockfile are gone, and no tracked file names them.

    Restoring any of the three files, or any line naming one (such as
    ``pip install -r requirements.lock``), fails this test.
    """
    tracked = _tracked_files()
    present = [name for name in _REMOVED_FILES if name in tracked or (_ROOT / name).exists()]
    scanned = [rel for rel in tracked if rel != _SELF and (_ROOT / rel).is_file()]
    refs = [
        f"{rel}:{number}"
        for rel in scanned
        for number, line in enumerate((_ROOT / rel).read_bytes().splitlines(), start=1)
        if _REMOVED_FILE_REF.search(line)
    ]
    print(f"files scanned: {len(scanned)}")
    print(f"references found: {refs}")
    assert scanned, "scanned 0 files"
    assert not present, f"still present: {present}"
    assert not refs, f"still referenced: {refs}"


def test_development_has_no_lockfile_procedure() -> None:
    """DEVELOPMENT.md carries no ``requirements.lock`` procedure.

    Restoring the "Reproducible installs (lockfile)" heading, its install or
    regenerate command, or its Dependabot sentence fails this test.
    """
    hits = _hits(_read("DEVELOPMENT.md"), _LOCKFILE_PROCEDURE)
    assert not hits, f"lockfile procedure still in DEVELOPMENT.md: {hits}"


def test_pyproject_names_no_setup_scripts() -> None:
    """pyproject.toml no longer sends users to the setup scripts.

    Restoring the old ``runtime`` comment fails this test, even wrapped across two
    comment lines, and so does its stale "narrow matrix" wheel claim.
    """
    text = _read("pyproject.toml")
    hits = [match.group(0) for pattern in _PYPROJECT_STALE for match in pattern.finditer(text)]
    assert not hits, f"stale claims in pyproject.toml: {hits}"


_README_TIMINGS = (
    "Approximate CPU timings",
    "Approx. CPU time",
    "A CUDA build is several times faster",
)
_DEVELOPMENT_GPU_CLAIMS = ("Performance Comparison", "no local CUDA toolkit required")


def test_readme_names_no_macos() -> None:
    """README's platform badge and System Requirements OS row name no macOS.

    Restoring ``macOS`` to either line fails this test, and deleting the badge or
    the row instead fails its locator (0 found).
    """
    lines = _read("README.md").splitlines()
    badges = [line for line in lines if "img.shields.io/badge/platform-" in line]
    os_rows = [line for line in lines if line.startswith("| OS |")]
    print(f"platform badges: {len(badges)}; OS rows: {len(os_rows)}")
    assert len(badges) == 1, f"expected one platform badge, found {len(badges)}"
    assert len(os_rows) == 1, f"expected one OS row, found {len(os_rows)}"
    macos = [line for line in badges + os_rows if "macos" in line.casefold()]
    assert not macos, f"README still names macOS: {macos}"


def test_readme_has_no_unsourced_timings() -> None:
    """README carries no CPU timing table and no CUDA speed claim.

    Restoring the "Approx. CPU time" table, its lead-in sentence or "A CUDA build
    is several times faster" fails this test.
    """
    hits = _hits(_read("README.md"), _README_TIMINGS)
    assert not hits, f"unsourced timings still in README: {hits}"


def test_development_has_no_unsourced_gpu_claims() -> None:
    """DEVELOPMENT.md has no GPU timing table and no "no CUDA toolkit" promise.

    Restoring the "Performance Comparison" table or the phrase "no local CUDA
    toolkit required" fails this test.
    """
    hits = _hits(_read("DEVELOPMENT.md"), _DEVELOPMENT_GPU_CLAIMS)
    assert not hits, f"unsourced GPU claims still in DEVELOPMENT.md: {hits}"
