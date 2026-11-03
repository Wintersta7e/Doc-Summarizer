"""Tests for publishing summary files: temp file, create-if-absent, path identity.

Every summary write renders into a temp file beside its target and hands it to
``io_helpers.publish``. These tests pin that a create-if-absent publish never
replaces a file that exists, that no temp file outlives a write, and that
``same_file`` recognises every spelling of one path. The CLI tests below run
``cli.main()`` against a fake model and pin folder-run naming, skipping and
refusals.
"""

from __future__ import annotations

import errno
import os
import socket
import stat
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import NoReturn

import pytest

from docsummarizer import cli, io_helpers
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


# --------------------------------------------------------------------------- #
# CLI folder runs
# --------------------------------------------------------------------------- #


class _CliHarness:
    """A fake model for ``cli.main()``: counts loads and calls, never touches a network."""

    def __init__(self) -> None:
        self.constructions = 0
        self.calls = 0
        self.connections = 0
        self.hook: Callable[[str], str | None] | None = None


@pytest.fixture
def cli_harness(monkeypatch: pytest.MonkeyPatch) -> Iterator[_CliHarness]:
    harness = _CliHarness()

    class FakeSummarizer:
        def __init__(self, model_path: object, n_threads: int | None = None, n_gpu_layers: int = 0):
            harness.constructions += 1

        def summarize(
            self, text: str, summary_type: str = "detailed", language: str = "auto"
        ) -> str:
            harness.calls += 1
            if harness.hook is not None:
                result = harness.hook(text)
                if result is not None:
                    return result
            return f"summary of {text}"

        def close(self) -> None:
            pass

    def no_network(*args: object, **kwargs: object) -> NoReturn:
        harness.connections += 1
        raise OSError("network access is not allowed in this test")

    monkeypatch.setattr(cli, "ensure_model", lambda: True)
    monkeypatch.setattr(cli, "Summarizer", FakeSummarizer)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    yield harness
    assert harness.connections == 0


def _run_cli(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int | str | None:
    monkeypatch.setattr(sys, "argv", ["docsummarizer-cli", *argv])
    try:
        cli.main()
    except SystemExit as exc:
        return exc.code
    return 0


def _make_inputs(folder: Path, *names: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_text(f"content of {name}", encoding="utf-8")
    return folder


def _snapshot(folder: Path) -> dict[Path, bytes | None]:
    return {p: (p.read_bytes() if p.is_file() else None) for p in sorted(folder.rglob("*"))}


def test_folder_run_never_replaces_unchecked_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_harness: _CliHarness,
) -> None:
    """Catches a folder run without the existence check: it replaces both files."""
    src = _make_inputs(tmp_path / "in", "a.txt", "b.md", "c.txt")
    out = tmp_path / "out"
    out.mkdir()
    planned = out / "c.txt.detailed.txt"
    planned.write_bytes(SENTINEL)
    user_file = out / "a_summary.txt"
    user_file.write_bytes(SENTINEL)
    mtimes = {p: p.stat().st_mtime_ns for p in (planned, user_file)}

    assert _run_cli(monkeypatch, str(src), "-o", str(out)) == 0
    printed = capsys.readouterr().out

    identical = sum(
        p.read_bytes() == SENTINEL and p.stat().st_mtime_ns == mtimes[p] for p in mtimes
    )
    written = sum((out / n).exists() for n in ("a.txt.detailed.txt", "b.md.detailed.txt"))
    print(f"written={written} preexisting_identical={identical}")
    assert written == 2
    assert identical == 2
    assert "c.txt: skipped: summary exists" in printed
    assert cli_harness.calls == 2


def test_named_single_input_replaces_its_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cli_harness: _CliHarness
) -> None:
    """Catches a named single input treated like a folder run (skipped, not written)."""
    _make_inputs(tmp_path, "a.txt")
    target = tmp_path / "out.txt"
    target.write_bytes(SENTINEL)

    assert _run_cli(monkeypatch, str(tmp_path / "a.txt"), "-o", str(target)) == 0

    assert target.read_text(encoding="utf-8").endswith("summary of content of a.txt")


def test_folder_run_overwrite_replaces_planned_outputs_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cli_harness: _CliHarness
) -> None:
    """Catches --overwrite being ignored, or replacing a file the run did not plan."""
    src = _make_inputs(tmp_path / "in", "a.txt", "b.md", "c.txt")
    out = tmp_path / "out"
    out.mkdir()
    planned = out / "c.txt.detailed.txt"
    planned.write_bytes(SENTINEL)
    user_file = out / "a_summary.txt"
    user_file.write_bytes(SENTINEL)

    assert _run_cli(monkeypatch, str(src), "-o", str(out), "--overwrite") == 0

    assert planned.read_text(encoding="utf-8").endswith("summary of content of c.txt")
    assert user_file.read_bytes() == SENTINEL
    assert cli_harness.calls == 3


def test_folder_run_outputs_are_distinct_per_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cli_harness: _CliHarness
) -> None:
    """Catches stem-based names: a.txt and a.md would share one output."""
    src = _make_inputs(tmp_path / "in", "a.txt", "a.md")
    out = tmp_path / "out"

    assert _run_cli(monkeypatch, str(src), "-o", str(out)) == 0

    assert sorted(p.name for p in out.iterdir()) == ["a.md.detailed.txt", "a.txt.detailed.txt"]


@pytest.mark.parametrize("trailing_sep", [False, True], ids=["plain", "trailing-sep"])
def test_folder_output_missing_dir_becomes_a_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cli_harness: _CliHarness,
    trailing_sep: bool,
) -> None:
    """Catches a missing -o treated as a file: every summary lands in one file."""
    src = _make_inputs(tmp_path / "in", "a.txt", "b.md", "c.txt")
    out = tmp_path / "summaries"
    out_arg = str(out) + (os.sep if trailing_sep else "")

    assert _run_cli(monkeypatch, str(src), "-o", out_arg) == 0

    assert out.is_dir()
    assert len(list(out.iterdir())) == 3


def test_folder_run_never_scans_its_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_harness: _CliHarness,
) -> None:
    """Catches a recursive scan: the second run would summarize the summaries."""
    src = _make_inputs(tmp_path / "in", "a.txt", "b.md", "c.txt")
    out = src / "summaries"
    scanned: list[int] = []
    real_find = cli.find_documents

    def spy(path: Path) -> list[Path]:
        found = real_find(path)
        scanned.append(len(found))
        return found

    monkeypatch.setattr(cli, "find_documents", spy)

    assert _run_cli(monkeypatch, str(src), "-o", str(out)) == 0
    first_calls = cli_harness.calls
    capsys.readouterr()
    assert _run_cli(monkeypatch, str(src), "-o", str(out)) == 0
    second = capsys.readouterr().out

    print(f"scanned per run: {scanned}")
    assert scanned == [3, 3]
    assert first_calls == 3
    assert cli_harness.calls == first_calls  # the second run wrote nothing
    assert second.count(": skipped: summary exists") == 3


@pytest.mark.parametrize("case", ["folder", "folder-dot-dot", "file"])
def test_output_equal_to_input_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_harness: _CliHarness,
    case: str,
) -> None:
    """Catches a missing output-equals-input check (the file case replaces a.txt)."""
    src = _make_inputs(tmp_path / "in", "a.txt", "b.md", "c.txt")
    (src / "sub").mkdir()
    argv = {
        "folder": (str(src), "-o", str(src)),
        "folder-dot-dot": (str(src), "-o", str(src / "sub" / "..")),
        "file": (str(src / "a.txt"), "-o", str(src / "a.txt")),
    }[case]
    before = _snapshot(src)

    assert _run_cli(monkeypatch, *argv) == 2

    assert cli_harness.constructions == 0
    assert _snapshot(src) == before
    assert capsys.readouterr().err.startswith("Error: ")


def test_folder_input_with_existing_file_output_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_harness: _CliHarness,
) -> None:
    """Catches a folder input writing every summary over one existing file."""
    src = _make_inputs(tmp_path / "in", "a.txt", "b.md")
    existing = tmp_path / "existing.txt"
    existing.write_bytes(SENTINEL)

    assert _run_cli(monkeypatch, str(src), "-o", str(existing)) == 2

    assert "a folder input needs an output folder" in capsys.readouterr().err
    assert existing.read_bytes() == SENTINEL
    assert cli_harness.constructions == 0


@pytest.mark.windows
def test_two_writers_appeared_during_run_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_harness: _CliHarness,
) -> None:
    """Catches a folder run publishing over a file that appeared after its check."""
    src = _make_inputs(tmp_path / "in", "a.txt", "b.md", "c.txt")
    out = tmp_path / "out"
    target = out / "b.md.detailed.txt"

    def other_writer(text: str) -> None:
        if text == "content of b.md":
            target.write_bytes(SENTINEL)

    cli_harness.hook = other_writer

    assert _run_cli(monkeypatch, str(src), "-o", str(out)) == 0

    assert target.read_bytes() == SENTINEL
    assert "b.md: skipped (appeared during run)" in capsys.readouterr().out
    assert _partials(out) == []


@pytest.mark.windows
def test_two_writers_concurrent_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_harness: _CliHarness,
) -> None:
    """Catches create-if-absent publishing done with a replacing rename under a real race."""
    names = ("a.txt", "b.md", "c.txt")
    rounds = 20
    violations = 0
    checked = 0
    for round_no in range(rounds):
        src = _make_inputs(tmp_path / f"in{round_no}", *names)
        out = tmp_path / f"out{round_no}"
        barrier = threading.Barrier(2, timeout=10)

        def contested(text: str, barrier: threading.Barrier = barrier) -> str:
            # Both writers have checked their targets before reaching here.
            barrier.wait()
            return f"rendered by {threading.current_thread().name}"

        cli_harness.hook = contested
        monkeypatch.setattr(sys, "argv", ["docsummarizer-cli", str(src), "-o", str(out)])
        threads = [threading.Thread(target=cli.main, name=f"w{i}") for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        printed = capsys.readouterr().out

        for name in names:
            checked += 1
            target = out / f"{name}.detailed.txt"
            saved = printed.count(f"Saved to: {target}")
            appeared = printed.count(f"{name}: skipped (appeared during run)")
            content = target.read_text(encoding="utf-8") if target.exists() else ""
            enders = sum(content.endswith(f"rendered by w{i}") for i in range(2))
            if (saved, appeared, enders) != (1, 1, 1) or _partials(out):
                violations += 1

    print(f"rounds x inputs = {checked}, violations = {violations}")
    assert checked > 0
    assert violations == 0
