#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check a built DocSummarizer binary: it starts, and its size is in budget.

Runs ``<binary> --smoke-test --smoke-result <file>`` offscreen and passes when it
exits 0 within the timeout and the file holds ``DocSummarizer <version> smoke
ok`` for the installed package's version. The file, not stdout, is the proof:
a windowed Windows build has no stdout. The size must lie between a floor that
a build without Qt falls under (those measured 41 and 64 MB) and the size last
measured on this platform plus 15 %.

Usage:
    python scripts/check_binary.py <binary> [--timeout SECONDS]
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_MB = 1_000_000
# Below this the build is missing its Qt bundle.
MIN_BYTES = 150 * _MB
# A build may grow this far past its measured size before the budget fails.
_GROWTH = 1.15
# Measured sizes, per platform: the v2.0.1 release builds, from 9ed1b48.
# Raise one only in its own commit, saying why the binary grew.
MEASURED_BYTES = {
    "linux": 243_432_120,
    "win32": 198_527_605,
}
_DEFAULT_TIMEOUT_S = 120
_TAIL_LINES = 20
# Offscreen Qt reads fonts from a folder; a frozen Windows build ships none
# (PySide6/lib/fonts), so Qt warns and the smoke test fails on the warning.
# The native platform uses the system's fonts and needs nothing.
_FONTS_DIR = (
    Path(__file__).resolve().parents[1] / "src" / "docsummarizer" / "ui" / "qml" / "App" / "fonts"
)


def size_problems(size: int, platform: str) -> list[str]:
    """What is wrong with a binary of ``size`` bytes built on ``platform``.

    Returns:
        One line per problem; empty when the size is in budget.
    """
    measured = MEASURED_BYTES.get(platform)
    if measured is None:
        return [f"no size measured for {platform}; measure a build and record it"]
    limit = int(measured * _GROWTH)
    if size < MIN_BYTES:
        return [f"{size / _MB:.1f} MB is under the {MIN_BYTES / _MB:.0f} MB floor: no Qt bundled?"]
    if size > limit:
        return [f"{size / _MB:.1f} MB is over the {limit / _MB:.1f} MB budget"]
    return []


def expected_line() -> str:
    """The line a passing smoke run of this checkout's version writes."""
    return f"DocSummarizer {importlib.metadata.version('docsummarizer')} smoke ok\n"


def _tail(text: str) -> str:
    return "\n".join(text.splitlines()[-_TAIL_LINES:])


def smoke_problems(command: list[str], timeout_s: float) -> list[str]:
    """What is wrong with ``command --smoke-test``, run offscreen.

    Returns:
        One entry per problem; empty when the smoke run passed.
    """
    env = {
        **os.environ,
        "QT_QPA_PLATFORM": "offscreen",
        "QT_QUICK_BACKEND": "software",
        "QT_QPA_FONTDIR": str(_FONTS_DIR),
    }
    with tempfile.TemporaryDirectory(prefix="check-binary-") as tmp:
        result = Path(tmp) / "smoke-result.txt"
        args = [*command, "--smoke-test", "--smoke-result", str(result)]
        try:
            done = subprocess.run(
                args, capture_output=True, text=True, env=env, timeout=timeout_s, check=False
            )
        except subprocess.TimeoutExpired:
            return [f"the smoke run did not finish in {timeout_s:g} s"]
        except OSError as exc:
            return [f"the binary did not start: {exc}"]
        written = result.read_text(encoding="utf-8") if result.is_file() else None
    problems: list[str] = []
    if done.returncode != 0:
        problems.append(f"the smoke run exited {done.returncode}:\n{_tail(done.stderr)}")
    if written != expected_line():
        problems.append(f"the smoke result is {written!r}, not {expected_line()!r}")
    return problems


class _Args(argparse.Namespace):
    """The parsed command line."""

    binary: Path = Path()
    timeout: float = _DEFAULT_TIMEOUT_S


def main(argv: list[str] | None = None) -> int:
    """Check the binary the command line names; 0 when every check passes."""
    parser = argparse.ArgumentParser(description="Smoke-test a built binary and check its size.")
    _ = parser.add_argument("binary", type=Path)
    _ = parser.add_argument("--timeout", type=float, default=_Args.timeout, metavar="SECONDS")
    args = parser.parse_args(argv, namespace=_Args())

    problems = size_problems(args.binary.stat().st_size, sys.platform)
    problems += smoke_problems([str(args.binary.resolve())], args.timeout)
    for problem in problems:
        print(f"FAILED: {problem}")
    if not problems:
        print(f"ok: {args.binary.name} starts and is in budget")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
