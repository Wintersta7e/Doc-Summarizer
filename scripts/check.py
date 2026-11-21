# SPDX-License-Identifier: MIT
"""Run the project's gate, or one step of it.

Usage::

    python scripts/check.py {format,lint,types,arch,qml,test,all}
    python scripts/check.py --per-file-floor


Every tool comes from the environment of the interpreter running this script
(its scripts directory first, then ``PATH``); nothing is installed. Each tool
reads its own configuration from ``pyproject.toml``. ``all`` runs every step in
order and stops at the first red one, naming it. ``--per-file-floor`` reads the
coverage report the ``test`` step writes and fails when any file is below
``FILE_FLOOR`` percent; ``fail_under`` in ``pyproject.toml`` holds the total.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypedDict, cast

REPO = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
# jscpd is an npm tool, outside uv.lock: npx runs exactly this version.
JSCPD_VERSION = "5.4.3"
JSCPD_PATHS = ("src", "scripts", "run.py")
# The exit status a shell gives a command it cannot find.
EXIT_NOT_FOUND = 127
# The coverage report the test step writes and the per-file floor reads.
COVERAGE_JSON = "coverage.json"
# Every measured file covers at least this percent of its lines and branches.
FILE_FLOOR = 80.0

Command = tuple[str, ...]
Runner = Callable[[Command], int]

STEPS: dict[str, tuple[Command, ...]] = {
    "format": (
        ("ruff", "format", "--check", "."),
        (PYTHON, "scripts/check_qml.py", "--format"),
    ),
    "lint": (("ruff", "check", "."),),
    "types": (("basedpyright",),),
    "arch": (
        ("lint-imports",),
        ("deptry", "."),
        ("vulture",),
        (
            "npx",
            "--yes",
            f"jscpd@{JSCPD_VERSION}",
            "--min-lines",
            "6",
            "--exit-code",
            "1",
            *JSCPD_PATHS,
        ),
    ),
    "qml": ((PYTHON, "scripts/check_qml.py"),),
    "test": (
        ("pytest", "--cov", "--cov-report=term-missing", f"--cov-report=json:{COVERAGE_JSON}"),
        (PYTHON, "scripts/check.py", "--per-file-floor"),
    ),
}
ORDER = ("format", "lint", "types", "arch", "qml", "test")


def resolve(tool: str) -> str | None:
    """Return the executable for tool from the active environment, or None."""
    if Path(tool).is_absolute():
        return tool
    return shutil.which(tool, path=sysconfig.get_path("scripts")) or shutil.which(tool)


def run_command(command: Command) -> int:
    """Run command from the repository root and return its exit status."""
    exe = resolve(command[0])
    if exe is None:
        print(f"check.py: {command[0]} is not installed in this environment", file=sys.stderr)
        return EXIT_NOT_FOUND
    return subprocess.run([exe, *command[1:]], cwd=REPO, check=False).returncode


def run_steps(names: Sequence[str], runner: Runner) -> int:
    """Run each named step's commands in order; stop at the first that fails.

    Args:
        names: Step names, each a key of ``STEPS``.
        runner: Runs one command and returns its exit status.

    Returns:
        0 when every command passed, else the failing command's exit status.
    """
    for name in names:
        for command in STEPS[name]:
            print(f"==> {name}: {' '.join(command)}", flush=True)
            status = runner(command)
            if status != 0:
                print(f"check.py: step '{name}' failed (exit {status})", file=sys.stderr)
                return status
    return 0


class _FileSummary(TypedDict):
    percent_covered: float


class _FileReport(TypedDict):
    summary: _FileSummary


class _CoverageReport(TypedDict):
    files: dict[str, _FileReport]


def per_file_floor(report: Path, floor: float = FILE_FLOOR) -> int:
    """Fail when a file in the coverage JSON report covers less than floor percent.

    Args:
        report: A report written by ``coverage json``.
        floor: The lowest percent of lines and branches any one file may cover.

    Returns:
        0 when every file meets the floor, else 1 after naming each that does not.
    """
    data = cast("_CoverageReport", json.loads(report.read_text(encoding="utf-8")))
    below = {
        name: entry["summary"]["percent_covered"]
        for name, entry in sorted(data["files"].items())
        if entry["summary"]["percent_covered"] < floor
    }
    for name, percent in below.items():
        print(
            f"check.py: {name} covers {percent:.1f}%, below the {floor:.0f}% floor", file=sys.stderr
        )
    return 1 if below else 0


class Arguments(argparse.Namespace):
    """The parsed command line."""

    step: str = "all"
    per_file_floor: bool = False


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the step name from argv and run it; ``all`` runs every step."""
    parser = argparse.ArgumentParser(description="Run the project's gate, or one step of it.")
    _ = parser.add_argument("step", nargs="?", default="all", choices=(*ORDER, "all"))
    _ = parser.add_argument(
        "--per-file-floor",
        action="store_true",
        help=f"check {COVERAGE_JSON} against the {FILE_FLOOR:.0f}%% per-file floor",
    )
    args = parser.parse_args(argv, namespace=Arguments())
    if args.per_file_floor:
        return per_file_floor(REPO / COVERAGE_JSON)
    return run_steps(ORDER if args.step == "all" else (args.step,), run_command)


if __name__ == "__main__":
    sys.exit(main())
