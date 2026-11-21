# SPDX-License-Identifier: MIT
"""Lint every QML file with qmllint, or check that qmlformat leaves it unchanged.

Usage::

    python scripts/check_qml.py            # qmllint, every finding fatal
    python scripts/check_qml.py --format   # qmlformat over a copy, any change fatal

The files are every ``.qml`` under ``src/docsummarizer/ui/qml/`` plus the
offscreen preview shell ``scripts/_preview.qml``. Both tools come from the
environment the way ``check.py`` finds its own and ignore any settings file, so
this command line is their whole configuration.
"""

from __future__ import annotations

import argparse
import difflib
import shutil
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import check

if TYPE_CHECKING:
    from collections.abc import Sequence

QML_ROOT = check.REPO / "src" / "docsummarizer" / "ui" / "qml"
PREVIEW = check.REPO / "scripts" / "_preview.qml"
QMLLINT = "pyside6-qmllint"
QMLFORMAT = "pyside6-qmlformat"
# qmllint leaves these categories off or at info level; each is raised to a
# warning, and `--max-warnings 0` makes every warning fatal.
RAISED_CATEGORIES = (
    "attached-property-reuse",
    "deferred-property-id",
    "function-used-before-declaration",
    "import-file-selector",
    "multiline-strings",
    "shadow",
    "unused-imports",
    "use-proper-function",
    "void",
)
# `compiler` stays at qmllint's default (off): it reports which bindings the
# ahead-of-time QML compiler cannot turn into C++. The app never compiles QML
# ahead of time (the engine loads it from source), so those findings describe
# a build this project does not have. 158 findings when enabled.


def qml_files() -> list[Path]:
    """Return every QML file the checks cover, absolute, in a stable order."""
    return [*sorted(QML_ROOT.rglob("*.qml")), PREVIEW]


def lint_command(files: Sequence[Path]) -> check.Command:
    """Return the qmllint command that fails on any finding in files."""
    raised = [arg for name in RAISED_CATEGORIES for arg in (f"--{name}", "warning")]
    # Absolute paths: given relative ones, qmllint misreads the module's qmldir
    # and reports the Theme singleton as undeclared.
    return (
        QMLLINT,
        "--ignore-settings",
        "--max-warnings",
        "0",
        *raised,
        "-I",
        str(QML_ROOT),
        *(str(path) for path in files),
    )


def check_format(files: Sequence[Path], runner: check.Runner) -> int:
    """Format a copy of files with qmlformat; 1 and a diff when any copy changed.

    Args:
        files: The QML files to check; none of them is written.
        runner: Runs one command and returns its exit status.

    Returns:
        0 when qmlformat leaves every file as it is, 1 when it changes one, or
        qmlformat's own status when it fails.
    """
    with tempfile.TemporaryDirectory() as scratch:
        copies = [Path(scratch) / f"{index}-{path.name}" for index, path in enumerate(files)]
        for path, copy in zip(files, copies, strict=True):
            _ = shutil.copyfile(path, copy)
        status = runner((QMLFORMAT, "--ignore-settings", "-i", *(str(copy) for copy in copies)))
        if status != 0:
            return status
        changed = 0
        for path, copy in zip(files, copies, strict=True):
            before = path.read_text(encoding="utf-8").splitlines(keepends=True)
            after = copy.read_text(encoding="utf-8").splitlines(keepends=True)
            if before != after:
                changed += 1
                name = str(path)
                _ = sys.stdout.writelines(
                    difflib.unified_diff(before, after, name, f"{name} (qmlformat)")
                )
    if changed:
        print(f"check_qml.py: qmlformat changes {changed} file(s); run it with -i", file=sys.stderr)
        return 1
    return 0


class Arguments(argparse.Namespace):
    """The parsed command line."""

    format: bool = False


def main(argv: Sequence[str] | None = None, runner: check.Runner = check.run_command) -> int:
    """Run qmllint, or the qmlformat check with ``--format``; return its status."""
    parser = argparse.ArgumentParser(description="Lint or format-check every QML file.")
    _ = parser.add_argument(
        "--format", action="store_true", help="check that qmlformat leaves each file unchanged"
    )
    args = parser.parse_args(argv, namespace=Arguments())
    files = qml_files()
    if args.format:
        return check_format(files, runner)
    return runner(lint_command(files))


if __name__ == "__main__":
    sys.exit(main())
