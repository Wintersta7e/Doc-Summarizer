# SPDX-License-Identifier: MIT
"""Run the project's gate, or one step of it.

Usage::

    python scripts/check.py {format,lint,types,arch,qml,test,all}

Every tool comes from the environment of the interpreter running this script
(its scripts directory first, then ``PATH``); nothing is installed. Each tool
reads its own configuration from ``pyproject.toml``. ``all`` runs every step in
order and stops at the first red one, naming it.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Callable, Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
# jscpd is an npm tool, outside uv.lock: the gate refuses any other version.
JSCPD_VERSION = "5.4.3"
JSCPD_PATHS = ("src", "scripts", "run.py")
# The exit status a shell gives a command it cannot find.
EXIT_NOT_FOUND = 127

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
        ("jscpd", "--min-lines", "6", "--exit-code", "1", *JSCPD_PATHS),
    ),
    "qml": ((PYTHON, "scripts/check_qml.py"),),
    "test": (("pytest", "--cov"),),
}
ORDER = ("format", "lint", "types", "arch", "qml", "test")


def resolve(tool: str) -> str | None:
    """Return the executable for tool from the active environment, or None."""
    if Path(tool).is_absolute():
        return tool
    return shutil.which(tool, path=sysconfig.get_path("scripts")) or shutil.which(tool)


def version_error(tool: str, exe: str) -> str | None:
    """Return why exe is not the pinned version of tool, or None when it is."""
    if tool != "jscpd":
        return None
    out = subprocess.run([exe, "--version"], capture_output=True, text=True, check=False)
    found = out.stdout.split()[-1] if out.stdout.split() else "nothing"
    if found == JSCPD_VERSION:
        return None
    return f"jscpd {JSCPD_VERSION} is pinned, found {found}"


def run_command(command: Command) -> int:
    """Run command from the repository root and return its exit status."""
    exe = resolve(command[0])
    if exe is None:
        print(f"check.py: {command[0]} is not installed in this environment", file=sys.stderr)
        return EXIT_NOT_FOUND
    if error := version_error(command[0], exe):
        print(f"check.py: {error}", file=sys.stderr)
        return 1
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


class Arguments(argparse.Namespace):
    """The parsed command line."""

    step: str = "all"


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the step name from argv and run it; ``all`` runs every step."""
    parser = argparse.ArgumentParser(description="Run the project's gate, or one step of it.")
    _ = parser.add_argument("step", choices=(*ORDER, "all"))
    step = parser.parse_args(argv, namespace=Arguments()).step
    return run_steps(ORDER if step == "all" else (step,), run_command)


if __name__ == "__main__":
    sys.exit(main())
