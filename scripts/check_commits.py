# SPDX-License-Identifier: MIT
"""Check commit messages: a Conventional Commits subject and no major-version marker.

A subject is ``type(scope): description``, at most 72 characters, with no ``!``
after the type and no trailing ``.`` or ``!``; no body line starts a
``BREAKING CHANGE:`` footer. Either marker would make a release tool bump the
major version. Subjects git writes itself (``Merge …``, ``Revert "…"``,
``fixup!`` and its kin) are not checked.

Usage:
    python scripts/check_commits.py MESSAGE_FILE    # the commit-msg hook
    python scripts/check_commits.py --range BASE HEAD
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

TYPES = ("feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore")
MAX_SUBJECT = 72
_SCOPE = r"[a-z0-9][a-z0-9._/-]*"
_SUBJECT = re.compile(rf"^(?:{'|'.join(TYPES)})(?:\({_SCOPE}(?:,{_SCOPE})*\))?: \S")
_BANG = re.compile(r"^\w+(?:\([^)]*\))?!:")
_BREAKING = re.compile(r"^BREAKING[ -]CHANGE:", re.MULTILINE)
_GENERATED = ("Merge ", 'Revert "', "fixup! ", "squash! ", "amend! ")
# The all-zero id GitHub sends as the base of a branch's first push.
_NO_COMMIT = re.compile(r"^0+$")
# git writes a NUL after each message, which no message holds.
_SEPARATOR = "\x00"


def problems(message: str) -> list[str]:
    """What is wrong with one commit message.

    Returns:
        One line per broken rule; empty when the message passes.
    """
    lines = [line for line in message.splitlines() if not line.startswith("#")]
    subject = next((line.strip() for line in lines if line.strip()), "")
    if subject.startswith(_GENERATED):
        return []
    found: list[str] = []
    if len(subject) > MAX_SUBJECT:
        found.append(f"subject is {len(subject)} characters, over {MAX_SUBJECT}")
    if _BANG.match(subject):
        found.append("no '!' after the type: it marks a major-version change")
    elif not _SUBJECT.match(subject):
        found.append(f"subject is not 'type(scope): description' with a type of {', '.join(TYPES)}")
    if subject.endswith((".", "!")):
        found.append("subject ends with '.' or '!'")
    if _BREAKING.search("\n".join(lines)):
        found.append("no 'BREAKING CHANGE:' footer: it marks a major-version change")
    return found


def _git(repo: Path, *args: str) -> str:
    git = shutil.which("git")
    if git is None:
        msg = "git is not on PATH"
        raise SystemExit(msg)
    return subprocess.run(
        [git, "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def messages(base: str, head: str, repo: Path) -> dict[str, str]:
    """The messages of the commits in ``base..head``, by commit id.

    An empty or all-zero ``base`` (a first push, a manual run) selects ``head`` alone.

    Returns:
        Each commit's full message, oldest first.
    """
    selection = [head, "-1"] if not base or _NO_COMMIT.match(base) else [f"{base}..{head}"]
    out = _git(repo, "log", "--reverse", "--format=%H%n%B%x00", *selection)
    commits: dict[str, str] = {}
    for entry in out.split(_SEPARATOR):
        sha, _, body = entry.strip("\n").partition("\n")
        if sha:
            commits[sha] = body
    return commits


def main(argv: Sequence[str] | None = None, repo: Path | None = None) -> int:
    """Check one message file, or every commit in a range.

    Returns:
        0 when every message passes, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Check commit messages.")
    which = parser.add_mutually_exclusive_group(required=True)
    _ = which.add_argument("message_file", nargs="?", type=Path, help="a commit message file")
    _ = which.add_argument("--range", nargs=2, metavar=("BASE", "HEAD"), help="commits BASE..HEAD")
    args = parser.parse_args(argv)
    rev_range = cast("list[str] | None", args.range)
    if rev_range is None:
        path = cast("Path", args.message_file)
        found = {str(path): problems(path.read_text(encoding="utf-8"))}
    else:
        base, head = rev_range
        commits = messages(base, head, repo or Path.cwd())
        found = {sha[:12]: problems(body) for sha, body in commits.items()}
        print(f"checked {len(commits)} commit message(s)")
    failed = {name: lines for name, lines in found.items() if lines}
    for name, lines in failed.items():
        for line in lines:
            print(f"{name}: {line}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
