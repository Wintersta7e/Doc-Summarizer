# SPDX-License-Identifier: MIT
"""Mutation testing: run mutmut over the package and hold its score to a floor.

``--changed <ref>`` runs the mutants in the package modules changed since the
merge base of ``<ref>`` and ``HEAD``; ``--all`` runs every mutant. Either way
the run starts from a fresh ``mutants/`` copy, so no result from an earlier run
counts, and the score is printed:

    caught / checked

A mutant is caught when a test fails on it or it times out. Checked is every
mutant mutmut ran; a mutant no test reaches counts as checked and not caught.
The script fails when the score is under ``MUTATION_THRESHOLD`` percent, and
then lists the survivors. The threshold only ever goes up.

Usage:
    python scripts/mutation.py --changed <ref> | --all
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

# The score of a full run, rounded down; raise it, never lower it. Measured
# 67.8 % (2250 of 3321 caught) over the whole package in 25 minutes.
MUTATION_THRESHOLD = 67
_PACKAGE = "src/docsummarizer"
_WORK_DIR = Path("mutants")
_STATS = _WORK_DIR / "mutmut-cicd-stats.json"
# The mutmut summary's counts of mutants it ran, by outcome.
_CAUGHT = ("killed", "timeout")
_NOT_CAUGHT = ("survived", "no_tests", "suspicious", "segfault")
_PERCENT = 100
# Each process's data limit during a run. A mutant can turn a loop into one
# that allocates without end; under the limit it dies with MemoryError, a
# failing test, instead of taking the machine's memory (an uncapped full run
# went past 16 GB).
_DATA_LIMIT_BYTES = 4 * 1024**3


@dataclass(frozen=True)
class Score:
    """How many of the mutants run were caught."""

    caught: int
    checked: int

    @property
    def percent(self) -> float:
        """The share caught, 0 to 100; 100 when nothing was run."""
        return _PERCENT * self.caught / self.checked if self.checked else float(_PERCENT)


def score(stats: Mapping[str, int]) -> Score:
    """The score of a mutmut summary (``mutmut export-cicd-stats``'s counts)."""
    caught = sum(stats[key] for key in _CAUGHT)
    return Score(caught=caught, checked=caught + sum(stats[key] for key in _NOT_CAUGHT))


def mutant_patterns(paths: Iterable[str]) -> list[str]:
    """Mutmut name patterns selecting the mutants of the package files ``paths``.

    Returns:
        One pattern per Python file under the package, in path order.
    """
    patterns: list[str] = []
    for path in sorted(paths):
        file = Path(path)
        if file.suffix != ".py" or not file.is_relative_to(_PACKAGE):
            continue
        module = ".".join(file.relative_to("src").with_suffix("").parts)
        # A package's mutants are named after the package, and every mutated
        # function's name starts with `x`; no module here does.
        if file.stem == "__init__":
            module = module.removesuffix(".__init__") + ".x"
        else:
            module += "."
        patterns.append(f"{module}*")
    return patterns


def _git(*args: str) -> str:
    git = shutil.which("git")
    if git is None:
        msg = "git is not on PATH"
        raise SystemExit(msg)
    return subprocess.run(
        [git, *args], check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout


def _defines_a_function(path: Path) -> bool:
    # mutmut mutates only inside functions, and stops when a name pattern
    # selects no mutant (a module of constants, such as the version's).
    lines = path.read_text(encoding="utf-8").splitlines()
    return any(line.lstrip().startswith(("def ", "async def ")) for line in lines)


def changed_paths(ref: str) -> list[str]:
    """The package's Python files changed since the merge base of ``ref`` and ``HEAD``.

    Returns:
        Their paths from the repository root, each holding a function; deleted
        files are left out.
    """
    base = _git("merge-base", ref, "HEAD").strip()
    names = _git("diff", "--name-only", "--diff-filter=d", base, "HEAD", "--", f"{_PACKAGE}/*.py")
    return [name for name in names.splitlines() if _defines_a_function(Path(name))]


def _mutmut(*args: str) -> int:
    return subprocess.run([sys.executable, "-m", "mutmut", *args], check=False).returncode


if sys.platform == "win32":

    def _limit_memory() -> None:
        """Nothing to do: Windows has no per-process data limit, and mutmut does not run there."""

else:
    import resource

    def _limit_memory() -> None:
        """Hold this process, and every process it starts, to ``_DATA_LIMIT_BYTES`` of data."""
        resource.setrlimit(resource.RLIMIT_DATA, (_DATA_LIMIT_BYTES, _DATA_LIMIT_BYTES))


def run(patterns: Sequence[str]) -> Score | None:
    """Run the mutants ``patterns`` select (all when empty) from a fresh copy.

    Returns:
        Their score, or None when mutmut itself failed.
    """
    shutil.rmtree(_WORK_DIR, ignore_errors=True)
    _limit_memory()
    # Half the cores: the machine stays usable while the run goes on.
    children = max(1, (os.cpu_count() or 2) // 2)
    if _mutmut("run", "--max-children", str(children), *patterns) != 0:
        return None
    if _mutmut("export-cicd-stats") != 0:
        return None
    return score(cast("dict[str, int]", json.loads(_STATS.read_text(encoding="utf-8"))))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected mutants and compare their score with the threshold.

    Returns:
        0 when the score reaches ``MUTATION_THRESHOLD``, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Run mutation testing and check its score.")
    which = parser.add_mutually_exclusive_group(required=True)
    _ = which.add_argument("--changed", metavar="REF", help="only modules changed since REF")
    _ = which.add_argument("--all", action="store_true", help="every mutant")
    args = parser.parse_args(argv)
    changed = cast("str | None", args.changed)
    patterns: list[str] = []
    if changed is not None:
        patterns = mutant_patterns(changed_paths(changed))
        if not patterns:
            print(f"no package module changed since {changed}: nothing to mutate")
            return 0
    result = run(patterns)
    if result is None:
        print("mutmut failed; its output is above")
        return 1
    print(
        f"mutation score {result.percent:.1f}% ({result.caught} of {result.checked} caught); "
        f"threshold {MUTATION_THRESHOLD}%"
    )
    if result.percent < MUTATION_THRESHOLD:
        _ = _mutmut("results")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
