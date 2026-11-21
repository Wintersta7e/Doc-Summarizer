# SPDX-License-Identifier: MIT
"""``scripts/mutation.py``: which mutants a change selects, and how they score.

mutmut itself is not run here; its summary counts and the package paths are
given, and the score, the name patterns and the threshold check follow from
them. The changed files come from a throwaway git repository.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

import mutation

if TYPE_CHECKING:
    from pathlib import Path

_STATS = {
    "killed": 70,
    "timeout": 5,
    "survived": 20,
    "no_tests": 4,
    "suspicious": 1,
    "segfault": 0,
    "skipped": 3,
    "total": 110,
    "check_was_interrupted_by_user": 0,
}
_PACKAGE = "src/docsummarizer"
_FUNCTION = "def f() -> int:\n    return 1\n"


def test_a_score_counts_killed_and_timed_out_mutants_as_caught() -> None:
    result = mutation.score(_STATS)

    assert result == mutation.Score(caught=75, checked=100)
    assert result.percent == pytest.approx(75)


def test_a_run_with_no_mutant_scores_full() -> None:
    assert mutation.Score(caught=0, checked=0).percent == pytest.approx(100)


def test_patterns_select_each_changed_package_module() -> None:
    paths = [
        f"{_PACKAGE}/ui/bridge.py",
        f"{_PACKAGE}/provenance.py",
        f"{_PACKAGE}/__init__.py",
        f"{_PACKAGE}/ui/qml/App/Main.qml",
        "scripts/check.py",
    ]

    assert mutation.mutant_patterns(paths) == [
        "docsummarizer.x*",
        "docsummarizer.provenance.*",
        "docsummarizer.ui.bridge.*",
    ]


def _git(repo: Path, *args: str) -> None:
    git = shutil.which("git")
    assert git is not None, "git is not on PATH"
    # No hooks: the user's commit hooks have no say over a throwaway repository.
    config = ["-c", "core.hooksPath=/dev/null", "-c", "user.name=Test"]
    config += ["-c", "user.email=test@example.invalid"]
    _ = subprocess.run([git, "-C", str(repo), *config, *args], check=True)


def _commit(repo: Path, files: dict[str, str | None]) -> None:
    """Write (or, for None, delete) ``files`` in ``repo`` and commit them."""
    for name, text in files.items():
        path = repo / name
        if text is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            _ = path.write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "change")


def test_changed_paths_are_the_package_modules_with_a_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _git(tmp_path, "init", "-q")
    _commit(tmp_path, {f"{_PACKAGE}/kept.py": _FUNCTION, f"{_PACKAGE}/gone.py": _FUNCTION})
    _git(tmp_path, "tag", "base")
    _commit(
        tmp_path,
        {
            f"{_PACKAGE}/kept.py": _FUNCTION + "\n\nX = 2\n",
            f"{_PACKAGE}/gone.py": None,
            f"{_PACKAGE}/ui/new.py": "class A:\n    async def g(self) -> None: ...\n",
            f"{_PACKAGE}/__init__.py": '__version__ = "1.0"\n',
            f"{_PACKAGE}/ui/qml/App/Main.qml": "Item {}\n",
            "scripts/tool.py": _FUNCTION,
        },
    )
    monkeypatch.chdir(tmp_path)

    changed = sorted(mutation.changed_paths("base"))

    assert changed == [f"{_PACKAGE}/kept.py", f"{_PACKAGE}/ui/new.py"]


def _nothing_changed(_ref: str) -> list[str]:
    return []


def _no_run(_patterns: object) -> None:
    pytest.fail("mutmut ran for a change with no package module")


def _no_mutmut(*_args: str) -> int:
    return 0


def test_a_change_to_no_package_module_passes_without_running(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(mutation, "changed_paths", _nothing_changed)
    monkeypatch.setattr(mutation, "run", _no_run)

    assert mutation.main(["--changed", "HEAD~1"]) == 0
    assert "nothing to mutate" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("caught", "passes"),
    [(mutation.MUTATION_THRESHOLD, True), (mutation.MUTATION_THRESHOLD - 1, False)],
)
def test_the_run_fails_under_the_threshold(
    monkeypatch: pytest.MonkeyPatch, *, caught: int, passes: bool
) -> None:
    def scored(_patterns: object) -> mutation.Score:
        return mutation.Score(caught=caught, checked=100)

    monkeypatch.setattr(mutation, "run", scored)
    monkeypatch.setattr(mutation, "_mutmut", _no_mutmut)

    assert (mutation.main(["--all"]) == 0) is passes


def test_a_failed_mutmut_run_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed(_patterns: object) -> None:
        return None

    monkeypatch.setattr(mutation, "run", failed)

    assert mutation.main(["--all"]) == 1
