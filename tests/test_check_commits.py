# SPDX-License-Identifier: MIT
"""``scripts/check_commits.py``: the commit-subject rules, a message file and a range.

The range is read from a throwaway git repository whose commits skip every hook.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

import check_commits

if TYPE_CHECKING:
    from pathlib import Path

_GOOD = "fix(ui,paths): keep the smoke run out of the user's app data\n\n- why\n"


@pytest.mark.parametrize(
    "message",
    [
        _GOOD,
        "feat: add a flag",
        "chore(deps): bump a pin",
        "# a comment git drops\nci(pre-commit): run the gate\n",
        "Merge pull request #12 from a branch",
        'Revert "feat: add a flag"',
        "fixup! feat: add a flag",
        "test: the longest subject allowed " + "x" * (check_commits.MAX_SUBJECT - 34),
        "docs: say why\n\n- a body line that says BREAKING CHANGE: in passing\n",
    ],
)
def test_a_conventional_message_passes(message: str) -> None:
    assert check_commits.problems(message) == []


@pytest.mark.parametrize(
    ("message", "rule"),
    [
        ("Add a flag", "not 'type(scope): description'"),
        ("feature: add a flag", "not 'type(scope): description'"),
        ("fix(UI): add a flag", "not 'type(scope): description'"),
        ("fix(ui, paths): add a flag", "not 'type(scope): description'"),
        ("fix:add a flag", "not 'type(scope): description'"),
        ("", "not 'type(scope): description'"),
        ("feat!: drop a flag", "no '!' after the type"),
        ("feat(cli)!: drop a flag", "no '!' after the type"),
        ("fix: add a flag.", "ends with '.' or '!'"),
        ("fix: add a flag!", "ends with '.' or '!'"),
        ("test: one too long " + "x" * (check_commits.MAX_SUBJECT - 18), "over 72"),
        ("feat: drop a flag\n\nBREAKING CHANGE: the flag is gone\n", "'BREAKING CHANGE:'"),
        ("feat: drop a flag\n\nBREAKING-CHANGE: the flag is gone\n", "'BREAKING CHANGE:'"),
    ],
)
def test_a_message_breaking_a_rule_is_named(message: str, rule: str) -> None:
    found = check_commits.problems(message)

    assert len(found) == 1
    assert rule in found[0]


def test_a_message_file_is_checked(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = tmp_path / "good"
    _ = good.write_text(_GOOD, encoding="utf-8")
    bad = tmp_path / "bad"
    _ = bad.write_text("feat!: drop a flag\n", encoding="utf-8")

    assert check_commits.main([str(good)]) == 0
    assert check_commits.main([str(bad)]) == 1
    assert f"{bad}: no '!' after the type" in capsys.readouterr().err


def _git(repo: Path, *args: str) -> str:
    git = shutil.which("git")
    assert git is not None, "git is not on PATH"
    # No hooks: the user's commit hooks have no say over a throwaway repository.
    config = ["-c", "core.hooksPath=/dev/null", "-c", "user.name=Test"]
    config += ["-c", "user.email=test@example.invalid"]
    return subprocess.run(
        [git, "-C", str(repo), *config, *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _ = _git(repo, "commit", "-q", "--allow-empty", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository holding one bad commit, then a good, a bad and a good one."""
    _ = _git(tmp_path, "init", "-q")
    _ = _commit(tmp_path, "Initial commit")
    _ = _git(tmp_path, "tag", "base")
    _ = _commit(tmp_path, "feat: add a flag\n\n- why")
    _ = _commit(tmp_path, "feat!: drop a flag")
    _ = _commit(tmp_path, "fix: keep the flag\n\n- why")
    return tmp_path


def test_a_range_reads_each_commit_after_the_base_oldest_first(repo: Path) -> None:
    commits = check_commits.messages("base", "HEAD", repo)

    assert list(commits.values()) == [
        "feat: add a flag\n\n- why",
        "feat!: drop a flag",
        "fix: keep the flag\n\n- why",
    ]
    assert list(commits)[-1] == _git(repo, "rev-parse", "HEAD")


@pytest.mark.parametrize("base", ["", "0" * 40])
def test_a_range_without_a_base_reads_the_head_commit_alone(repo: Path, base: str) -> None:
    assert list(check_commits.messages(base, "HEAD", repo).values()) == [
        "fix: keep the flag\n\n- why"
    ]


def test_a_range_fails_on_its_one_bad_commit(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = _git(repo, "rev-parse", "HEAD~1")

    assert check_commits.main(["--range", "base", "HEAD"], repo) == 1
    out = capsys.readouterr()
    assert "checked 3 commit message(s)" in out.out
    assert out.err == f"{bad[:12]}: no '!' after the type: it marks a major-version change\n"
    assert check_commits.main(["--range", "HEAD~1", "HEAD"], repo) == 0
