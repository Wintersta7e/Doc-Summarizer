# SPDX-License-Identifier: MIT
"""Tests for scripts/check.py, the one gate entry point."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

import check

if TYPE_CHECKING:
    from pathlib import Path

EXPECTED_ORDER = ("format", "lint", "types", "arch", "qml", "test")
FAILED_STATUS = 3
USAGE_ERROR = 2


def recorder(ran: list[check.Command], failing_tool: str | None = None) -> check.Runner:
    """Return a runner that records each command and fails the one naming failing_tool."""

    def run(command: check.Command) -> int:
        ran.append(command)
        return FAILED_STATUS if command[0] == failing_tool else 0

    return run


def test_all_runs_every_step_in_order() -> None:
    """`all` runs format, lint, types, arch, qml, test, each step's commands in turn."""
    ran: list[check.Command] = []

    assert check.ORDER == EXPECTED_ORDER
    assert check.run_steps(check.ORDER, recorder(ran)) == 0
    assert ran == [command for name in EXPECTED_ORDER for command in check.STEPS[name]]


def test_a_failing_step_stops_the_gate_and_exits_non_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The first red command ends the run with its status, naming its step."""
    ran: list[check.Command] = []

    status = check.run_steps(check.ORDER, recorder(ran, failing_tool="basedpyright"))

    assert status == FAILED_STATUS
    assert ran[-1] == ("basedpyright",)
    later = {command for name in ("arch", "qml", "test") for command in check.STEPS[name]}
    assert later.isdisjoint(ran)
    assert "step 'types' failed (exit 3)" in capsys.readouterr().err


def test_a_missing_tool_fails_its_step() -> None:
    """A tool absent from the environment is a red step, not a crash."""
    assert check.run_command(("no-such-gate-tool",)) == check.EXIT_NOT_FOUND


def test_an_unknown_step_is_a_usage_error() -> None:
    """The CLI accepts only the named steps and `all`."""
    with pytest.raises(SystemExit) as exit_info:
        _ = check.main(["deploy"])

    assert exit_info.value.code == USAGE_ERROR


def _coverage_report(folder: Path, percents: dict[str, float]) -> Path:
    """Write a coverage JSON report holding only each file's summary percent."""
    report = folder / "coverage.json"
    files = {name: {"summary": {"percent_covered": pct}} for name, pct in percents.items()}
    _ = report.write_text(json.dumps({"files": files}), encoding="utf-8")
    return report


def test_per_file_floor_names_every_file_below_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A file under the floor fails the gate even when the total passes."""
    report = _coverage_report(tmp_path, {"a.py": 79.9, "b.py": 100.0, "c.py": 12.5})

    assert check.per_file_floor(report) == 1
    err = capsys.readouterr().err
    assert "a.py covers 79.9%, below the 80% floor" in err
    assert "c.py covers 12.5%" in err
    assert "b.py" not in err


def test_per_file_floor_passes_files_at_the_floor(tmp_path: Path) -> None:
    """Exactly the floor is enough."""
    report = _coverage_report(tmp_path, {"a.py": check.FILE_FLOOR, "b.py": 95.0})

    assert check.per_file_floor(report) == 0


def test_the_test_step_writes_the_report_the_floor_reads() -> None:
    """The floor runs after pytest, on the JSON report pytest was told to write."""
    pytest_command, floor_command = check.STEPS["test"]

    assert f"--cov-report=json:{check.COVERAGE_JSON}" in pytest_command
    assert floor_command[1:] == ("scripts/check.py", "--per-file-floor")


def test_the_floor_flag_reads_the_repository_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--per-file-floor`` checks the report in the repository root, not a step."""
    _ = _coverage_report(tmp_path, {"a.py": 50.0})
    monkeypatch.setattr(check, "REPO", tmp_path)

    assert check.main(["--per-file-floor"]) == 1
