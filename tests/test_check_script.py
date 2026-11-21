# SPDX-License-Identifier: MIT
"""Tests for scripts/check.py, the one gate entry point."""

from __future__ import annotations

import check
import pytest

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
