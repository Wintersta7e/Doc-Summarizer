# SPDX-License-Identifier: MIT
"""``scripts/check_binary.py``: the built-binary smoke run and size budget.

The smoke check runs against stand-in apps (a short Python script per
behaviour) and against the real app through ``run.py``; the size budget is
checked on sizes alone.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import check_binary

_REPO = Path(__file__).resolve().parents[1]
# Long enough for a fresh interpreter to start; the hanging stand-in sleeps longer.
_TIMEOUT_S = 30
_SHORT_TIMEOUT_S = 1
_MISSING_FLAG = 3
_STAND_IN = """
import sys
import time
from pathlib import Path

args = sys.argv[1:]
if "--smoke-test" not in args:
    sys.exit({missing_flag})
time.sleep({sleep_s})
if {writes}:
    result = Path(args[args.index("--smoke-result") + 1])
    result.write_text({line!r}, encoding="utf-8")
sys.exit({status})
"""


def _stand_in(
    folder: Path,
    *,
    line: str | None = None,
    status: int = 0,
    sleep_s: float = 0,
) -> list[str]:
    """A command running a fake app that writes ``line`` and exits ``status``.

    Returns:
        The command, interpreter first.
    """
    folder.mkdir(exist_ok=True)
    script = folder / "stand_in.py"
    source = _STAND_IN.format(
        missing_flag=_MISSING_FLAG,
        sleep_s=sleep_s,
        writes=line is not None,
        line=line,
        status=status,
    )
    _ = script.write_text(source, encoding="utf-8")
    return [sys.executable, str(script)]


def test_a_passing_smoke_run_has_no_problems(tmp_path: Path) -> None:
    command = _stand_in(tmp_path, line=check_binary.expected_line())

    assert check_binary.smoke_problems(command, _TIMEOUT_S) == []


@pytest.mark.skipif(importlib.util.find_spec("PySide6") is None, reason="the app needs Qt to start")
def test_the_real_app_passes_the_smoke_check() -> None:
    command = [sys.executable, str(_REPO / "run.py")]

    assert check_binary.smoke_problems(command, _TIMEOUT_S) == []


def test_a_failing_exit_status_is_a_problem(tmp_path: Path) -> None:
    command = _stand_in(tmp_path, line=check_binary.expected_line(), status=1)

    (problem,) = check_binary.smoke_problems(command, _TIMEOUT_S)

    assert problem.startswith("the smoke run exited 1")


def test_another_version_or_no_result_is_a_problem(tmp_path: Path) -> None:
    other = _stand_in(tmp_path / "other", line="DocSummarizer 0.0.1 smoke ok\n")
    silent = _stand_in(tmp_path / "silent")

    (other_problem,) = check_binary.smoke_problems(other, _TIMEOUT_S)
    (silent_problem,) = check_binary.smoke_problems(silent, _TIMEOUT_S)

    assert "0.0.1" in other_problem
    assert silent_problem.startswith("the smoke result is None")


def test_a_smoke_run_past_the_timeout_is_a_problem(tmp_path: Path) -> None:
    command = _stand_in(tmp_path, line=check_binary.expected_line(), sleep_s=_TIMEOUT_S)

    assert check_binary.smoke_problems(command, _SHORT_TIMEOUT_S) == [
        f"the smoke run did not finish in {_SHORT_TIMEOUT_S} s"
    ]


def test_the_size_must_lie_between_the_floor_and_the_budget() -> None:
    measured = check_binary.MEASURED_BYTES["linux"]

    assert check_binary.size_problems(measured, "linux") == []
    assert check_binary.size_problems(int(measured * 1.149), "linux") == []
    (over,) = check_binary.size_problems(int(measured * 1.151), "linux")
    (under,) = check_binary.size_problems(check_binary.MIN_BYTES - 1, "linux")
    (unknown,) = check_binary.size_problems(measured, "plan9")

    assert "over the" in over
    assert "no Qt bundled" in under
    assert "no size measured for plan9" in unknown


def test_main_reports_every_problem_and_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A small file that is no program: under the floor, and it does not start.
    binary = tmp_path / "DocSummarizer"
    _ = binary.write_text("not a program", encoding="utf-8")

    status = check_binary.main([str(binary), "--timeout", str(_TIMEOUT_S)])

    out = capsys.readouterr().out
    assert status == 1
    assert "under the 150 MB floor" in out
    assert "the binary did not start" in out
