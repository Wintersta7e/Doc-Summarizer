# SPDX-License-Identifier: MIT
"""Tests for scripts/check_qml.py: qmllint findings and qmlformat changes are fatal."""

from __future__ import annotations

from typing import TYPE_CHECKING

import check
import check_qml

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

FORMATTED = "import QtQuick\n\nItem {\n    width: 10\n}\n"


def test_every_app_qml_file_and_the_preview_shell_are_covered() -> None:
    """The checks cover each QML file under the app's qml dir and the preview shell."""
    files = check_qml.qml_files()

    assert files[-1] == check_qml.PREVIEW
    assert set(files[:-1]) == set(check_qml.QML_ROOT.rglob("*.qml"))
    assert all(path.is_absolute() for path in files)


def test_an_info_level_finding_fails_the_lint(tmp_path: Path) -> None:
    """An unused import, which qmllint reports only as info by default, is fatal."""
    qml = tmp_path / "Unused.qml"
    _ = qml.write_text("import QtQuick\nimport QtQuick.Layouts\n\nItem {}\n", encoding="utf-8")

    assert check.run_command(check_qml.lint_command([qml])) != 0


def test_a_clean_file_passes_the_lint(tmp_path: Path) -> None:
    """The same command passes a file with no finding."""
    qml = tmp_path / "Clean.qml"
    _ = qml.write_text(FORMATTED, encoding="utf-8")

    assert check.run_command(check_qml.lint_command([qml])) == 0


def test_a_format_change_fails_with_its_diff_and_leaves_the_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Qmlformat runs over a copy: the check prints the diff and writes nothing."""
    qml = tmp_path / "Messy.qml"
    messy = "import QtQuick\nItem { width: 10 }\n"
    _ = qml.write_text(messy, encoding="utf-8")

    assert check_qml.check_format([qml], check.run_command) == 1
    assert qml.read_text(encoding="utf-8") == messy
    captured = capsys.readouterr()
    assert "+    width: 10" in captured.out
    assert "qmlformat changes 1 file(s)" in captured.err


def test_a_formatted_file_passes_the_format_check(tmp_path: Path) -> None:
    """A file already in qmlformat's layout passes."""
    qml = tmp_path / "Clean.qml"
    _ = qml.write_text(FORMATTED, encoding="utf-8")

    assert check_qml.check_format([qml], check.run_command) == 0


def test_a_failing_formatter_fails_the_check(tmp_path: Path) -> None:
    """Qmlformat's own failure is the check's status."""
    qml = tmp_path / "Clean.qml"
    _ = qml.write_text(FORMATTED, encoding="utf-8")
    failed = 4

    assert check_qml.check_format([qml], lambda _command: failed) == failed


def test_main_runs_the_lint_or_the_format_check() -> None:
    """Without flags main runs qmllint; with --format it runs qmlformat."""
    ran: list[check.Command] = []

    def record(command: check.Command) -> int:
        ran.append(command)
        return 0

    assert check_qml.main([], record) == 0
    assert check_qml.main(["--format"], record) == 0
    assert [command[0] for command in ran] == [check_qml.QMLLINT, check_qml.QMLFORMAT]
