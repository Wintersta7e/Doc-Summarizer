# SPDX-License-Identifier: MIT
"""Tests for `docsummarizer.paths.app_data_dir`.

Platform-branch coverage lives here (rather than scattered across the
logger/model_manager tests) since both modules now delegate to
`paths.app_data_dir`.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from docsummarizer import paths

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_app_data_dir_linux(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    d = paths.app_data_dir("widgets")
    assert d == tmp_path / "DocSummarizer" / "widgets"
    assert d.exists()


def test_app_data_dir_darwin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("HOME", str(tmp_path))
    d = paths.app_data_dir("logs")
    assert d == tmp_path / "Library" / "Application Support" / "DocSummarizer" / "logs"
    assert d.exists()


def test_app_data_dir_win32(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    d = paths.app_data_dir("models")
    assert d == tmp_path / "DocSummarizer" / "models"
    assert d.exists()


def test_app_data_dir_linux_falls_back_to_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without XDG_DATA_HOME, Linux uses `~/.local/share/...`."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    d = paths.app_data_dir("logs")
    assert d == tmp_path / ".local" / "share" / "DocSummarizer" / "logs"
