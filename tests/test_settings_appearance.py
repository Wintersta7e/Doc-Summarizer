# SPDX-License-Identifier: MIT
"""Tests for the persisted `appearance` (theme) setting.

Appearance was previously not persisted (reset to "System" every launch). The
QML Config screen needs it remembered. It joins n_threads/use_gpu in
settings.json, with tolerant validation against {System, Light, Dark}.
"""

from __future__ import annotations

import json

from docsummarizer.settings import Settings, load_settings, save_settings, settings_path


def test_default_appearance_is_system() -> None:
    assert Settings().appearance == "System"


def test_appearance_round_trips() -> None:
    save_settings(Settings(n_threads=6, use_gpu=True, appearance="Dark"))
    loaded = load_settings()
    assert loaded.appearance == "Dark"
    # Existing fields are unaffected.
    assert loaded.n_threads == 6
    assert loaded.use_gpu is True


def test_save_writes_appearance_into_json() -> None:
    save_settings(Settings(appearance="Light"))
    data = json.loads(settings_path().read_text(encoding="utf-8"))
    assert data["appearance"] == "Light"


def test_invalid_appearance_falls_back_to_system() -> None:
    path = settings_path()
    _ = path.write_text(json.dumps({"appearance": "Chartreuse"}), encoding="utf-8")
    assert load_settings().appearance == "System"


def test_missing_appearance_key_defaults_to_system() -> None:
    # Backward compatibility: an old settings.json predates the field.
    path = settings_path()
    _ = path.write_text(json.dumps({"n_threads": 4, "use_gpu": False}), encoding="utf-8")
    loaded = load_settings()
    assert loaded.appearance == "System"
    assert loaded.n_threads == 4
