"""Tests for ``docsummarizer.settings`` — the persisted settings file.

These exercise load/save round-tripping and the tolerant fallback
behavior: a missing or corrupt file must never raise, so the app always
starts. The autouse ``_isolated_app_dirs`` fixture in conftest redirects
the app-data dir to a per-test tmp dir, so every ``settings_path()``
lands under ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path

from docsummarizer.settings import (
    Settings,
    load_settings,
    save_settings,
    settings_path,
)


def test_defaults_when_no_file() -> None:
    s = load_settings()
    assert s == Settings()
    assert s.n_threads is None
    assert s.use_gpu is False
    assert s.n_gpu_layers == 0


def test_settings_path_under_isolated_dir(tmp_path: Path) -> None:
    p = settings_path()
    assert str(p).startswith(str(tmp_path))
    assert p.name == "settings.json"
    assert p.parent.is_dir()  # config dir auto-created


def test_n_gpu_layers_maps_from_use_gpu() -> None:
    assert Settings(use_gpu=True).n_gpu_layers == -1  # offload all layers
    assert Settings(use_gpu=False).n_gpu_layers == 0


def test_save_then_load_roundtrip() -> None:
    save_settings(Settings(n_threads=6, use_gpu=True))
    s = load_settings()
    assert s.n_threads == 6
    assert s.use_gpu is True


def test_save_writes_expected_json() -> None:
    save_settings(Settings(n_threads=4, use_gpu=False))
    data = json.loads(settings_path().read_text(encoding="utf-8"))
    assert data == {
        "n_threads": 4,
        "use_gpu": False,
        "appearance": "System",
        "output_language": "auto",
    }


def test_corrupt_file_falls_back_to_defaults() -> None:
    settings_path().write_text("{not valid json", encoding="utf-8")
    assert load_settings() == Settings()  # must not raise


def test_non_object_json_falls_back() -> None:
    settings_path().write_text("[1, 2, 3]", encoding="utf-8")
    assert load_settings() == Settings()


def test_partial_keys_use_defaults() -> None:
    settings_path().write_text(json.dumps({"use_gpu": True}), encoding="utf-8")
    s = load_settings()
    assert s.use_gpu is True
    assert s.n_threads is None


def test_garbage_threads_coerced_to_none() -> None:
    settings_path().write_text(json.dumps({"n_threads": "lots"}), encoding="utf-8")
    assert load_settings().n_threads is None


def test_unknown_keys_ignored() -> None:
    settings_path().write_text(json.dumps({"surprise": 1, "use_gpu": True}), encoding="utf-8")
    assert load_settings().use_gpu is True


def test_output_language_round_trips() -> None:
    save_settings(Settings(output_language="Chinese"))
    assert load_settings().output_language == "Chinese"


def test_output_language_defaults_to_auto() -> None:
    assert load_settings().output_language == "auto"
    settings_path().write_text(json.dumps({"use_gpu": True}), encoding="utf-8")
    assert load_settings().output_language == "auto"


def test_unusable_output_language_falls_back_to_auto() -> None:
    """A hand-edited file must never put junk into the prompt."""
    for junk in ("", "   ", 42, None, ["English"]):
        settings_path().write_text(json.dumps({"output_language": junk}), encoding="utf-8")
        assert load_settings().output_language == "auto"


def test_unlisted_output_language_is_kept() -> None:
    """The GUI's picker list is not a whitelist — a hand-edited name survives."""
    settings_path().write_text(json.dumps({"output_language": "Swahili"}), encoding="utf-8")
    assert load_settings().output_language == "Swahili"
