"""Persisted user settings (CPU threads, GPU offload, appearance, language).

A tiny JSON file under the app-data ``config`` dir. Everything is
optional and tolerant: a missing or corrupt file yields defaults rather
than raising, so the app always starts even if the file was hand-edited
into nonsense. Settings act as *defaults* — the CLI's flags override
them at call time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .logger import log_debug, log_warning
from .model_manager import LANGUAGE_AUTO, normalize_language
from .paths import app_data_dir

# llama.cpp uses -1 to mean "offload every layer to the GPU".
_ALL_GPU_LAYERS = -1

# Appearance modes accepted by the GUI (mirrors CustomTkinter / the Config
# screen's segmented control). Anything else falls back to "System".
_VALID_APPEARANCES = ("System", "Light", "Dark")
_DEFAULT_APPEARANCE = "System"


def settings_path() -> Path:
    """Path to the settings JSON file (its parent ``config`` dir is created)."""
    return app_data_dir("config") / "settings.json"


@dataclass
class Settings:
    """User-tunable runtime settings.

    Attributes:
        n_threads: CPU threads for inference. ``None`` means auto (half of
            the available cores, decided in ``Summarizer``).
        use_gpu: Offload all model layers to the GPU when ``True``. Harmless
            on CPU-only llama-cpp builds — the flag is simply ignored there.
        appearance: GUI theme, one of ``System``/``Light``/``Dark``. Persisted
            so the chosen theme survives a restart.
        output_language: Language the summary is written in. ``"auto"`` (the
            default) matches the document; any other name pins it.
    """

    n_threads: int | None = None
    use_gpu: bool = False
    appearance: str = _DEFAULT_APPEARANCE
    output_language: str = LANGUAGE_AUTO

    @property
    def n_gpu_layers(self) -> int:
        """The llama.cpp ``n_gpu_layers`` value implied by ``use_gpu``."""
        return _ALL_GPU_LAYERS if self.use_gpu else 0


def load_settings() -> Settings:
    """Load settings, falling back to defaults on any problem."""
    path = settings_path()
    if not path.exists():
        return Settings()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log_warning(f"Could not read settings file ({exc!s}); using defaults")
        return Settings()

    if not isinstance(data, dict):
        log_warning("Settings file is not a JSON object; using defaults")
        return Settings()

    threads = data.get("n_threads")
    # Tolerate hand-edited garbage: only a positive int is a valid thread
    # count (bool is an int subclass, so exclude it explicitly).
    if not (isinstance(threads, int) and not isinstance(threads, bool) and threads >= 1):
        threads = None

    appearance = data.get("appearance")
    if appearance not in _VALID_APPEARANCES:
        appearance = _DEFAULT_APPEARANCE

    # Deliberately not restricted to the GUI's picker list: a hand-edited
    # language the picker doesn't offer is still honoured, and anything
    # unusable (empty, wrong type, a pasted paragraph) normalizes to "auto".
    language = data.get("output_language")
    language = normalize_language(language) if isinstance(language, str) else LANGUAGE_AUTO

    return Settings(
        n_threads=threads,
        use_gpu=bool(data.get("use_gpu", False)),
        appearance=appearance,
        output_language=language,
    )


def save_settings(settings: Settings) -> None:
    """Persist settings atomically. I/O failures are logged, not raised."""
    path = settings_path()
    payload = {
        "n_threads": settings.n_threads,
        "use_gpu": settings.use_gpu,
        "appearance": settings.appearance,
        "output_language": settings.output_language,
    }
    try:
        # Write to a sibling temp file then atomically replace, so a crash
        # mid-write can't leave a half-written (corrupt) settings file.
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)
        log_debug(f"Settings saved to {path}")
    except OSError as exc:
        log_warning(f"Could not save settings file ({exc!s})")
