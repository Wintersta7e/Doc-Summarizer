# SPDX-License-Identifier: MIT
"""Register the bundled UI fonts with Qt so QML can resolve them by family name.

The four design families (Cormorant Garamond, Chakra Petch, Share Tech Mono,
Saira) ship as ``.ttf`` files under ``qml/App/fonts/``. Calling
:func:`register_fonts` once at startup (and in the render harness) makes
``font.family: "Cormorant Garamond"`` resolve instead of falling back to a
system sans. Safe to call when the fonts are absent (returns an empty list).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFontDatabase

from docsummarizer.logger import log_debug

_FONTS_DIR = Path(__file__).parent / "qml" / "App" / "fonts"


def register_fonts() -> list[str]:
    """Load every bundled ``.ttf`` into Qt; return the registered family names."""
    families: list[str] = []
    if not _FONTS_DIR.is_dir():
        return families
    for ttf in sorted(_FONTS_DIR.glob("*.ttf")):
        font_id = QFontDatabase.addApplicationFont(str(ttf))
        if font_id != -1:
            families.extend(QFontDatabase.applicationFontFamilies(font_id))
    log_debug(f"Registered {len(families)} bundled font families")
    return families
