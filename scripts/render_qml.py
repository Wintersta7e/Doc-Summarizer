#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Render a DocSummarizer QML screen to a PNG, offscreen — visual QA without a display.

Lets us *see* the Qt/QML UI (and compare it to the design screenshots) without a
real display, so the redesign can be iterated visually instead of blind.

Usage:
    python scripts/render_qml.py [screen] [theme] [out.png]
      screen : summary | extract | batch | config | firstrun   (default: summary)
      theme  : dark | light                                     (default: dark)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import final

_ = os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ = os.environ.setdefault("QT_QUICK_BACKEND", "software")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from PySide6.QtCore import QEventLoop, QTimer, QUrl
from PySide6.QtGui import QGuiApplication, QImage
from PySide6.QtQuick import QQuickView

from docsummarizer.model_manager import StructuredSummary, SummaryPoint
from docsummarizer.provenance import SourceSpan
from docsummarizer.ui.bridge import ConsoleBridge, summary_to_variant
from docsummarizer.ui.fonts import register_fonts

_QML_DIR = REPO / "src" / "docsummarizer" / "ui" / "qml"
_PREVIEW = REPO / "scripts" / "_preview.qml"
# Every screen _preview.qml shows, and every theme.
SCREENS = ("summary", "extract", "batch", "config", "firstrun")
THEMES = ("dark", "light")
# How long a screen runs before it is grabbed: its fades have finished.
SETTLE_MS = 600

SAMPLE_SOURCE = (
    "Most small farm wells today rely on piston pumps that need a skilled mechanic, "
    "spare seals and a steady supply of grease to keep them running. We describe a "
    "new simple pump design, the spiral-vane pump, based solely on a turning vane, "
    "dispensing with pistons and valves entirely. Trials at two village wells show "
    "these pumps to be superior in output while being easier to repair and requiring "
    "significantly less time to assemble."
)


def _sample_summary() -> StructuredSummary:
    src = SAMPLE_SOURCE
    q1 = (
        "We describe a new simple pump design, the spiral-vane pump, based solely on a "
        "turning vane, dispensing with pistons and valves entirely."
    )
    q2 = (
        "Trials at two village wells show these pumps to be superior in output while being "
        "easier to repair and requiring significantly less time to assemble."
    )
    s1, s2 = src.find(q1), src.find(q2)
    return StructuredSummary(
        "detailed",
        "The authors introduce the spiral-vane pump — a vane-only design that removes "
        "pistons and valves, assembles faster, and sets a new output record.",
        [
            SummaryPoint("A piston-free pump design", SourceSpan(s1, s1 + len(q1), q1, 1.0)),
            SummaryPoint("Faster, simpler repair", SourceSpan(s2, s2 + len(q2), q2, 1.0)),
            SummaryPoint("Record output at village wells", None),
        ],
        None,
        "rendered",
    )


class _FakeSummarizer:
    def summarize_structured(
        self, _text: str, _summary_type: str, /, **_options: object
    ) -> StructuredSummary:
        return _sample_summary()

    def summarize(self, _text: str, _summary_type: str, /, **_options: object) -> str:
        return _sample_summary().lead or ""

    def close(self) -> None:
        pass


def _fake_factory(_model: Path, _threads: int | None, _gpu_layers: int) -> _FakeSummarizer:
    return _FakeSummarizer()


@final
class _PreviewBridge(ConsoleBridge):
    """A bridge seeded with a ready model, a document and batch rows, for a still image."""

    def seed_document(self, name: str, text: str) -> None:
        self._summarizer = _FakeSummarizer()  # mark model-ready
        self._current_file = name
        self._extracted_text = text

    def seed_batch_rows(self, rows: list[dict[str, object]]) -> None:
        self._batch_rows = rows
        self.batchRowsChanged.emit()


class _Args(argparse.Namespace):
    """The parsed command line."""

    screen: str = "summary"
    theme: str = "dark"
    out: str | None = None


def _parse_args() -> _Args:
    parser = argparse.ArgumentParser(description="Render a QML screen to a PNG, offscreen.")
    _ = parser.add_argument("screen", nargs="?", default=_Args.screen, choices=SCREENS)
    _ = parser.add_argument("theme", nargs="?", default=_Args.theme, choices=THEMES)
    _ = parser.add_argument(
        "out", nargs="?", help="default: render-<screen>-<theme>.png in the repo"
    )
    return parser.parse_args(namespace=_Args())


def load_preview(screen: str, theme: str) -> QQuickView:
    """Show screen in theme, seeded with sample state; the caller keeps the returned view.

    Raises ``RuntimeError`` when the QML does not load or the preview shell lacks
    a property this function sets.
    """
    _ = register_fonts()
    view = QQuickView()
    # The view owns the bridge, so the bridge outlives every binding that reads it;
    # a bridge freed first turns each of them into a "Cannot read property of null".
    bridge = _PreviewBridge(summarizer_factory=_fake_factory, synchronous=True, parent=view)
    bridge.seed_document("spiral-vane-pump.pdf", SAMPLE_SOURCE)
    view.engine().addImportPath(str(_QML_DIR))
    view.setInitialProperties({"bridge": bridge})
    view.setResizeMode(QQuickView.ResizeMode.SizeRootObjectToView)
    view.resize(1240, 820)
    view.setSource(QUrl.fromLocalFile(str(_PREVIEW)))
    if view.status() != QQuickView.Status.Ready:
        raise RuntimeError("\n".join(err.toString() for err in view.errors()))

    root = view.rootObject()
    seeded = (
        root.setProperty("darkTheme", theme == "dark"),
        root.setProperty("screen", screen),
        root.setProperty("firstRun", screen == "firstrun"),
    )
    if not all(seeded):
        msg = f"{_PREVIEW.name} lacks a darkTheme, screen or firstRun property"
        raise RuntimeError(msg)
    view.show()

    bridge.modelReadyChanged.emit()
    bridge.docChanged.emit()
    if screen == "summary":
        bridge.summaryReady.emit(summary_to_variant(_sample_summary()))
    if screen == "batch":
        bridge.seed_batch_rows(
            [
                {"name": "pump-trial-north.pdf", "status": "DONE", "tokens": 312},
                {"name": "pump-trial-south.pdf", "status": "PROCESSING", "tokens": 0},
                {"name": "pump-trial-east.pdf", "status": "QUEUED", "tokens": 0},
            ]
        )
    return view


def render(screen: str, theme: str) -> QImage:
    """Show screen in theme, let it settle, and grab it.

    Raises ``RuntimeError`` as ``load_preview`` does.
    """
    view = load_preview(screen, theme)
    loop = QEventLoop()
    QTimer.singleShot(SETTLE_MS, loop.quit)
    _ = loop.exec()
    image = view.grabWindow()
    _ = view.close()
    return image


def main() -> int:
    """Render the screen and theme the command line names; 0 when the PNG is saved."""
    args = _parse_args()
    out = args.out or str(REPO / f"render-{args.screen}-{args.theme}.png")

    _app = QGuiApplication(sys.argv)  # kept alive for the view
    try:
        image = render(args.screen, args.theme)
    except RuntimeError as exc:
        print(f"FAILED: {exc}")
        return 1

    if image.isNull() or image.width() == 0:
        print("FAILED: grabbed image is null/empty")
        return 1
    if not image.save(out):
        print(f"FAILED: could not write {out}")
        return 1
    print(f"saved {out}  ({image.width()}x{image.height()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
