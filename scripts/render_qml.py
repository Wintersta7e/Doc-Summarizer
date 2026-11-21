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

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from PySide6.QtCore import QEventLoop, QTimer, QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQuick import QQuickView

from docsummarizer.model_manager import StructuredSummary, SummaryPoint
from docsummarizer.provenance import SourceSpan
from docsummarizer.ui.bridge import ConsoleBridge, summary_to_variant
from docsummarizer.ui.fonts import register_fonts

_QML_DIR = REPO / "src" / "docsummarizer" / "ui" / "qml"
_PREVIEW = REPO / "scripts" / "_preview.qml"

SAMPLE_SOURCE = (
    "The dominant sequence transduction models are based on complex recurrent or "
    "convolutional neural networks that include an encoder and a decoder. We propose a "
    "new simple network architecture, the Transformer, based solely on attention "
    "mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two "
    "machine translation tasks show these models to be superior in quality while being "
    "more parallelizable and requiring significantly less time to train."
)


def _sample_summary() -> StructuredSummary:
    src = SAMPLE_SOURCE
    q1 = (
        "We propose a new simple network architecture, the Transformer, based solely on "
        "attention mechanisms, dispensing with recurrence and convolutions entirely."
    )
    q2 = (
        "Experiments on two machine translation tasks show these models to be superior in "
        "quality while being more parallelizable and requiring significantly less time to train."
    )
    s1, s2 = src.find(q1), src.find(q2)
    return StructuredSummary(
        "detailed",
        "Vaswani et al. introduce the Transformer — an attention-only architecture that "
        "removes recurrence and convolution, trains faster, and sets a new benchmark.",
        [
            SummaryPoint("A pure-attention architecture", SourceSpan(s1, s1 + len(q1), q1, 1.0)),
            SummaryPoint("Faster, parallelizable training", SourceSpan(s2, s2 + len(q2), q2, 1.0)),
            SummaryPoint("State-of-the-art translation quality", None),
        ],
        None,
        "rendered",
    )


class _FakeSummarizer:
    def summarize_structured(self, text: str, summary_type: str = "detailed") -> StructuredSummary:
        return _sample_summary()

    def summarize(self, text: str, summary_type: str = "detailed") -> str:
        return _sample_summary().lead or ""

    def close(self) -> None:
        pass


def main() -> int:
    screen = sys.argv[1] if len(sys.argv) > 1 else "summary"
    theme = sys.argv[2] if len(sys.argv) > 2 else "dark"
    out = sys.argv[3] if len(sys.argv) > 3 else str(REPO / f"render-{screen}-{theme}.png")

    _app = QGuiApplication(sys.argv)  # kept alive for the view

    register_fonts()
    bridge = ConsoleBridge(summarizer_factory=lambda *_: _FakeSummarizer(), synchronous=True)
    bridge._summarizer = _FakeSummarizer()  # mark model-ready
    bridge._current_file = "attention.pdf"
    bridge._extracted_text = SAMPLE_SOURCE

    view = QQuickView()
    view.engine().addImportPath(str(_QML_DIR))
    view.rootContext().setContextProperty("bridge", bridge)
    view.setResizeMode(QQuickView.ResizeMode.SizeRootObjectToView)
    view.resize(1240, 820)
    view.setSource(QUrl.fromLocalFile(str(_PREVIEW)))
    if view.status() != QQuickView.Status.Ready:
        for err in view.errors():
            print(err.toString())
        return 1

    root = view.rootObject()
    root.setProperty("darkTheme", theme == "dark")
    root.setProperty("screen", screen)
    root.setProperty("firstRun", screen == "firstrun")
    view.show()

    bridge.modelReadyChanged.emit()
    bridge.docChanged.emit()
    if screen == "summary":
        bridge.summaryReady.emit(summary_to_variant(_sample_summary()))
    if screen == "batch":
        bridge._batch_rows = [
            {"name": "1706.03762.pdf", "status": "DONE", "tokens": 312},
            {"name": "1810.04805.pdf", "status": "PROCESSING", "tokens": 0},
            {"name": "2005.14165.pdf", "status": "QUEUED", "tokens": 0},
        ]
        bridge.batchRowsChanged.emit()

    loop = QEventLoop()
    QTimer.singleShot(600, loop.quit)
    loop.exec()
    image = view.grabWindow()
    if image.isNull() or image.width() == 0:
        print("FAILED: grabbed image is null/empty")
        return 1
    image.save(out)
    print(f"saved {out}  ({image.width()}x{image.height()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
