# SPDX-License-Identifier: MIT
"""Every screen, in every theme, loads and tears down without a QML warning.

Loads each screen the way ``scripts/render_qml.py`` does and collects every Qt
message through ``qInstallMessageHandler``: a binding that reads a property of
null, a failed assignment or a missing type each arrive as a warning naming the
QML file and line, which a status check or a still image never shows.
"""

from __future__ import annotations

import gc
import itertools
import logging

import pytest
from PySide6.QtCore import (
    QEventLoop,
    QMessageLogContext,
    QTimer,
    QtMsgType,
    qInstallMessageHandler,
)
from PySide6.QtGui import QGuiApplication

from render_qml import SCREENS, THEMES, load_preview

_LOG = logging.getLogger(__name__)
# How long each screen runs before teardown: long enough for its first bindings,
# timers and animations to fire.
_SETTLE_MS = 250
_REPORTED = (QtMsgType.QtWarningMsg, QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg)


@pytest.fixture(scope="session")
def qapp() -> QGuiApplication:
    app = QGuiApplication.instance() or QGuiApplication([])
    assert isinstance(app, QGuiApplication)
    return app


def _spin(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    _ = loop.exec()


@pytest.mark.usefixtures("qapp")
@pytest.mark.parametrize(("screen", "theme"), list(itertools.product(SCREENS, THEMES)))
def test_every_screen_loads_without_qml_warnings(screen: str, theme: str) -> None:
    messages: list[str] = []

    def collect(mode: QtMsgType, context: QMessageLogContext, message: str) -> None:
        if mode in _REPORTED:
            messages.append(f"{context.category}: {message}")

    previous = qInstallMessageHandler(collect)
    try:
        view = load_preview(screen, theme)
        _spin(_SETTLE_MS)
        _ = view.close()
        del view
        _ = gc.collect()
        _spin(0)
    finally:
        _ = qInstallMessageHandler(previous)

    _LOG.info("%s/%s: %s Qt warnings", screen, theme, len(messages))
    assert messages == []
