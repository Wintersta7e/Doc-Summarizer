# SPDX-License-Identifier: MIT
"""Headless QML smoke test: every component must compile AND instantiate.

Runs under the offscreen platform (set in conftest). ``status == Ready`` means
the QML parsed and all types resolved; ``component.create()`` returning non-null
means it instantiated (catches binding/type errors a bare parse misses). Nothing
visual is asserted — that needs a display.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine, QQmlComponent

from docsummarizer.ui.bridge import ConsoleBridge

_QML_DIR = Path(__file__).resolve().parents[1] / "src" / "docsummarizer" / "ui" / "qml"
_APP_DIR = _QML_DIR / "App"

# Every QML file in the App module.
_COMPONENTS = [
    "Main",
    "TopBar",
    "Rail",
    "SegmentedControl",
    "Hex",
    "ConsoleButton",
    "SectionLabel",
    "StatusChip",
    "CutPanel",
    "GlowEffect",
    "Skeleton",
    "SummaryScreen",
    "ExtractScreen",
    "BatchScreen",
    "ConfigScreen",
    "FirstRunOverlay",
]
# The components that declare `required property var bridge`.
_TAKES_BRIDGE = frozenset(
    {
        "Main",
        "TopBar",
        "SummaryScreen",
        "ExtractScreen",
        "BatchScreen",
        "ConfigScreen",
        "FirstRunOverlay",
    }
)


@pytest.fixture(scope="session")
def qapp() -> QGuiApplication:
    app = QGuiApplication.instance() or QGuiApplication([])
    assert isinstance(app, QGuiApplication)
    return app


@pytest.fixture
def engine(qapp: QGuiApplication) -> QQmlApplicationEngine:
    # The engine needs the application object to exist first.
    assert qapp is QGuiApplication.instance()
    eng = QQmlApplicationEngine()
    eng.addImportPath(str(_QML_DIR))
    return eng


@pytest.mark.parametrize("name", _COMPONENTS)
def test_component_compiles_and_instantiates(engine: QQmlApplicationEngine, name: str) -> None:
    component = QQmlComponent(engine, QUrl.fromLocalFile(str(_APP_DIR / f"{name}.qml")))
    assert component.status() == QQmlComponent.Status.Ready, component.errorString()
    # The engine owns the bridge, so it outlives every binding that reads it.
    bridge = ConsoleBridge(parent=engine)
    initial = {"bridge": bridge} if name in _TAKES_BRIDGE else {}
    obj = component.createWithInitialProperties(initial, engine.rootContext())
    assert obj is not None, f"{name}.qml failed to instantiate: {component.errorString()}"
