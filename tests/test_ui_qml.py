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


@pytest.fixture(scope="session")
def qapp() -> QGuiApplication:
    return QGuiApplication.instance() or QGuiApplication([])


@pytest.fixture
def engine(qapp: QGuiApplication) -> QQmlApplicationEngine:
    from docsummarizer.ui.bridge import ConsoleBridge

    eng = QQmlApplicationEngine()
    eng.addImportPath(str(_QML_DIR))
    bridge = ConsoleBridge()
    eng.rootContext().setContextProperty("bridge", bridge)
    # Keep the bridge alive for the engine's lifetime.
    eng.setProperty("_bridge", bridge)
    return eng


@pytest.mark.parametrize("name", _COMPONENTS)
def test_component_compiles_and_instantiates(engine: QQmlApplicationEngine, name: str) -> None:
    component = QQmlComponent(engine, QUrl.fromLocalFile(str(_APP_DIR / f"{name}.qml")))
    assert component.status() == QQmlComponent.Status.Ready, component.errorString()
    obj = component.create(engine.rootContext())
    assert obj is not None, f"{name}.qml failed to instantiate: {component.errorString()}"
