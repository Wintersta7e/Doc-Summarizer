# SPDX-License-Identifier: MIT
"""Qt/QML application entry point.

Boots a ``QQmlApplicationEngine``, wires the ``ConsoleBridge`` in as the ``bridge``
context object, and loads ``App/Main.qml``. Run with ``python -m docsummarizer.ui.app``
(a console entry point replaces ``run.py`` when the CustomTkinter GUI is retired).
``--smoke-test`` starts everything but the model, prints
``DocSummarizer <version> smoke ok`` and exits 0, or names what failed and exits 1.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
import threading
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, final

from PySide6.QtCore import (
    QEventLoop,
    QIODevice,
    QMessageLogContext,
    QObject,
    QTimer,
    QtMsgType,
    QUrl,
    qInstallMessageHandler,
)
from PySide6.QtGui import QGuiApplication
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtQml import QQmlApplicationEngine, QQmlNetworkAccessManagerFactory
from typing_extensions import override

from docsummarizer import __version__
from docsummarizer.logger import get_logger, log_error, log_info, log_startup
from docsummarizer.settings import load_settings, save_settings, settings_path
from docsummarizer.ui.bridge import ConsoleBridge
from docsummarizer.ui.fonts import register_fonts

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

_QML_DIR = Path(__file__).parent / "qml"
_MAIN_QML = _QML_DIR / "App" / "Main.qml"
_LOCAL_SCHEMES = frozenset({"file", "qrc"})
SMOKE_TEST_FLAG = "--smoke-test"
# How long the smoke run lets the window run before teardown: long enough for
# its first bindings, timers and animations to fire.
_SMOKE_SETTLE_MS = 250
_QT_PROBLEMS = (QtMsgType.QtWarningMsg, QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg)


@final
class _LocalOnlyNetworkAccessManager(QNetworkAccessManager):
    """Swaps a non-local request for an empty URL, which fails before any socket opens."""

    def __init__(self, guard: LocalOnlyNetworkAccessManagerFactory, parent: QObject) -> None:
        super().__init__(parent)
        self._guard = guard

    @override
    def createRequest(
        self,
        op: QNetworkAccessManager.Operation,
        request: QNetworkRequest,
        /,
        outgoingData: QIODevice | None = None,
    ) -> QNetworkReply:
        if request.url().scheme() not in _LOCAL_SCHEMES:
            self._guard.count_refusal()
            request = QNetworkRequest(QUrl())
        return super().createRequest(op, request, outgoingData)


@final
class LocalOnlyNetworkAccessManagerFactory(QQmlNetworkAccessManagerFactory):
    """Gives the QML engine managers that refuse every URL but ``file:`` and ``qrc:``.

    Model text can carry ``<img src="http://…">`` or ``![x](http://…)``, and a
    QML ``Text`` fetches either one. Every network request the engine makes
    passes through a manager this factory created. A ``Text`` opens a local-file
    image itself and never through a manager, so the summary screen escapes
    Markdown image syntax instead.
    """

    def __init__(self) -> None:
        """Start with no refusals counted."""
        super().__init__()
        self._lock = threading.Lock()  # managers run on the engine's loader threads
        self._refused = 0

    @property
    def refused(self) -> int:
        """Requests refused so far, across every manager this factory made."""
        with self._lock:
            return self._refused

    def count_refusal(self) -> None:
        """Count one refused request."""
        with self._lock:
            self._refused += 1

    @override
    def create(self, parent: QObject) -> QNetworkAccessManager:
        """Return a local-only manager owned by ``parent``."""
        return _LocalOnlyNetworkAccessManager(self, parent)


@final
class _LocalOnlyEngine(QQmlApplicationEngine):
    """A QML engine whose network access is local-only.

    Qt does not own a network-access-manager factory and PySide keeps no
    reference to one; a factory collected before its engine crashes the
    process on the engine's next request. The engine holds its own.
    """

    def __init__(self) -> None:
        super().__init__()
        self._network_guard = LocalOnlyNetworkAccessManagerFactory()
        self.setNetworkAccessManagerFactory(self._network_guard)


def create_engine(app: QGuiApplication) -> tuple[QQmlApplicationEngine, ConsoleBridge]:
    """Build the QML engine + bridge and load the root window.

    Returns the engine and bridge (kept alive by the caller). The engine refuses
    every non-local URL; ``engine.networkAccessManagerFactory()`` returns its
    ``LocalOnlyNetworkAccessManagerFactory``. Raises ``RuntimeError`` if the QML
    failed to instantiate.
    """
    engine = _LocalOnlyEngine()
    bridge = ConsoleBridge()
    _ = register_fonts()  # so QML resolves Cormorant/Chakra/Saira/Share Tech Mono
    engine.addImportPath(str(_QML_DIR))
    # Main.qml's `required property var bridge`: a typed hand-off qmllint can
    # follow, where a context property is an unqualified global.
    engine.setInitialProperties({"bridge": bridge})
    _ = app.aboutToQuit.connect(bridge.shutdown)
    engine.load(QUrl.fromLocalFile(str(_MAIN_QML)))
    if not engine.rootObjects():
        msg = f"Failed to load QML from {_MAIN_QML}"
        raise RuntimeError(msg)
    return engine, bridge


def _install_excepthook(bridge: ConsoleBridge) -> None:
    """Route uncaught exceptions to the log + a best-effort toast.

    A windowed (no-console) build discards stderr, so without this an uncaught
    exception in a Qt slot vanishes silently. Logging it makes the next such
    failure diagnosable from the log file.
    """

    def hook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: TracebackType | None,
    ) -> None:
        log_error(
            "Uncaught exception:\n"
            + "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        )
        with contextlib.suppress(Exception):  # best-effort during a crash
            bridge.toast.emit("Something went wrong — see the log file")
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = hook


class SmokeTestError(RuntimeError):
    """A part of the application the smoke run checks did not start."""


def _spin(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    _ = loop.exec()


def _import_model_runtime() -> bool:
    """Import ``llama_cpp`` when the runtime extra is installed, without loading a model.

    Returns:
        Whether the runtime is installed. A runtime that is installed but fails
        to import (a missing shared library) raises.
    """
    try:
        _ = importlib.import_module("llama_cpp")
    except ModuleNotFoundError as exc:
        if exc.name != "llama_cpp":
            raise
        return False
    return True


def _smoke_steps(app: QGuiApplication) -> None:
    """Start every part of the application but the model; raise on the first that fails.

    The settings file is read and written back with the values read.
    """
    if not register_fonts():
        msg = "no bundled fonts registered"
        raise SmokeTestError(msg)
    # Main.qml instantiates every screen, so a screen that fails to load fails here.
    engine, bridge = create_engine(app)
    try:
        settings = load_settings()
        save_settings(settings)
        if not settings_path().is_file() or load_settings() != settings:
            msg = f"settings did not round-trip through {settings_path()}"
            raise SmokeTestError(msg)
        runtime = _import_model_runtime()
        log_info(f"Smoke test: model runtime {'imported' if runtime else 'not installed'}")
        _spin(_SMOKE_SETTLE_MS)
    finally:
        # The window's bindings read the bridge: delete the engine first, also
        # when a step failed and its traceback still holds this frame.
        del engine
        _spin(0)
        bridge.shutdown()


def smoke_test(app: QGuiApplication) -> int:
    """Run the smoke steps; a failed step or any Qt warning on the way is a failure.

    Returns:
        0 after printing ``DocSummarizer <version> smoke ok``; 1 after writing
        what failed to stderr.
    """
    problems: list[str] = []

    def collect(mode: QtMsgType, context: QMessageLogContext, message: str) -> None:
        if mode in _QT_PROBLEMS:
            problems.append(f"{context.category}: {message}")

    previous = qInstallMessageHandler(collect)
    try:
        _smoke_steps(app)
    except Exception:  # every failure is the result: report it, exit 1
        logger = get_logger()
        logger.exception("Smoke test failed")
        traceback.print_exc()
        return 1
    finally:
        _ = qInstallMessageHandler(previous)
    if problems:
        log_error(f"Smoke test: {len(problems)} Qt warnings")
        _ = sys.stderr.write("".join(f"{problem}\n" for problem in problems))
        return 1
    log_info("Smoke test passed")
    _ = sys.stdout.write(f"DocSummarizer {__version__} smoke ok\n")
    return 0


def _application(args: list[str]) -> QGuiApplication:
    """The process's Qt application: the one already running, or a new one."""
    existing = QGuiApplication.instance()
    if isinstance(existing, QGuiApplication):
        return existing
    app = QGuiApplication(args)
    app.setApplicationName("DocSummarizer")
    app.setOrganizationName("DocSummarizer")
    return app


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the DocSummarizer console UI, or with ``--smoke-test`` check that it starts.

    Args:
        argv: The command line, program name first; ``sys.argv`` when omitted.

    Returns:
        The process exit status.
    """
    args = list(sys.argv if argv is None else argv)
    smoke = SMOKE_TEST_FLAG in args
    log_startup()
    app = _application([arg for arg in args if arg != SMOKE_TEST_FLAG])
    if smoke:
        return smoke_test(app)
    # Keep the engine referenced for the app's lifetime (GC would tear down the
    # window). bridge is owned by the engine's context but used here too.
    _engine, bridge = create_engine(app)
    _install_excepthook(bridge)
    bridge.checkModel()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
