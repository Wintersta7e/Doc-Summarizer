"""Qt/QML application entry point.

Boots a ``QQmlApplicationEngine``, wires the ``ConsoleBridge`` in as the ``bridge``
context object, and loads ``App/Main.qml``. Run with ``python -m docsummarizer.ui.app``
(a console entry point replaces ``run.py`` when the CustomTkinter GUI is retired).
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from PySide6.QtCore import QIODevice, QObject, QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtQml import QQmlApplicationEngine, QQmlNetworkAccessManagerFactory

from docsummarizer.logger import log_error, log_startup
from docsummarizer.ui.bridge import ConsoleBridge

_QML_DIR = Path(__file__).parent / "qml"
_MAIN_QML = _QML_DIR / "App" / "Main.qml"
_LOCAL_SCHEMES = frozenset({"file", "qrc"})


class _LocalOnlyNetworkAccessManager(QNetworkAccessManager):
    """Swaps a non-local request for an empty URL, which fails before any socket opens."""

    def __init__(self, guard: LocalOnlyNetworkAccessManagerFactory, parent: QObject) -> None:
        super().__init__(parent)
        self._guard = guard

    def createRequest(
        self,
        op: QNetworkAccessManager.Operation,
        request: QNetworkRequest,
        outgoing_data: QIODevice | None = None,
    ) -> QNetworkReply:
        if request.url().scheme() not in _LOCAL_SCHEMES:
            self._guard.count_refusal()
            request = QNetworkRequest(QUrl())
        return super().createRequest(op, request, outgoing_data)


class LocalOnlyNetworkAccessManagerFactory(QQmlNetworkAccessManagerFactory):
    """Gives the QML engine managers that refuse every URL but ``file:`` and ``qrc:``.

    Model text can carry ``<img src="http://…">`` or ``![x](http://…)``, and a
    QML ``Text`` fetches either one. Every network request the engine makes
    passes through a manager this factory created.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()  # managers run on the engine's loader threads
        self._refused = 0

    @property
    def refused(self) -> int:
        """Requests refused so far, across every manager this factory made."""
        with self._lock:
            return self._refused

    def count_refusal(self) -> None:
        with self._lock:
            self._refused += 1

    def create(self, parent: QObject) -> QNetworkAccessManager:
        return _LocalOnlyNetworkAccessManager(self, parent)


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
    from docsummarizer.ui.fonts import register_fonts

    engine = _LocalOnlyEngine()
    bridge = ConsoleBridge()
    register_fonts()  # so QML resolves Cormorant/Chakra/Saira/Share Tech Mono
    engine.addImportPath(str(_QML_DIR))
    engine.rootContext().setContextProperty("bridge", bridge)
    app.aboutToQuit.connect(bridge.shutdown)
    engine.load(QUrl.fromLocalFile(str(_MAIN_QML)))
    if not engine.rootObjects():
        raise RuntimeError(f"Failed to load QML from {_MAIN_QML}")
    return engine, bridge


def _install_excepthook(bridge: ConsoleBridge) -> None:
    """Route uncaught exceptions to the log + a best-effort toast.

    A windowed (no-console) build discards stderr, so without this an uncaught
    exception in a Qt slot vanishes silently. Logging it makes the next such
    failure diagnosable from the log file.
    """
    import contextlib
    import traceback
    from types import TracebackType

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


def main() -> int:
    """Launch the DocSummarizer console UI."""
    log_startup()
    app = QGuiApplication(sys.argv)
    app.setApplicationName("DocSummarizer")
    app.setOrganizationName("DocSummarizer")
    # Keep the engine referenced for the app's lifetime (GC would tear down the
    # window). bridge is owned by the engine's context but used here too.
    _engine, bridge = create_engine(app)
    _install_excepthook(bridge)
    bridge.checkModel()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
