"""Behavioural QML tests through the production engine (``create_engine`` + ``Main.qml``).

Runs offscreen (conftest). Each test drives the real bridge and the real screens;
nothing here instantiates a QML file on its own.
"""

from __future__ import annotations

import http.server
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import shiboken6
from PySide6.QtCore import QEventLoop, QObject, QTimer, QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtNetwork import QNetworkReply, QNetworkRequest
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuick import QQuickItem, QQuickWindow

from docsummarizer.model_manager import StructuredSummary, SummaryPoint
from docsummarizer.ui.app import create_engine
from docsummarizer.ui.bridge import ConsoleBridge, summary_to_variant


@pytest.fixture(scope="session")
def qapp() -> QGuiApplication:
    return QGuiApplication.instance() or QGuiApplication([])


@pytest.fixture
def loopback() -> Iterator[tuple[int, list[str]]]:
    """A loopback HTTP server that records the path of every GET it receives."""
    hits: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            hits.append(self.path)
            self.send_response(404)
            self.end_headers()

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1], hits
    server.shutdown()
    server.server_close()


@pytest.fixture
def console(
    qapp: QGuiApplication,
) -> Iterator[tuple[QQmlApplicationEngine, ConsoleBridge, QQuickWindow]]:
    engine, bridge = create_engine(qapp)
    window = engine.rootObjects()[0]
    assert isinstance(window, QQuickWindow)
    yield engine, bridge, window
    # Destroy the engine (and its window) while the bridge its bindings read is alive.
    shiboken6.delete(engine)


def _spin(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def _items(window: QQuickWindow) -> Iterator[QQuickItem]:
    stack = [window.contentItem()]
    while stack:
        item = stack.pop()
        stack.extend(item.childItems())
        yield item


def _payload(port: int, tag: str) -> str:
    """Text carrying a remote image as HTML and as Markdown, plus Markdown emphasis."""
    base = f"http://127.0.0.1:{port}/{tag}"
    return f'{tag} <img src="{base}-html.png"> ![x]({base}-md.png) **{tag}**'


def _render_everything(
    bridge: ConsoleBridge, window: QQuickWindow, port: int, after_each: Callable[[], None]
) -> None:
    """Show a payload in every renderer of model output, document text and file names
    (Summary, Extract and Batch screens, the toast) through today's production paths,
    calling ``after_each`` once each view has laid out and painted."""

    def settle() -> None:
        window.grabWindow()
        _spin(500)  # loader threads issue image requests after layout
        after_each()

    bridge._current_file = _payload(port, "FILEMARK")
    bridge._extracted_text = f"{_payload(port, 'DOCMARK')}\nsecond line"
    bridge.docChanged.emit()
    window.setProperty("activeScreen", "summary")
    for summary in (
        StructuredSummary(
            "detailed",
            _payload(port, "LEADMARK"),
            [SummaryPoint(_payload(port, "POINTMARK"))],
            None,
            "t",
        ),
        StructuredSummary(
            "structured", None, [], {"PURPOSE": [SummaryPoint(_payload(port, "SECTIONMARK"))]}, "t"
        ),
        StructuredSummary("brief", _payload(port, "PROSEMARK"), [], None, "t"),
    ):
        bridge.summaryReady.emit(summary_to_variant(summary))
        settle()
    window.setProperty("activeScreen", "extract")
    settle()
    bridge._batch_rows = [{"name": _payload(port, "ROWMARK"), "status": "QUEUED", "tokens": 0}]
    bridge.batchRowsChanged.emit()
    batch = next(i for i in _items(window) if "BatchScreen" in i.metaObject().className())
    batch.setProperty("outDir", _payload(port, "OUTDIRMARK"))
    window.setProperty("activeScreen", "batch")
    bridge.toast.emit(_payload(port, "TOASTMARK"))
    settle()


def test_rendered_text_fetches_nothing(
    console: tuple[QQmlApplicationEngine, ConsoleBridge, QQuickWindow],
    loopback: tuple[int, list[str]],
) -> None:
    """Rendering model output, document text and file names opens no socket.

    Mutation caught: ``create_engine`` building a plain ``QQmlApplicationEngine``
    (no local-only factory). Every ``Text`` left on ``AutoText`` then GETs the
    payload's ``<img>``, and every ``MarkdownText`` one its ``![x](...)``, from
    the loopback server.
    """
    engine, bridge, window = console
    port, hits = loopback
    _render_everything(bridge, window, port, lambda: None)
    print(f"requests={len(hits)} {sorted(hits)}")
    assert hits == [], f"rendering fetched {sorted(hits)}"
    guard = engine.networkAccessManagerFactory()
    print(f"refused={guard.refused}")
    assert guard.refused > 0, "no payload reached a renderer: the test proves nothing"


def test_local_only_manager_refuses_remote_schemes(
    qapp: QGuiApplication, loopback: tuple[int, list[str]], tmp_path: Path
) -> None:
    """Every scheme but ``file:`` and ``qrc:`` is refused before a socket opens.

    Mutation caught: a scheme check that lets ``http`` through (the loopback
    server records the GET) or one that refuses ``file:`` (the local read fails).
    """
    from docsummarizer.ui.app import LocalOnlyNetworkAccessManagerFactory

    port, hits = loopback
    local = tmp_path / "local.txt"
    local.write_text("local bytes", encoding="utf-8")
    factory = LocalOnlyNetworkAccessManagerFactory()
    parent = QObject()
    manager = factory.create(parent)

    def fetch(url: QUrl) -> QNetworkReply:
        reply = manager.get(QNetworkRequest(url))
        if not reply.isFinished():
            loop = QEventLoop()
            reply.finished.connect(loop.quit)
            QTimer.singleShot(5000, loop.quit)
            loop.exec()
        return reply

    remote = [fetch(QUrl(f"{scheme}://127.0.0.1:{port}/x.png")) for scheme in ("http", "https")]
    local_reply = fetch(QUrl.fromLocalFile(str(local)))
    print(f"requests={len(hits)} refused={factory.refused}")
    assert hits == []
    assert factory.refused == 2
    assert all(r.error() != QNetworkReply.NetworkError.NoError for r in remote)
    assert local_reply.error() == QNetworkReply.NetworkError.NoError
    assert bytes(local_reply.readAll().data()) == b"local bytes"
