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
from PySide6.QtCore import (
    QEventLoop,
    QMessageLogContext,
    QObject,
    QTimer,
    QtMsgType,
    QUrl,
    qInstallMessageHandler,
)
from PySide6.QtGui import QGuiApplication, QTextDocument
from PySide6.QtNetwork import QNetworkReply, QNetworkRequest
from PySide6.QtQml import QQmlApplicationEngine, QQmlComponent, QQmlEngine, QQmlExpression
from PySide6.QtQuick import QQuickItem, QQuickWindow

from docsummarizer.model_manager import StructuredSummary, SummaryPoint
from docsummarizer.ui.app import create_engine
from docsummarizer.ui.bridge import ConsoleBridge, summary_to_variant

_APP_QML = Path(__file__).resolve().parents[1] / "src" / "docsummarizer" / "ui" / "qml" / "App"

# QML's Text.textFormat values (AutoText is 2).
_PLAIN, _RICH, _MARKDOWN = 0, 1, 3

# Where each payload is shown, and the format its renderer must use.
_EXPECTED_FORMAT = {
    "LEADMARK": _MARKDOWN,
    "POINTMARK": _MARKDOWN,
    "SECTIONMARK": _MARKDOWN,
    "PROSEMARK": _MARKDOWN,
    "FILEMARK": _PLAIN,
    "ROWMARK": _PLAIN,
    "OUTDIRMARK": _PLAIN,
    "TOASTMARK": _PLAIN,
    "DOCMARK": _RICH,  # escaped before it is rendered
}


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

    The text renderers no longer issue any request, so an ``Image`` with a remote
    source, created in the same engine, keeps the guard under test.

    Mutation caught: ``create_engine`` building a plain ``QQmlApplicationEngine``
    (no local-only factory). The ``Image`` then GETs its source from the loopback
    server, as would any ``Text`` left on ``AutoText`` (its ``<img>``) or any
    ``MarkdownText`` whose image syntax is not escaped (its ``![x](...)``).
    """
    engine, bridge, window = console
    port, hits = loopback
    component = QQmlComponent(engine)
    component.setData(
        f'import QtQuick\nImage {{ source: "http://127.0.0.1:{port}/IMAGEMARK.png" }}'.encode(),
        QUrl(),
    )
    image = component.create()
    assert isinstance(image, QQuickItem), component.errorString()
    image.setParentItem(window.contentItem())
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


def _text_format(item: QQuickItem) -> int:
    # PySide has no converter for QQuickText::TextFormat; QML evaluates it to a number.
    expr = QQmlExpression(QQmlEngine.contextForObject(item), item, "textFormat")
    value, undefined = expr.evaluate()
    assert not undefined
    return int(value)


def test_content_text_formats_are_explicit(
    console: tuple[QQmlApplicationEngine, ConsoleBridge, QQuickWindow],
    loopback: tuple[int, list[str]],
) -> None:
    """Model output renders as Markdown, file names as plain text and document text as
    escaped rich text; none of their renderers is left on ``AutoText``.

    Mutation caught: deleting ``textFormat`` from the lead, a point, a section point,
    a file-name line, a Batch row, the Batch output line or the toast - that
    ``Text`` reads ``AutoText``.
    """
    _engine, bridge, window = console
    port, _hits = loopback
    seen = dict.fromkeys(_EXPECTED_FORMAT, 0)
    wrong: list[str] = []

    def check() -> None:
        for item in _items(window):
            name = item.metaObject().className()
            if not name.startswith("QQuickText") or name.startswith("QQuickTextInput"):
                continue
            text = str(item.property("text") or "")
            for tag, expected in _EXPECTED_FORMAT.items():
                if tag not in text:
                    continue
                seen[tag] += 1
                actual = _text_format(item)
                if actual != expected or (expected == _RICH and "<img" in text):
                    wrong.append(f"{tag} in {name}: textFormat {actual}, expected {expected}")

    _render_everything(bridge, window, port, check)
    print(f"renderers checked per payload: {seen}")
    assert all(seen.values()), f"a payload reached no renderer: {seen}"
    assert wrong == []


def test_no_qml_opens_a_link() -> None:
    """No QML handler reacts to a link, so a Markdown link in model text stays inert.

    Mutation caught: ``onLinkActivated: link => Qt.openUrlExternally(link)`` on
    any ``Text``.
    """
    files = sorted(_APP_QML.glob("*.qml"))
    offenders = [
        f"{path.name}:{n}"
        for path in files
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "LinkActivated" in line
    ]
    print(f"qml files scanned={len(files)} offenders={offenders}")
    assert files, "scanned no QML file"
    assert offenders == []


def _local_image_payload(tag: str) -> str:
    """Text carrying an inline and a reference-style image at a file URL with a host."""
    base = f"file://example.invalid/share/{tag}"
    ref = tag.lower()
    return f"{tag} ![a]({base}-inline.png) ![b][{ref}]\n\n[{ref}]: {base}-ref.png"


def test_summary_text_loads_no_local_file_image(
    console: tuple[QQmlApplicationEngine, ConsoleBridge, QQuickWindow],
) -> None:
    """Model output never makes Qt open a ``file://<host>/`` image, inline or by reference.

    Qt opens a Markdown image's file URL itself, outside the engine's network
    access, and a host in that URL is a network share on Windows. The summary
    screen escapes image syntax, so the characters show and nothing is opened.

    Mutation caught: rendering the lead, a Detailed point or a Structured section
    point without ``escapeImages`` - Qt then tries to open that payload's image
    (the log names its tag) and the ``![`` no longer shows.
    """
    _engine, bridge, window = console
    tags = ("LEADMARK", "POINTMARK", "SECTIONMARK")
    messages: list[str] = []

    def handler(_mode: QtMsgType, _context: QMessageLogContext, message: str) -> None:
        messages.append(message)

    shown: dict[str, list[str]] = {tag: [] for tag in tags}
    bridge._current_file = "local.txt"
    bridge._extracted_text = "document text"
    bridge.docChanged.emit()
    window.setProperty("activeScreen", "summary")
    previous = qInstallMessageHandler(handler)
    try:
        for summary in (
            StructuredSummary(
                "detailed",
                _local_image_payload("LEADMARK"),
                [SummaryPoint(_local_image_payload("POINTMARK"))],
                None,
                "t",
            ),
            StructuredSummary(
                "structured",
                None,
                [],
                {"PURPOSE": [SummaryPoint(_local_image_payload("SECTIONMARK"))]},
                "t",
            ),
        ):
            bridge.summaryReady.emit(summary_to_variant(summary))
            window.grabWindow()
            _spin(500)
            for item in _items(window):
                if not item.metaObject().className().startswith("QQuickText"):
                    continue
                text = str(item.property("text") or "")
                for tag in tags:
                    if tag in text and _text_format(item) == _MARKDOWN:
                        document = QTextDocument()
                        document.setMarkdown(text)
                        shown[tag].append(document.toPlainText())
    finally:
        qInstallMessageHandler(previous)
    opened = sorted({m for m in messages if "example.invalid" in m})
    print(f"opened={opened} shown={shown}")
    assert opened == []
    assert all(shown.values()), f"a payload reached no Markdown renderer: {shown}"
    hidden = [tag for tag, texts in shown.items() for t in texts if t.count("![") != 2]
    assert hidden == [], f"image syntax rendered as an image for {hidden}"
