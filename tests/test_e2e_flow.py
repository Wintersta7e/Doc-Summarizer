# SPDX-License-Identifier: MIT
"""End-to-end: open, summarize and export through the real QML app.

Each test builds the production engine (``create_engine`` + ``Main.qml``) and
drives it through the QML handlers a user triggers: the open dialog, the
Regenerate button, the summary-type control and the save dialog. Only the model
is fake: a stand-in ``llama_cpp`` module whose ``Llama`` answers each prompt
the way the real one is asked to, so the real ``Summarizer``, the bridge's
worker threads, the parsers and the exporters all run. Every wait is on a Qt
signal with a timeout; nothing sleeps.
"""

from __future__ import annotations

import functools
import json
import sys
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest
import shiboken6
from PySide6.QtCore import QEventLoop, QObject, QTimer, QUrl, SignalInstance
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlEngine, QQmlExpression
from PySide6.QtQuick import QQuickItem, QQuickWindow

from docsummarizer.model_manager import (
    SUMMARY_TYPE_BRIEF,
    SUMMARY_TYPE_DETAILED,
    SUMMARY_TYPE_STRUCTURED,
    get_model_path,
    gpu_offload_supported,
)
from docsummarizer.ui.app import create_engine

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from docsummarizer.ui.bridge import ConsoleBridge

_TIMEOUT_MS = 10_000

# The document every format carries; the model quotes its sentences back.
_SENTENCES = (
    "Lanterns guide the harbour pilots through the night.",
    "Each pilot records every crossing in a shared ledger.",
    "Storm seasons double the number of ledger entries.",
)
_DOCUMENT = " ".join(_SENTENCES)

_LEAD = "E2ELEAD harbour pilots rely on lanterns and a ledger."
_POINT = "E2EPOINT lanterns guide the pilots"
_SECTION = "E2ESECTION the ledger grows in storms"
_BRIEF = "E2EBRIEF Harbour pilots steer by lanterns and log each crossing."

# The marker that proves each summary type reached its view on the Summary screen.
_SHOWN_MARKER = {
    SUMMARY_TYPE_DETAILED: _POINT,
    SUMMARY_TYPE_BRIEF: _BRIEF,
    SUMMARY_TYPE_STRUCTURED: _SECTION,
}


class _FakeLlama:
    """Stand-in for ``llama_cpp.Llama``: answers by the kind of request it gets.

    A JSON request naming ``sections`` gets a structured reply, any other JSON
    request a detailed one, and a prose request the brief paragraph. Quotes are
    verbatim document sentences, so the real grounding code attaches citations.
    """

    def __init__(self, prompts: list[str], **_kwargs: object) -> None:
        """Record every user prompt into ``prompts``; ignore the model arguments."""
        self._prompts: list[str] = prompts

    @staticmethod
    def tokenize(text: bytes, *, add_bos: bool = True) -> list[int]:
        """Approximate the real tokenizer at three UTF-8 bytes per token."""
        del add_bos
        return [0] * (len(text) // 3)

    def create_chat_completion(
        self, *, messages: list[dict[str, str]], stream: bool = False, **kwargs: object
    ) -> object:
        """Return the reply for ``messages``, as one streamed chunk when ``stream`` is set."""
        user = messages[-1]["content"]
        self._prompts.append(user)
        reply = _reply(user, json_mode="response_format" in kwargs)
        if stream:
            return iter([{"choices": [{"delta": {"content": reply}, "finish_reason": "stop"}]}])
        return {"choices": [{"message": {"content": reply}, "finish_reason": "stop"}]}


def _reply(user: str, *, json_mode: bool) -> str:
    """The model's answer to the prompt ``user``."""
    if not json_mode:
        return _BRIEF
    if '"sections"' in user:
        point = {"text": _SECTION, "quote": _SENTENCES[2]}
        sections: dict[str, list[dict[str, str]]] = {
            name: [point] for name in ("PURPOSE", "METHOD", "RESULTS")
        }
        sections["CONCLUSIONS"] = [{"text": _SECTION}]
        return json.dumps({"sections": sections})
    points = [{"text": f"{_POINT} {n}", "quote": quote} for n, quote in enumerate(_SENTENCES)]
    return json.dumps({"lead": _LEAD, "points": points})


def _pdf_bytes(text: str) -> bytes:
    """A one-page PDF whose only content is ``text`` in a standard font."""
    stream = f"BT /F1 12 Tf 40 720 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b" ".join(
            (
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]",
                b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
            )
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    out += f"startxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def _write_pdf(path: Path) -> None:
    _ = path.write_bytes(_pdf_bytes(_DOCUMENT))


def _write_docx(path: Path) -> None:
    from docx import Document

    document = Document()
    _ = document.add_paragraph(_DOCUMENT)
    document.save(str(path))


def _write_rtf(path: Path) -> None:
    _ = path.write_text(rf"{{\rtf1\ansi {_DOCUMENT}}}", encoding="utf-8")


def _write_plain(path: Path) -> None:
    _ = path.write_text(_DOCUMENT, encoding="utf-8")


# One fixture writer per supported extension.
_WRITERS: dict[str, Callable[[Path], None]] = {
    ".pdf": _write_pdf,
    ".docx": _write_docx,
    ".rtf": _write_rtf,
    ".txt": _write_plain,
    ".md": _write_plain,
    ".text": _write_plain,
}


@dataclass(frozen=True)
class _App:
    """The running production window, its bridge and the prompts the model saw."""

    window: QQuickWindow
    bridge: ConsoleBridge
    prompts: list[str]

    @property
    def screen(self) -> QQuickItem:
        """The Summary screen."""
        return _find(self.window.contentItem(), "SummaryScreen")

    def qml(self, expression: str) -> object:
        """Evaluate ``expression`` among the Summary screen file's ids.

        Returns:
            The expression's value.
        """
        # The screen's first child (its DropArea) lives in SummaryScreen.qml's
        # context, where ids such as ``openDialog`` and ``srcText`` resolve.
        return _qml(self.screen.childItems()[0], expression)


@pytest.fixture(scope="session")
def qapp() -> QGuiApplication:
    """The process-wide Qt application the engine needs.

    Returns:
        The existing application, or a new one.
    """
    existing = QGuiApplication.instance()
    return existing if isinstance(existing, QGuiApplication) else QGuiApplication([])


@pytest.fixture
def app(qapp: QGuiApplication, monkeypatch: pytest.MonkeyPatch) -> Iterator[_App]:
    """The production window and bridge with a downloaded fake model loaded.

    Yields:
        The running app.
    """
    prompts: list[str] = []
    fake = types.ModuleType("llama_cpp")
    fake.__dict__["Llama"] = functools.partial(_FakeLlama, prompts)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    # The GPU probe caches its answer; keep the fake module's out of other tests.
    gpu_offload_supported.cache_clear()
    _ = get_model_path().write_bytes(b"")  # the app only checks that the file exists
    engine, bridge = create_engine(qapp)
    window = engine.rootObjects()[0]
    assert isinstance(window, QQuickWindow)
    _wait_for(bridge.loadComplete, lambda: bridge.property("modelReady") is True, bridge.checkModel)
    yield _App(window, bridge, prompts)
    # Delete the engine (and its window) while the bridge its bindings read is alive.
    shiboken6.delete(engine)
    gpu_offload_supported.cache_clear()


def _wait_for(
    signal: SignalInstance, done: Callable[[], bool], start: Callable[[], object]
) -> None:
    """Run ``start``, then spin the event loop until ``done`` holds after ``signal``."""
    loop = QEventLoop()

    def check(*_args: object) -> None:
        if done():
            loop.quit()

    _ = signal.connect(check)
    QTimer.singleShot(_TIMEOUT_MS, loop.quit)
    _ = start()
    if not done():
        _ = loop.exec()
    _ = signal.disconnect(check)
    assert done(), "the app did not reach the expected state in time"


def _items(root: QQuickItem) -> Iterator[QQuickItem]:
    stack = [root]
    while stack:
        item = stack.pop()
        stack.extend(item.childItems())
        yield item


def _find(root: QQuickItem, type_name: str, text: str = "") -> QQuickItem:
    """The first item under ``root`` of QML type ``type_name`` (with ``text`` when given).

    Returns:
        The item.
    """
    for item in _items(root):
        if item.metaObject().className().startswith(type_name) and (
            not text or item.property("text") == text
        ):
            return item
    pytest.fail(f"no {type_name} {text!r} on screen")


def _qml(scope: QObject, expression: str) -> object:
    """Evaluate ``expression`` in the QML file that declares ``scope``.

    Returns:
        The expression's value.
    """
    evaluated = QQmlExpression(QQmlEngine.contextForObject(scope), scope, expression)
    result: tuple[object, bool] = evaluated.evaluate()
    assert not evaluated.hasError(), evaluated.error().toString()
    return result[0]


def _visible_texts(root: QQuickItem) -> list[str]:
    return [
        str(cast("object", item.property("text")))
        for item in _items(root)
        if item.metaObject().className().startswith("QQuickText") and item.isVisible()
    ]


def _pick(app: _App, button: str, dialog_id: str, target: Path) -> None:
    """Click ``button`` and accept the dialog it opens on ``target``.

    The selection is made before the click: an open dialog keeps the file its
    own view selects, and a headless test cannot click inside that view.
    """
    dialog = app.qml(dialog_id)
    assert isinstance(dialog, QObject)
    _ = dialog.setProperty("selectedFile", QUrl.fromLocalFile(str(target)))
    _ = _qml(_find(app.screen, "ConsoleButton", button), "clicked()")
    assert dialog.property("visible") is True, f"{button} opened no dialog"
    _ = _qml(dialog, "accept()")


def _summarize(app: _App, start: Callable[[], object]) -> dict[str, object]:
    """Run ``start`` and return the summary the bridge hands the screens.

    Returns:
        The summary variant.
    """
    received: list[dict[str, object]] = []
    _ = app.bridge.summaryReady.connect(received.append)
    try:
        _wait_for(app.bridge.summaryReady, lambda: bool(received), start)
    finally:
        _ = app.bridge.summaryReady.disconnect(received.append)
    return received[-1]


def _save(app: _App, target: Path) -> None:
    """Save through the save dialog to ``target`` and wait for the saved toast."""
    toasts: list[str] = []
    _ = app.bridge.toast.connect(toasts.append)
    try:
        _wait_for(
            app.bridge.toast,
            lambda: f"Saved: {target.name}" in toasts,
            functools.partial(_pick, app, "Save Summary", "saveDialog", target),
        )
    finally:
        _ = app.bridge.toast.disconnect(toasts.append)


def _open(app: _App, source: Path) -> None:
    """Open ``source`` through the open dialog and wait for its text."""
    bridge = app.bridge
    _wait_for(
        bridge.docChanged,
        lambda: bridge.property("hasDoc") is True and bridge.property("busy") is False,
        functools.partial(_pick, app, "Select File", "openDialog", source),
    )


def _check_exports(app: _App, tmp_path: Path, source: Path, summary: dict[str, object]) -> None:
    """Export ``summary`` as txt and docx and reopen both files."""
    summary_type = str(summary["summaryType"])
    text = str(summary["text"])
    txt = tmp_path / f"out-{summary_type}.txt"
    _save(app, txt)
    written = txt.read_text(encoding="utf-8")
    assert written.splitlines()[:2] == [f"Summary of: {source.name}", f"Type: {summary_type}"]
    assert written.endswith(text)
    word = tmp_path / f"out-{summary_type}.docx"
    _save(app, word)
    from docx import Document

    paragraphs = [paragraph.text for paragraph in Document(str(word)).paragraphs]
    assert paragraphs == [f"Summary: {source.name}", text]


@pytest.mark.parametrize("suffix", sorted(_WRITERS))
def test_open_summarize_export(app: _App, tmp_path: Path, suffix: str) -> None:
    """A document of each format opens, summarizes in every type and exports.

    Mutations caught: the RTF extractor swapped for the plain-text one (the
    source pane shows the markup), the bridge sending the model an empty text,
    a type change that no longer re-summarizes (the wait times out), the txt
    export without its ``Type:`` line, and the save dialog never passing
    ``as_docx`` (the docx file is plain text).
    """
    source = tmp_path / f"harbour{suffix}"
    _WRITERS[suffix](source)
    assert app.qml("dropZone.visible") is True

    _open(app, source)
    assert app.qml("dropZone.visible") is False
    assert app.bridge.property("currentFileName") == source.name
    shown_source = str(app.qml("srcText.getText(0, srcText.length)"))
    assert shown_source.strip() == _DOCUMENT

    regenerate = _find(app.screen, "ConsoleButton", "Regenerate")
    segments = _find(app.screen, "SegmentedControl")
    steps = (
        (SUMMARY_TYPE_DETAILED, functools.partial(_qml, regenerate, "clicked()")),
        (
            SUMMARY_TYPE_BRIEF,
            functools.partial(_qml, segments, f"selected('{SUMMARY_TYPE_BRIEF}')"),
        ),
        (
            SUMMARY_TYPE_STRUCTURED,
            functools.partial(_qml, segments, f"selected('{SUMMARY_TYPE_STRUCTURED}')"),
        ),
    )
    for summary_type, start in steps:
        assert regenerate.property("enabled") is True
        summary = _summarize(app, start)
        assert summary["summaryType"] == summary_type
        assert app.bridge.property("summaryType") == summary_type
        assert _SENTENCES[0] in app.prompts[-1]
        assert any(_SHOWN_MARKER[summary_type] in text for text in _visible_texts(app.screen))
        _check_exports(app, tmp_path, source, summary)
