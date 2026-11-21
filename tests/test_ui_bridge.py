# SPDX-License-Identifier: MIT
"""Tests for the Qt <-> QML bridge controller (headless / offscreen).

The bridge runs long ops on a thread pool in production; tests use its
``synchronous=True`` seam so flows run inline and deterministically. A fake
summarizer stands in for the llama-cpp-backed one.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, TypedDict, cast

import pytest
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtTest import QSignalSpy
from typing_extensions import override

from docsummarizer.model_manager import (
    LANGUAGE_AUTO,
    StructuredSummary,
    SummarizationCancelledError,
    SummaryPoint,
)
from docsummarizer.provenance import SourceSpan
from docsummarizer.settings import load_settings
from docsummarizer.ui import app as app_mod
from docsummarizer.ui import bridge as bridge_mod
from docsummarizer.ui.bridge import (
    ConsoleBridge,
    SummaryEngine,
    _quant_label,
    summary_to_variant,
)

_LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class _PointVariant(TypedDict):
    """The keys of one marshalled summary point these tests read."""

    text: str
    start: int
    end: int
    hasCitation: bool


class _SummaryVariant(TypedDict):
    """The keys of a marshalled summary these tests read."""

    summaryType: str
    lead: str
    points: list[_PointVariant]
    sections: dict[str, list[_PointVariant]]


@pytest.fixture(scope="session")
def qapp() -> QGuiApplication:
    app = QGuiApplication.instance() or QGuiApplication([])
    assert isinstance(app, QGuiApplication)
    return app


def _signal_args(spy: QSignalSpy, index: int = 0) -> list[object]:
    """The arguments of the index-th emission a spy recorded."""
    args: list[object] = spy.at(index)
    return args


def _as_variant(value: object) -> _SummaryVariant:
    """Read a marshalled summary as the keys these tests check."""
    return cast("_SummaryVariant", value)


def _model_on_disk() -> bool:
    return True


class _FakeSummarizer(SummaryEngine):
    def __init__(self) -> None:
        self.closed: bool = False
        # Last output language the bridge asked for, so tests can assert the
        # saved setting actually reaches inference.
        self.language: str | None = None

    @override
    def summarize_structured(
        self,
        text: str,
        summary_type: str,
        /,
        *,
        should_cancel: Callable[[], bool] | None,
        language: str,
    ) -> StructuredSummary:
        self.language = language
        return StructuredSummary(
            summary_type,
            "the lead",
            [SummaryPoint("a point", SourceSpan(0, 5, "Hello", 1.0))],
            None,
            "rendered text",
        )

    @override
    def summarize(self, text: str, summary_type: str, /, *, language: str) -> str:
        self.language = language
        return "FAKE PLAIN SUMMARY"

    @override
    def close(self) -> None:
        self.closed = True


def _qt(bridge: ConsoleBridge, name: str) -> object:
    """Read a bridge property through Qt's meta-object, as QML reads it."""
    return cast("object", bridge.property(name))


def _bridge(*, synchronous: bool = True) -> ConsoleBridge:
    return ConsoleBridge(
        summarizer_factory=lambda _model, _threads, _layers: _FakeSummarizer(),
        synchronous=synchronous,
    )


# --------------------------------------------------------------------------- #
# Pure marshalling
# --------------------------------------------------------------------------- #
def test_quant_label_parses_gguf_filename() -> None:
    assert _quant_label("Qwen3-4B-Instruct-2507-Q4_K_M.gguf") == "Q4_K_M"
    assert _quant_label("model-without-quant.gguf") == ""


def test_summary_to_variant_shapes_points_and_sections() -> None:
    summary = StructuredSummary(
        "structured",
        None,
        [],
        {
            "PURPOSE": [SummaryPoint("p", SourceSpan(2, 7, "world", 0.9))],
            "CONCLUSIONS": [SummaryPoint("c", None)],
        },
        "txt",
    )
    variant = _as_variant(summary_to_variant(summary))
    assert variant["summaryType"] == "structured"
    purpose = variant["sections"]["PURPOSE"][0]
    assert purpose["start"] == 2
    assert purpose["end"] == 7
    assert purpose["hasCitation"] is True
    assert variant["sections"]["CONCLUSIONS"][0]["hasCitation"] is False


# --------------------------------------------------------------------------- #
# Model lifecycle + summarize flow
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("qapp")
def test_check_model_loads_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", _model_on_disk)
    bridge = _bridge()
    bridge.checkModel()
    assert _qt(bridge, "modelReady") is True


@pytest.mark.usefixtures("qapp")
def test_summarize_emits_marshalled_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", _model_on_disk)
    bridge = _bridge()
    bridge.checkModel()

    doc = tmp_path / "d.txt"
    _ = doc.write_text("Hello world. More text follows here.", encoding="utf-8")
    bridge.loadDocument(str(doc))
    assert _qt(bridge, "canSummarize") is True

    spy = QSignalSpy(bridge.summaryReady)
    bridge.summarize()
    assert spy.count() == 1
    variant = _as_variant(_signal_args(spy)[0])
    assert variant["summaryType"] == "detailed"
    assert variant["lead"] == "the lead"
    assert variant["points"][0]["text"] == "a point"
    assert variant["points"][0]["hasCitation"] is True
    assert _qt(bridge, "busy") is False


@pytest.mark.usefixtures("qapp")
def test_load_document_error_blocks_summarize(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", _model_on_disk)
    bridge = _bridge()
    bridge.checkModel()
    bridge.loadDocument("/nonexistent/file.txt")
    assert _qt(bridge, "hasDoc") is False
    assert _qt(bridge, "canSummarize") is False
    assert _qt(bridge, "statusColor") == "error"


class _CountingFake(_FakeSummarizer):
    """Fake that records how many times inference is invoked."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: int = 0

    @override
    def summarize_structured(
        self,
        text: str,
        summary_type: str,
        /,
        *,
        should_cancel: Callable[[], bool] | None,
        language: str,
    ) -> StructuredSummary:
        self.calls += 1
        return super().summarize_structured(
            text, summary_type, should_cancel=should_cancel, language=language
        )


def _loaded_bridge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, fake: _FakeSummarizer | None = None
) -> tuple[ConsoleBridge, _FakeSummarizer]:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", _model_on_disk)
    summarizer = fake or _FakeSummarizer()
    bridge = ConsoleBridge(
        summarizer_factory=lambda _model, _threads, _layers: summarizer, synchronous=True
    )
    bridge.checkModel()
    doc = tmp_path / "d.txt"
    _ = doc.write_text("Hello world. More text follows here.", encoding="utf-8")
    bridge.loadDocument(str(doc))
    return bridge, summarizer


@pytest.mark.usefixtures("qapp")
def test_save_summary_writes_displayed_summary_without_rerunning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _CountingFake()
    bridge, _ = _loaded_bridge(monkeypatch, tmp_path, fake=fake)
    bridge.summarize()
    assert fake.calls == 1
    out = tmp_path / "out.txt"
    bridge.saveSummary(out.as_uri())
    assert fake.calls == 1  # save must NOT run inference again
    assert out.read_text(encoding="utf-8").strip() != ""


@pytest.mark.usefixtures("qapp")
def test_save_summary_without_a_summary_is_a_noop_with_toast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bridge, _ = _loaded_bridge(monkeypatch, tmp_path)
    spy = QSignalSpy(bridge.toast)
    out = tmp_path / "out.txt"
    bridge.saveSummary(out.as_uri())
    assert not out.exists()
    assert spy.count() == 1


@pytest.mark.usefixtures("qapp")
def test_can_summarize_notifies_when_model_becomes_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", _model_on_disk)
    fake = _FakeSummarizer()
    bridge = ConsoleBridge(
        summarizer_factory=lambda _model, _threads, _layers: fake, synchronous=True
    )
    doc = tmp_path / "d.txt"
    _ = doc.write_text("Hello world.", encoding="utf-8")
    bridge.loadDocument(str(doc))
    assert _qt(bridge, "canSummarize") is False
    spy = QSignalSpy(bridge.docChanged)  # canSummarize's NOTIFY
    bridge.loadModel()
    assert _qt(bridge, "canSummarize") is True
    assert spy.count() >= 1


@pytest.mark.usefixtures("qapp")
def test_url_to_path_slot_exposed_to_qml() -> None:
    bridge = _bridge()
    assert bridge.urlToPath("file:///C:/x/y.pdf") == "C:/x/y.pdf"


@pytest.mark.usefixtures("qapp")
def test_batch_bad_folder_toasts_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", _model_on_disk)
    bridge = _bridge()
    bridge.checkModel()
    spy = QSignalSpy(bridge.toast)
    bridge.batchProcess(str(tmp_path / "does-not-exist"), str(tmp_path / "out"))
    assert spy.count() == 1
    assert _qt(bridge, "busy") is False


@pytest.mark.usefixtures("qapp")
def test_batch_creates_missing_output_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", _model_on_disk)
    bridge = _bridge()
    bridge.checkModel()
    src = tmp_path / "src"
    src.mkdir()
    _ = (src / "a.txt").write_text("Hello world. More text here.", encoding="utf-8")
    out = tmp_path / "out_missing"
    bridge.batchProcess(str(src), str(out))
    assert out.is_dir()
    assert (out / "a.txt.detailed.txt").exists()


@pytest.mark.usefixtures("qapp")
def test_async_load_document_completes_off_thread(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The real off-thread worker path must finish.

    Regression guard for the Worker being garbage-collected before run()
    completes (QThreadPool.start keeps no Python reference), which silently lost
    the result and hung every async op. Uses synchronous=False on purpose — the
    sync test seam cannot catch this.
    """
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", _model_on_disk)
    bridge = ConsoleBridge(synchronous=False)
    doc = tmp_path / "d.txt"
    _ = doc.write_text("Hello world. Real off-thread extraction here.", encoding="utf-8")

    loop = QEventLoop()

    def on_doc_changed() -> None:
        if _qt(bridge, "extractedText"):
            loop.quit()

    _ = bridge.docChanged.connect(on_doc_changed)
    timed_out = {"v": False}

    def on_timeout() -> None:
        timed_out["v"] = True
        loop.quit()

    QTimer.singleShot(8000, on_timeout)
    bridge.loadDocument(doc.as_uri())
    _ = loop.exec()

    assert not timed_out["v"], "async loadDocument hung (worker GC'd before completion)"
    assert _qt(bridge, "hasDoc") is True
    assert _qt(bridge, "busy") is False


class _CancelledFake(_FakeSummarizer):
    @override
    def summarize_structured(
        self,
        text: str,
        summary_type: str,
        /,
        *,
        should_cancel: Callable[[], bool] | None,
        language: str,
    ) -> StructuredSummary:
        raise SummarizationCancelledError


@pytest.mark.usefixtures("qapp")
def test_summarize_handles_cancellation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bridge, _ = _loaded_bridge(monkeypatch, tmp_path, fake=_CancelledFake())
    spy = QSignalSpy(bridge.summaryReady)
    bridge.summarize()
    assert spy.count() == 0  # no summary emitted on a clean stop
    assert _qt(bridge, "statusText") == "Stopped"
    assert _qt(bridge, "busy") is False


@pytest.mark.usefixtures("qapp")
def test_cancel_summarize_sets_flag_when_busy() -> None:
    bridge = _bridge()
    bridge._busy = True
    bridge.cancelSummarize()
    assert bridge._cancel_requested is True


@pytest.mark.usefixtures("qapp")
def test_compute_label_honest_about_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = _bridge()
    bridge.toggleGpu()  # off by default
    monkeypatch.setattr(bridge_mod, "gpu_offload_supported", lambda: False)
    label = _qt(bridge, "computeLabel")
    assert isinstance(label, str)
    assert label.startswith("CPU")  # CPU-only build
    monkeypatch.setattr(bridge_mod, "gpu_offload_supported", lambda: True)
    label = _qt(bridge, "computeLabel")
    assert isinstance(label, str)
    assert label.startswith("GPU")


def _silent_excepthook(*_args: object) -> None:
    """Stand in for the default hook, which would print the traceback."""


@pytest.mark.usefixtures("qapp")
def test_excepthook_logs_and_toasts(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = _bridge()
    logged: list[str] = []
    monkeypatch.setattr(app_mod, "log_error", logged.append)
    monkeypatch.setattr(sys, "__excepthook__", _silent_excepthook)  # keep test output clean
    spy = QSignalSpy(bridge.toast)
    original = sys.excepthook
    try:
        app_mod._install_excepthook(bridge)
        exc = ValueError("boom")
        sys.excepthook(type(exc), exc, exc.__traceback__)
    finally:
        sys.excepthook = original
    assert any("boom" in m for m in logged)
    assert spy.count() == 1


@pytest.mark.usefixtures("qapp")
def test_batch_same_stem_inputs_do_not_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", _model_on_disk)
    bridge = _bridge()
    bridge.checkModel()
    src = tmp_path / "src"
    src.mkdir()
    _ = (src / "report.txt").write_text("Hello world one.", encoding="utf-8")
    _ = (src / "report.md").write_text("Hello world two.", encoding="utf-8")
    out = tmp_path / "out"
    bridge.batchProcess(str(src), str(out))
    produced = sorted(p.name for p in out.glob("*.detailed.txt"))
    # both saved, neither overwritten
    assert produced == ["report.md.detailed.txt", "report.txt.detailed.txt"], produced


@pytest.mark.usefixtures("qapp")
def test_url_to_path_handles_windows_and_posix_urls() -> None:
    # A QML FileDialog hands back a file:// URL; a Windows URL must not keep the
    # spurious leading slash before the drive letter (the cause of the silent
    # file-open failure on Windows). This must hold on every platform.
    bridge = _bridge()
    assert bridge.urlToPath("file:///C:/Users/me/report.pdf") == "C:/Users/me/report.pdf"
    assert bridge.urlToPath("file:///home/me/report.pdf") == "/home/me/report.pdf"
    assert bridge.urlToPath("/home/me/report.pdf") == "/home/me/report.pdf"


@pytest.mark.usefixtures("qapp")
def test_load_document_from_file_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", _model_on_disk)
    bridge = _bridge()
    bridge.checkModel()
    doc = tmp_path / "sample.txt"
    _ = doc.write_text("Hello world. This is a real document.", encoding="utf-8")
    bridge.loadDocument(doc.as_uri())  # file:///...
    assert _qt(bridge, "hasDoc") is True
    assert _qt(bridge, "extractedText") != ""


# --------------------------------------------------------------------------- #
# Settings: GPU persists immediately; threads persist only on reload
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("qapp")
def test_toggle_gpu_persists_immediately() -> None:
    bridge = _bridge()
    bridge.toggleGpu()  # off by default
    assert _qt(bridge, "gpuEnabled") is True
    assert load_settings().use_gpu is True


@pytest.mark.usefixtures("qapp")
def test_threads_arm_reload_but_persist_only_on_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", _model_on_disk)
    bridge = _bridge()
    bridge.checkModel()  # model loaded so reload can arm

    bridge.setThreads(3)
    assert _qt(bridge, "threads") == 3
    assert _qt(bridge, "reloadArmed") is True
    assert load_settings().n_threads is None  # not persisted yet

    bridge.reloadModel()
    assert load_settings().n_threads == 3  # persisted on reload
    assert _qt(bridge, "reloadArmed") is False


@pytest.mark.usefixtures("qapp")
def test_set_appearance_persists() -> None:
    bridge = _bridge()
    bridge.setAppearance("Dark")
    assert load_settings().appearance == "Dark"


@pytest.mark.usefixtures("qapp")
def test_set_output_language_persists_without_arming_reload() -> None:
    """The language lives in the prompt, so the loaded model needs no reload."""
    bridge = _bridge()
    bridge.setOutputLanguage("Chinese")
    assert load_settings().output_language == "Chinese"
    assert _qt(bridge, "outputLanguage") == "Chinese"
    assert _qt(bridge, "reloadArmed") is False


@pytest.mark.usefixtures("qapp")
def test_output_language_reaches_summarize(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _FakeSummarizer()
    bridge, _ = _loaded_bridge(monkeypatch, tmp_path, fake=fake)
    bridge.setOutputLanguage("Japanese")
    bridge.summarize()
    assert fake.language == "Japanese"


@pytest.mark.usefixtures("qapp")
def test_output_language_reaches_batch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", _model_on_disk)
    fake = _FakeSummarizer()
    bridge = ConsoleBridge(
        summarizer_factory=lambda _model, _threads, _layers: fake, synchronous=True
    )
    bridge.checkModel()
    bridge.setOutputLanguage("French")
    _ = (tmp_path / "a.txt").write_text("Document A content here.", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()

    bridge.batchProcess(str(tmp_path), str(out))

    assert fake.language == "French"


@pytest.mark.usefixtures("qapp")
def test_output_language_defaults_to_auto_at_summarize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FakeSummarizer()
    bridge, _ = _loaded_bridge(monkeypatch, tmp_path, fake=fake)
    bridge.summarize()
    assert fake.language == LANGUAGE_AUTO


@pytest.mark.usefixtures("qapp")
def test_reload_not_armed_without_loaded_model() -> None:
    bridge = _bridge()  # no checkModel → no summarizer
    bridge.setThreads(5)
    assert _qt(bridge, "reloadArmed") is False


# --------------------------------------------------------------------------- #
# Batch
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("qapp")
def test_batch_process_summarizes_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", _model_on_disk)
    bridge = _bridge()
    bridge.checkModel()

    _ = (tmp_path / "a.txt").write_text("Document A content here.", encoding="utf-8")
    _ = (tmp_path / "b.md").write_text("Document B content here.", encoding="utf-8")
    _ = (tmp_path / "ignore.png").write_bytes(b"not a document")
    out = tmp_path / "out"
    out.mkdir()

    spy = QSignalSpy(bridge.batchComplete)
    bridge.batchProcess(str(tmp_path), str(out))

    assert spy.count() == 1
    done_count, total, failures, _out_dir = _signal_args(spy)
    assert done_count == 2
    assert total == 2  # the .png is not a supported document
    assert failures == []
    assert (out / "a.txt.detailed.txt").exists()
    assert (out / "b.md.detailed.txt").exists()


@pytest.mark.usefixtures("qapp")
def test_batch_process_no_documents_emits_toast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", _model_on_disk)
    bridge = _bridge()
    bridge.checkModel()
    out = tmp_path / "out"
    out.mkdir()
    spy = QSignalSpy(bridge.toast)
    bridge.batchProcess(str(tmp_path), str(out))  # empty folder
    assert spy.count() == 1


# --------------------------------------------------------------------------- #
# Batch: never replace an unchecked file
# --------------------------------------------------------------------------- #
_SENTINEL = b"a file the user wrote\n"


class _CountingSummarizer(_FakeSummarizer):
    """Counts plain summaries and runs an optional hook inside each model call."""

    def __init__(self, hook: Callable[[str], None] | None = None) -> None:
        super().__init__()
        self.calls: int = 0
        self.hook: Callable[[str], None] | None = hook

    @override
    def summarize(self, text: str, summary_type: str, /, *, language: str) -> str:
        self.calls += 1
        if self.hook is not None:
            self.hook(text)
        return super().summarize(text, summary_type, language=language)


def _batch_bridge(monkeypatch: pytest.MonkeyPatch, fake: _FakeSummarizer) -> ConsoleBridge:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", _model_on_disk)
    bridge = ConsoleBridge(
        summarizer_factory=lambda _model, _threads, _layers: fake, synchronous=True
    )
    bridge.checkModel()
    return bridge


def _batch_inputs(folder: Path, *names: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        _ = (folder / name).write_text(f"content of {name}", encoding="utf-8")
    return folder


def _rows_by_name(bridge: ConsoleBridge) -> dict[str, dict[str, object]]:
    rows = cast("list[dict[str, object]]", _qt(bridge, "batchRows"))
    return {str(row["name"]): row for row in rows}


@pytest.mark.usefixtures("qapp")
def test_batch_never_replaces_unchecked_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches Batch without the existence check: it replaces the planned output."""
    fake = _CountingSummarizer()
    bridge = _batch_bridge(monkeypatch, fake)
    src = _batch_inputs(tmp_path / "in", "a.txt", "b.md", "c.txt")
    out = tmp_path / "out"
    out.mkdir()
    planned = out / "c.txt.detailed.txt"
    _ = planned.write_bytes(_SENTINEL)
    user_file = out / "a_summary.txt"
    _ = user_file.write_bytes(_SENTINEL)
    mtimes = {p: p.stat().st_mtime_ns for p in (planned, user_file)}

    bridge.batchProcess(str(src), str(out))

    rows = _rows_by_name(bridge)
    identical = sum(
        p.read_bytes() == _SENTINEL and p.stat().st_mtime_ns == mtimes[p] for p in mtimes
    )
    written = sum((out / n).exists() for n in ("a.txt.detailed.txt", "b.md.detailed.txt"))
    _LOG.info("written=%s preexisting_identical=%s", written, identical)
    assert rows["c.txt"]["status"] == "SKIPPED"
    assert rows["c.txt"]["reason"] == "summary exists"
    assert written == 2
    assert identical == 2
    assert fake.calls == 2


@pytest.mark.usefixtures("qapp")
def test_batch_outputs_are_distinct_and_unscanned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches stem-based names (a.txt and a.md collide) and a rescan of the outputs."""
    fake = _CountingSummarizer()
    bridge = _batch_bridge(monkeypatch, fake)
    src = _batch_inputs(tmp_path / "in", "a.txt", "a.md")
    out = src / "summaries"

    bridge.batchProcess(str(src), str(out))
    first = _rows_by_name(bridge)
    bridge.batchProcess(str(src), str(out))
    second = _rows_by_name(bridge)

    _LOG.info("rows per run: %s, %s", len(first), len(second))
    assert sorted(p.name for p in out.iterdir()) == ["a.md.detailed.txt", "a.txt.detailed.txt"]
    assert len(first) == len(second) == 2
    assert all(row["status"] == "DONE" for row in first.values())
    assert all(row["status"] == "SKIPPED" for row in second.values())
    assert fake.calls == 2


@pytest.mark.usefixtures("qapp")
def test_batch_output_equal_to_input_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches Batch without the output-equals-input check: it writes into the input."""
    fake = _CountingSummarizer()
    bridge = _batch_bridge(monkeypatch, fake)
    src = _batch_inputs(tmp_path / "in", "a.txt", "b.md")
    (src / "sub").mkdir()
    before = sorted(src.rglob("*"))

    for out in (src, src / "sub" / ".."):
        spy = QSignalSpy(bridge.toast)
        bridge.batchProcess(str(src), str(out))
        assert spy.count() == 1
        assert str(_signal_args(spy)[0]).startswith("Choose a different output folder")
        assert _qt(bridge, "busy") is False
        assert _qt(bridge, "batchRows") == []

    assert sorted(src.rglob("*")) == before
    assert fake.calls == 0


@pytest.mark.windows
@pytest.mark.usefixtures("qapp")
def test_batch_two_writers_appeared_during_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches Batch publishing over a file that appeared after its check."""
    out = tmp_path / "out"
    target = out / "b.md.detailed.txt"

    def other_writer(text: str) -> None:
        if text == "content of b.md":
            _ = target.write_bytes(_SENTINEL)

    bridge = _batch_bridge(monkeypatch, _CountingSummarizer(other_writer))
    src = _batch_inputs(tmp_path / "in", "a.txt", "b.md")

    bridge.batchProcess(str(src), str(out))

    row = _rows_by_name(bridge)["b.md"]
    assert target.read_bytes() == _SENTINEL
    assert row["status"] == "SKIPPED"
    assert row["reason"] == "appeared during run"
    assert sorted(out.glob("*.partial")) == []
