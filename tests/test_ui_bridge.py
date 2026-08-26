"""Tests for the Qt <-> QML bridge controller (headless / offscreen).

The bridge runs long ops on a thread pool in production; tests use its
``synchronous=True`` seam so flows run inline and deterministically. A fake
summarizer stands in for the llama-cpp-backed one.
"""

from __future__ import annotations

import pytest
from PySide6.QtGui import QGuiApplication
from PySide6.QtTest import QSignalSpy

from docsummarizer.model_manager import (
    LANGUAGE_AUTO,
    SUMMARY_TYPE_DETAILED,
    StructuredSummary,
    SummarizationCancelledError,
    SummaryPoint,
)
from docsummarizer.provenance import SourceSpan
from docsummarizer.settings import load_settings
from docsummarizer.ui import bridge as bridge_mod
from docsummarizer.ui.bridge import ConsoleBridge, _quant_label, summary_to_variant


@pytest.fixture(scope="session")
def qapp() -> QGuiApplication:
    return QGuiApplication.instance() or QGuiApplication([])


class _FakeSummarizer:
    def __init__(self) -> None:
        self.closed = False
        # Last output language the bridge asked for, so tests can assert the
        # saved setting actually reaches inference.
        self.language: str | None = None

    def summarize_structured(
        self,
        text: str,
        summary_type: str = SUMMARY_TYPE_DETAILED,
        should_cancel: object = None,
        language: str = LANGUAGE_AUTO,
    ) -> StructuredSummary:
        self.language = language
        return StructuredSummary(
            summary_type,
            "the lead",
            [SummaryPoint("a point", SourceSpan(0, 5, "Hello", 1.0))],
            None,
            "rendered text",
        )

    def summarize(
        self,
        text: str,
        summary_type: str = SUMMARY_TYPE_DETAILED,
        language: str = LANGUAGE_AUTO,
    ) -> str:
        self.language = language
        return "FAKE PLAIN SUMMARY"

    def close(self) -> None:
        self.closed = True


def _bridge(synchronous: bool = True) -> ConsoleBridge:
    return ConsoleBridge(summarizer_factory=lambda *_: _FakeSummarizer(), synchronous=synchronous)


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
    variant = summary_to_variant(summary)
    assert variant["summaryType"] == "structured"
    purpose = variant["sections"]["PURPOSE"][0]
    assert purpose["start"] == 2
    assert purpose["end"] == 7
    assert purpose["hasCitation"] is True
    assert variant["sections"]["CONCLUSIONS"][0]["hasCitation"] is False


# --------------------------------------------------------------------------- #
# Model lifecycle + summarize flow
# --------------------------------------------------------------------------- #
def test_check_model_loads_when_present(qapp, monkeypatch) -> None:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", lambda: True)
    bridge = _bridge()
    bridge.checkModel()
    assert bridge._get_model_ready() is True


def test_summarize_emits_marshalled_summary(qapp, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", lambda: True)
    bridge = _bridge()
    bridge.checkModel()

    doc = tmp_path / "d.txt"
    doc.write_text("Hello world. More text follows here.", encoding="utf-8")
    bridge.loadDocument(str(doc))
    assert bridge._get_can_summarize() is True

    spy = QSignalSpy(bridge.summaryReady)
    bridge.summarize()
    assert spy.count() == 1
    variant = spy.at(0)[0]
    assert variant["summaryType"] == "detailed"
    assert variant["lead"] == "the lead"
    assert variant["points"][0]["text"] == "a point"
    assert variant["points"][0]["hasCitation"] is True
    assert bridge._get_busy() is False


def test_load_document_error_blocks_summarize(qapp, monkeypatch) -> None:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", lambda: True)
    bridge = _bridge()
    bridge.checkModel()
    bridge.loadDocument("/nonexistent/file.txt")
    assert bridge._get_has_doc() is False
    assert bridge._get_can_summarize() is False
    assert bridge._status_color == "error"


class _CountingFake(_FakeSummarizer):
    """Fake that records how many times inference is invoked."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def summarize_structured(
        self,
        text: str,
        summary_type: str = SUMMARY_TYPE_DETAILED,
        should_cancel: object = None,
        language: str = LANGUAGE_AUTO,
    ) -> StructuredSummary:
        self.calls += 1
        return super().summarize_structured(text, summary_type, should_cancel, language)


def _loaded_bridge(monkeypatch, tmp_path, *, fake=None):
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", lambda: True)
    summarizer = fake or _FakeSummarizer()
    bridge = ConsoleBridge(summarizer_factory=lambda *_: summarizer, synchronous=True)
    bridge.checkModel()
    doc = tmp_path / "d.txt"
    doc.write_text("Hello world. More text follows here.", encoding="utf-8")
    bridge.loadDocument(str(doc))
    return bridge, summarizer


def test_save_summary_writes_displayed_summary_without_rerunning(
    qapp, monkeypatch, tmp_path
) -> None:
    fake = _CountingFake()
    bridge, _ = _loaded_bridge(monkeypatch, tmp_path, fake=fake)
    bridge.summarize()
    assert fake.calls == 1
    out = tmp_path / "out.txt"
    bridge.saveSummary(out.as_uri(), False)
    assert fake.calls == 1  # save must NOT run inference again
    assert out.read_text(encoding="utf-8").strip() != ""


def test_save_summary_without_a_summary_is_a_noop_with_toast(qapp, monkeypatch, tmp_path) -> None:
    bridge, _ = _loaded_bridge(monkeypatch, tmp_path)
    spy = QSignalSpy(bridge.toast)
    out = tmp_path / "out.txt"
    bridge.saveSummary(out.as_uri(), False)
    assert not out.exists()
    assert spy.count() == 1


def test_can_summarize_notifies_when_model_becomes_ready(qapp, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", lambda: True)
    fake = _FakeSummarizer()
    bridge = ConsoleBridge(summarizer_factory=lambda *_: fake, synchronous=True)
    doc = tmp_path / "d.txt"
    doc.write_text("Hello world.", encoding="utf-8")
    bridge.loadDocument(str(doc))
    assert bridge._get_can_summarize() is False
    spy = QSignalSpy(bridge.docChanged)  # canSummarize's NOTIFY
    bridge.loadModel()
    assert bridge._get_can_summarize() is True
    assert spy.count() >= 1


def test_url_to_path_slot_exposed_to_qml(qapp) -> None:
    bridge = _bridge()
    assert bridge.urlToPath("file:///C:/x/y.pdf") == "C:/x/y.pdf"


def test_batch_bad_folder_toasts_instead_of_crashing(qapp, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", lambda: True)
    bridge = _bridge()
    bridge.checkModel()
    spy = QSignalSpy(bridge.toast)
    bridge.batchProcess(str(tmp_path / "does-not-exist"), str(tmp_path / "out"))
    assert spy.count() == 1
    assert bridge._get_busy() is False


def test_batch_creates_missing_output_dir(qapp, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", lambda: True)
    bridge = _bridge()
    bridge.checkModel()
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("Hello world. More text here.", encoding="utf-8")
    out = tmp_path / "out_missing"
    bridge.batchProcess(str(src), str(out))
    assert out.is_dir()
    assert (out / "a_summary.txt").exists()


def test_async_load_document_completes_off_thread(qapp, monkeypatch, tmp_path) -> None:
    """The real off-thread worker path must finish.

    Regression guard for the Worker being garbage-collected before run()
    completes (QThreadPool.start keeps no Python reference), which silently lost
    the result and hung every async op. Uses synchronous=False on purpose — the
    sync test seam cannot catch this.
    """
    from PySide6.QtCore import QEventLoop, QTimer

    monkeypatch.setattr(bridge_mod, "is_model_downloaded", lambda: True)
    bridge = ConsoleBridge(synchronous=False)
    doc = tmp_path / "d.txt"
    doc.write_text("Hello world. Real off-thread extraction here.", encoding="utf-8")

    loop = QEventLoop()
    bridge.docChanged.connect(lambda: bridge._extracted_text and loop.quit())
    timed_out = {"v": False}

    def on_timeout() -> None:
        timed_out["v"] = True
        loop.quit()

    QTimer.singleShot(8000, on_timeout)
    bridge.loadDocument(doc.as_uri())
    loop.exec()

    assert not timed_out["v"], "async loadDocument hung (worker GC'd before completion)"
    assert bridge._get_has_doc() is True
    assert bridge._get_busy() is False


class _CancelledFake(_FakeSummarizer):
    def summarize_structured(
        self,
        text: str,
        summary_type: str = SUMMARY_TYPE_DETAILED,
        should_cancel: object = None,
        language: str = LANGUAGE_AUTO,
    ) -> StructuredSummary:
        raise SummarizationCancelledError


def test_summarize_handles_cancellation(qapp, monkeypatch, tmp_path) -> None:
    bridge, _ = _loaded_bridge(monkeypatch, tmp_path, fake=_CancelledFake())
    spy = QSignalSpy(bridge.summaryReady)
    bridge.summarize()
    assert spy.count() == 0  # no summary emitted on a clean stop
    assert bridge._status_text == "Stopped"
    assert bridge._get_busy() is False


def test_cancel_summarize_sets_flag_when_busy(qapp) -> None:
    bridge = _bridge()
    bridge._busy = True
    bridge.cancelSummarize()
    assert bridge._cancel_requested is True


def test_compute_label_honest_about_gpu(qapp, monkeypatch) -> None:
    bridge = _bridge()
    bridge._settings.use_gpu = True
    monkeypatch.setattr(bridge_mod, "gpu_offload_supported", lambda: False)
    assert bridge._get_compute_label().startswith("CPU")  # CPU-only build
    monkeypatch.setattr(bridge_mod, "gpu_offload_supported", lambda: True)
    assert bridge._get_compute_label().startswith("GPU")


def test_excepthook_logs_and_toasts(qapp, monkeypatch) -> None:
    import sys

    from docsummarizer.ui import app as app_mod

    bridge = _bridge()
    logged: list[str] = []
    monkeypatch.setattr(app_mod, "log_error", logged.append)
    monkeypatch.setattr(sys, "__excepthook__", lambda *a: None)  # keep test output clean
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


def test_batch_same_stem_inputs_do_not_overwrite(qapp, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", lambda: True)
    bridge = _bridge()
    bridge.checkModel()
    src = tmp_path / "src"
    src.mkdir()
    (src / "report.txt").write_text("Hello world one.", encoding="utf-8")
    (src / "report.md").write_text("Hello world two.", encoding="utf-8")
    out = tmp_path / "out"
    bridge.batchProcess(str(src), str(out))
    produced = sorted(p.name for p in out.glob("*_summary.txt"))
    assert len(produced) == 2, produced  # both saved, neither overwritten


def test_url_to_path_handles_windows_and_posix_urls() -> None:
    # A QML FileDialog hands back a file:// URL; a Windows URL must not keep the
    # spurious leading slash before the drive letter (the cause of the silent
    # file-open failure on Windows). This must hold on every platform.
    assert bridge_mod._url_to_path("file:///C:/Users/me/report.pdf") == "C:/Users/me/report.pdf"
    assert bridge_mod._url_to_path("file:///home/me/report.pdf") == "/home/me/report.pdf"
    assert bridge_mod._url_to_path("/home/me/report.pdf") == "/home/me/report.pdf"


def test_load_document_from_file_url(qapp, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", lambda: True)
    bridge = _bridge()
    bridge.checkModel()
    doc = tmp_path / "sample.txt"
    doc.write_text("Hello world. This is a real document.", encoding="utf-8")
    bridge.loadDocument(doc.as_uri())  # file:///...
    assert bridge._get_has_doc() is True
    assert bridge._extracted_text != ""


# --------------------------------------------------------------------------- #
# Settings: GPU persists immediately; threads persist only on reload
# --------------------------------------------------------------------------- #
def test_toggle_gpu_persists_immediately(qapp) -> None:
    bridge = _bridge()
    bridge.toggleGpu(True)
    assert bridge._settings.use_gpu is True
    assert load_settings().use_gpu is True


def test_threads_arm_reload_but_persist_only_on_reload(qapp, monkeypatch) -> None:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", lambda: True)
    bridge = _bridge()
    bridge.checkModel()  # model loaded so reload can arm

    bridge.setThreads(3)
    assert bridge._settings.n_threads == 3
    assert bridge._get_reload_armed() is True
    assert load_settings().n_threads is None  # not persisted yet

    bridge.reloadModel()
    assert load_settings().n_threads == 3  # persisted on reload
    assert bridge._get_reload_armed() is False


def test_set_appearance_persists(qapp) -> None:
    bridge = _bridge()
    bridge.setAppearance("Dark")
    assert load_settings().appearance == "Dark"


def test_set_output_language_persists_without_arming_reload(qapp) -> None:
    """The language lives in the prompt, so the loaded model needs no reload."""
    bridge = _bridge()
    bridge.setOutputLanguage("Chinese")
    assert load_settings().output_language == "Chinese"
    assert bridge._get_output_language() == "Chinese"
    assert bridge._get_reload_armed() is False


def test_output_language_reaches_summarize(qapp, monkeypatch, tmp_path) -> None:
    fake = _FakeSummarizer()
    bridge, _ = _loaded_bridge(monkeypatch, tmp_path, fake=fake)
    bridge.setOutputLanguage("Japanese")
    bridge.summarize()
    assert fake.language == "Japanese"


def test_output_language_reaches_batch(qapp, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", lambda: True)
    fake = _FakeSummarizer()
    bridge = ConsoleBridge(summarizer_factory=lambda *_: fake, synchronous=True)
    bridge.checkModel()
    bridge.setOutputLanguage("French")
    (tmp_path / "a.txt").write_text("Document A content here.", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()

    bridge.batchProcess(str(tmp_path), str(out))

    assert fake.language == "French"


def test_output_language_defaults_to_auto_at_summarize(qapp, monkeypatch, tmp_path) -> None:
    fake = _FakeSummarizer()
    bridge, _ = _loaded_bridge(monkeypatch, tmp_path, fake=fake)
    bridge.summarize()
    assert fake.language == LANGUAGE_AUTO


def test_reload_not_armed_without_loaded_model(qapp) -> None:
    bridge = _bridge()  # no checkModel → no summarizer
    bridge.setThreads(5)
    assert bridge._get_reload_armed() is False


# --------------------------------------------------------------------------- #
# Batch
# --------------------------------------------------------------------------- #
def test_batch_process_summarizes_folder(qapp, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", lambda: True)
    bridge = _bridge()
    bridge.checkModel()

    (tmp_path / "a.txt").write_text("Document A content here.", encoding="utf-8")
    (tmp_path / "b.md").write_text("Document B content here.", encoding="utf-8")
    (tmp_path / "ignore.png").write_bytes(b"not a document")
    out = tmp_path / "out"
    out.mkdir()

    spy = QSignalSpy(bridge.batchComplete)
    bridge.batchProcess(str(tmp_path), str(out))

    assert spy.count() == 1
    done_count, total, failures, _out_dir = spy.at(0)
    assert done_count == 2
    assert total == 2  # the .png is not a supported document
    assert failures == []
    assert (out / "a_summary.txt").exists()
    assert (out / "b_summary.txt").exists()


def test_batch_process_no_documents_emits_toast(qapp, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bridge_mod, "is_model_downloaded", lambda: True)
    bridge = _bridge()
    bridge.checkModel()
    out = tmp_path / "out"
    out.mkdir()
    spy = QSignalSpy(bridge.toast)
    bridge.batchProcess(str(tmp_path), str(out))  # empty folder
    assert spy.count() == 1
