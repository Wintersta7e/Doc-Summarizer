"""The Python <-> QML bridge: a ``ConsoleBridge`` QObject exposed to QML.

This is the controller for the redesigned UI — the Qt successor to the old
CustomTkinter ``DocSummarizerApp``. It owns the document/summary state and a
lock-guarded ``Summarizer``, runs long operations on a ``QThreadPool``, and
surfaces everything to QML as properties (with NOTIFY signals) and slots.

Model metadata (name/size/quant) is read live from ``DEFAULT_MODEL`` so the UI
can never drift from the actual model. Structured summaries are marshalled to a
plain dict (a ``QVariant`` map) since QML cannot consume a frozen dataclass.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import Property, QMutex, QObject, QThreadPool, QUrl, Signal, Slot

from docsummarizer import document_parser
from docsummarizer.io_helpers import write_summary_docx, write_summary_txt
from docsummarizer.logger import log_debug, log_error, log_info
from docsummarizer.model_manager import (
    DEFAULT_MODEL,
    OUTPUT_LANGUAGES,
    SUMMARY_TYPE_DETAILED,
    SUMMARY_TYPES,
    StructuredSummary,
    SummarizationCancelledError,
    Summarizer,
    download_model,
    get_model_path,
    gpu_offload_supported,
    is_model_downloaded,
    normalize_language,
)
from docsummarizer.settings import Settings, load_settings, save_settings
from docsummarizer.ui.workers import Worker

# Context window the model is loaded with (mirrors Summarizer.__init__ default).
_DEFAULT_CONTEXT = 8192

SummarizerFactory = Callable[[Path, "int | None", int], Summarizer]


def _quant_label(filename: str) -> str:
    """Extract a quantization label (e.g. ``Q4_K_M``) from a GGUF filename."""
    stem = filename.removesuffix(".gguf")
    for part in reversed(stem.split("-")):
        if part.upper().startswith("Q") and any(ch.isdigit() for ch in part):
            return part
    return ""


_DRIVE_LETTER_RE = re.compile(r"^/([A-Za-z]:)")


def _url_to_path(url: str) -> str:
    """Convert a QML ``file://`` URL — or an already-local path — to a usable
    local filesystem path, cross-platform.

    A QML ``FileDialog`` returns ``file:///C:/x`` on Windows and
    ``file:///home/x`` on POSIX. Qt only strips the leading slash before a
    Windows drive letter on Windows builds, so normalise that ourselves —
    otherwise ``/C:/x`` reaches the parser and the open silently fails.
    """
    if url.startswith("file:"):
        return _DRIVE_LETTER_RE.sub(r"\1", QUrl(url).toLocalFile())
    return url


def _point_to_variant(point: Any) -> dict[str, Any]:
    """Marshal a SummaryPoint (+ optional citation) to a QML-friendly dict."""
    citation = point.citation
    return {
        "text": point.text,
        "hasCitation": citation is not None,
        "start": citation.start if citation else -1,
        "end": citation.end if citation else -1,
        "quote": citation.quote if citation else "",
        "score": citation.score if citation else 0.0,
    }


def summary_to_variant(summary: StructuredSummary) -> dict[str, Any]:
    """Marshal a ``StructuredSummary`` to a plain dict QML can read.

    QML cannot consume a frozen dataclass, so points become dicts and section
    lists become a dict of lists. Citation offsets are preserved for the
    source-pane provenance highlight.
    """
    return {
        "summaryType": summary.summary_type,
        "lead": summary.lead or "",
        "points": [_point_to_variant(p) for p in summary.points],
        "sections": {
            name: [_point_to_variant(p) for p in pts]
            for name, pts in (summary.sections or {}).items()
        },
        "text": summary.text,
    }


def _default_factory(model_path: Path, n_threads: int | None, n_gpu_layers: int) -> Summarizer:
    return Summarizer(model_path, n_threads=n_threads, n_gpu_layers=n_gpu_layers)


class ConsoleBridge(QObject):
    """Controller object registered with the QML engine as ``bridge``."""

    # Property-change notifications.
    busyChanged = Signal()
    statusChanged = Signal()
    modelReadyChanged = Signal()
    docChanged = Signal()
    settingsChanged = Signal()
    summaryTypeChanged = Signal()
    reloadArmedChanged = Signal()
    downloadChanged = Signal()

    # One-shot events.
    progress = Signal(float, str)
    # "QVariant" so the marshalled dict reaches QML as a property-accessible map.
    summaryReady = Signal("QVariant")  # type: ignore[arg-type]
    summaryError = Signal(str)
    loadComplete = Signal(bool, str)
    toast = Signal(str)
    savedFlash = Signal()
    batchProgress = Signal(int, int, str)
    batchComplete = Signal(int, int, list, str)
    batchRowsChanged = Signal()

    def __init__(
        self,
        *,
        summarizer_factory: SummarizerFactory | None = None,
        synchronous: bool = False,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings: Settings = load_settings()
        self._summarizer: Summarizer | None = None
        self._mutex = QMutex()
        self._pool = QThreadPool.globalInstance()
        self._synchronous = synchronous
        self._factory: SummarizerFactory = summarizer_factory or _default_factory

        self._busy = False
        self._status_text = ""
        self._status_color = "ok"  # ok | warn | error
        self._current_file = ""
        self._extracted_text = ""
        self._summary_type = SUMMARY_TYPE_DETAILED
        self._reload_armed = False
        self._download_pct = 0.0
        self._downloading = False
        self._batch_rows: list[dict[str, Any]] = []
        self._last_summary: StructuredSummary | None = None
        self._cancel_requested = False
        # Hold running workers: QThreadPool.start() does not keep a Python
        # reference, so without this the Worker (and its signals) is GC'd before
        # run() finishes and the done/failed signal is silently lost.
        self._active_workers: set[Worker] = set()
        # canSummarize depends on model-ready as well as the document, but a
        # Qt Property has a single NOTIFY; re-fire docChanged on model changes so
        # the Summarize/Save buttons re-evaluate when the model finishes loading.
        self.modelReadyChanged.connect(self.docChanged)

    # -- threading seam ----------------------------------------------------- #
    def _run(self, work: Callable[[], Any], on_done: Callable[[Any], None]) -> None:
        """Run ``work`` off-thread (or inline in synchronous/test mode).

        Logs the operation's start, completion (with duration), and failure —
        keyed off the current status text (never document content) — so a silent
        hang shows up in the log as a "started" with no matching "done".
        """
        if self._synchronous:
            on_done(work())
            return
        op = self._status_text or "operation"
        started = time.monotonic()
        worker = Worker(work)
        self._active_workers.add(worker)  # keep alive until run() finishes
        log_debug(f"{op}: started")

        def on_success(result: Any) -> None:
            self._active_workers.discard(worker)
            log_info(f"{op}: done in {time.monotonic() - started:.2f}s")
            on_done(result)

        def on_failure(message: str) -> None:
            self._active_workers.discard(worker)
            log_error(f"{op}: failed after {time.monotonic() - started:.2f}s — {message}")
            self._on_worker_failed(message)

        worker.signals.done.connect(on_success)
        worker.signals.failed.connect(on_failure)
        self._pool.start(worker)

    def _on_worker_failed(self, message: str) -> None:
        if self._downloading:
            self._downloading = False
            self.downloadChanged.emit()
        self._finish(f"Error: {message}", "error")
        self.summaryError.emit(message)

    # -- read-state properties --------------------------------------------- #
    def _get_busy(self) -> bool:
        return self._busy

    busy = Property(bool, _get_busy, notify=busyChanged)

    def _get_status_text(self) -> str:
        return self._status_text

    statusText = Property(str, _get_status_text, notify=statusChanged)

    def _get_status_color(self) -> str:
        return self._status_color

    statusColor = Property(str, _get_status_color, notify=statusChanged)

    def _get_model_ready(self) -> bool:
        return self._get_summarizer() is not None

    modelReady = Property(bool, _get_model_ready, notify=modelReadyChanged)

    def _get_model_downloaded(self) -> bool:
        return is_model_downloaded()

    modelDownloaded = Property(bool, _get_model_downloaded, notify=modelReadyChanged)

    def _get_model_name(self) -> str:
        return DEFAULT_MODEL.name

    modelName = Property(str, _get_model_name, constant=True)

    def _get_model_size(self) -> float:
        return DEFAULT_MODEL.size_gb

    modelSizeGb = Property(float, _get_model_size, constant=True)

    def _get_model_quant(self) -> str:
        return _quant_label(DEFAULT_MODEL.filename)

    modelQuant = Property(str, _get_model_quant, constant=True)

    def _get_context_size(self) -> int:
        return _DEFAULT_CONTEXT

    contextSize = Property(int, _get_context_size, constant=True)

    def _get_compute_label(self) -> str:
        threads = self._settings.n_threads or _auto_threads()
        gpu = self._settings.use_gpu and gpu_offload_supported()
        return f"{'GPU' if gpu else 'CPU'} · {threads}T"

    computeLabel = Property(str, _get_compute_label, notify=settingsChanged)

    def _get_threads(self) -> int:
        return self._settings.n_threads or _auto_threads()

    threads = Property(int, _get_threads, notify=settingsChanged)

    def _get_cpu_count(self) -> int:
        return os.cpu_count() or 8

    cpuCount = Property(int, _get_cpu_count, constant=True)

    def _get_gpu(self) -> bool:
        return self._settings.use_gpu

    gpuEnabled = Property(bool, _get_gpu, notify=settingsChanged)

    def _get_gpu_supported(self) -> bool:
        return gpu_offload_supported()

    gpuSupported = Property(bool, _get_gpu_supported, constant=True)

    def _get_appearance(self) -> str:
        return self._settings.appearance

    appearance = Property(str, _get_appearance, notify=settingsChanged)

    def _get_output_language(self) -> str:
        return self._settings.output_language

    outputLanguage = Property(str, _get_output_language, notify=settingsChanged)

    def _get_output_languages(self) -> list[str]:
        return list(OUTPUT_LANGUAGES)

    outputLanguages = Property(list, _get_output_languages, constant=True)

    def _get_reload_armed(self) -> bool:
        return self._reload_armed

    reloadArmed = Property(bool, _get_reload_armed, notify=reloadArmedChanged)

    def _get_has_doc(self) -> bool:
        return bool(self._extracted_text)

    hasDoc = Property(bool, _get_has_doc, notify=docChanged)

    def _get_can_summarize(self) -> bool:
        return self._get_model_ready() and bool(self._extracted_text)

    canSummarize = Property(bool, _get_can_summarize, notify=docChanged)

    def _get_current_file(self) -> str:
        return self._current_file

    currentFileName = Property(str, _get_current_file, notify=docChanged)

    def _get_extracted_text(self) -> str:
        return self._extracted_text

    extractedText = Property(str, _get_extracted_text, notify=docChanged)

    def _get_summary_type(self) -> str:
        return self._summary_type

    summaryType = Property(str, _get_summary_type, notify=summaryTypeChanged)

    def _get_summary_types(self) -> list[str]:
        return list(SUMMARY_TYPES)

    summaryTypes = Property(list, _get_summary_types, constant=True)

    def _get_download_pct(self) -> float:
        return self._download_pct

    downloadPercent = Property(float, _get_download_pct, notify=downloadChanged)

    def _get_batch_rows(self) -> list[dict[str, Any]]:
        return self._batch_rows

    batchRows = Property(list, _get_batch_rows, notify=batchRowsChanged)

    # -- internal state setters (emit on change) --------------------------- #
    def _set_busy(self, value: bool) -> None:
        if self._busy != value:
            self._busy = value
            self.busyChanged.emit()

    def _set_status(self, text: str, color: str = "ok") -> None:
        self._status_text = text
        self._status_color = color
        self.statusChanged.emit()

    def _finish(self, status: str, color: str = "ok") -> None:
        """Clear the busy flag and publish a final status (operation done)."""
        self._set_busy(False)
        self._set_status(status, color)

    def _get_summarizer(self) -> Summarizer | None:
        self._mutex.lock()
        try:
            return self._summarizer
        finally:
            self._mutex.unlock()

    def _set_summarizer(self, summarizer: Summarizer | None) -> None:
        self._mutex.lock()
        old = self._summarizer
        self._summarizer = summarizer
        self._mutex.unlock()
        # Close the previous model outside the lock (frees RAM/VRAM).
        if old is not None and old is not summarizer:
            old.close()
        self.modelReadyChanged.emit()

    def _arm_reload(self) -> None:
        if self._get_summarizer() is not None and not self._reload_armed:
            self._reload_armed = True
            self.reloadArmedChanged.emit()

    # -- model lifecycle ---------------------------------------------------- #
    @Slot()
    def checkModel(self) -> None:
        if is_model_downloaded():
            self._set_status(f"{DEFAULT_MODEL.name} ready", "ok")
            self.loadModel()
        else:
            self._set_status(f"{DEFAULT_MODEL.name} not downloaded", "warn")
        self.modelReadyChanged.emit()

    def _build_summarizer_async(self, status: str, *, free_old: bool, emit_complete: bool) -> None:
        """Build the model off-thread, then publish it as the active summarizer.

        ``free_old`` releases the current model before building so two never
        coexist (avoids VRAM OOM on reload); ``emit_complete`` fires
        ``loadComplete`` for the initial load.
        """
        self._set_busy(True)
        self._set_status(status, "ok")
        n_threads = self._settings.n_threads
        n_gpu = self._settings.n_gpu_layers
        path = get_model_path()
        if free_old:
            self._set_summarizer(None)

        def work() -> Summarizer:
            return self._factory(path, n_threads, n_gpu)

        def done(summarizer: Any) -> None:
            self._set_summarizer(summarizer)
            self._finish("Ready")
            if emit_complete:
                self.loadComplete.emit(True, "")

        self._run(work, done)

    @Slot()
    def loadModel(self) -> None:
        if self._busy or not is_model_downloaded():
            return
        self._build_summarizer_async("Loading model…", free_old=False, emit_complete=True)

    @Slot()
    def reloadModel(self) -> None:
        if not self._reload_armed or self._busy:
            return
        save_settings(self._settings)
        self._reload_armed = False
        self.reloadArmedChanged.emit()
        self._build_summarizer_async("Reloading…", free_old=True, emit_complete=False)

    @Slot()
    def beginDownload(self) -> None:
        if self._downloading:
            return
        self._downloading = True
        self._download_pct = 0.0
        self._set_status("Downloading model…", "ok")
        self.downloadChanged.emit()

        def on_progress(pct: float, message: str) -> None:
            self._download_pct = pct
            self.downloadChanged.emit()
            self.progress.emit(pct, message)

        def work() -> tuple[Path, str | None]:
            return download_model(progress_callback=on_progress)

        def done(result: Any) -> None:
            _path, error = result
            self._downloading = False
            self.downloadChanged.emit()
            if error:
                self._set_status(error, "error")
                self.toast.emit("Download failed")
            else:
                self._set_status("Model downloaded", "ok")
                self.modelReadyChanged.emit()
                self.loadModel()

        self._run(work, done)

    # -- document + summary ------------------------------------------------- #
    @Slot(str, result=str)
    def urlToPath(self, url: str) -> str:
        """Cross-platform file-URL → local-path conversion, callable from QML."""
        return _url_to_path(url)

    @Slot(str)
    def loadDocument(self, file_path: str) -> None:
        if self._busy:
            return
        path = _url_to_path(file_path)
        self._current_file = Path(path).name
        self._extracted_text = ""
        self._last_summary = None
        self._set_busy(True)
        self._set_status("Extracting text…", "ok")
        self.docChanged.emit()

        def work() -> tuple[str, str | None]:
            return document_parser.extract_text(path)

        def done(result: Any) -> None:
            text, error = result
            if error:
                self._extracted_text = ""
                self._finish(error, "error")
            else:
                self._extracted_text = text
                self._finish("Text extracted")
            self.docChanged.emit()

        self._run(work, done)

    @Slot()
    def unloadDocument(self) -> None:
        self._current_file = ""
        self._extracted_text = ""
        self._last_summary = None
        self.docChanged.emit()

    @Slot(str)
    def setSummaryType(self, summary_type: str) -> None:
        # Ignore type changes mid-generation: summarize() would no-op on the busy
        # guard, leaving the label and the shown summary out of sync.
        if self._busy:
            return
        if summary_type in SUMMARY_TYPES and summary_type != self._summary_type:
            self._summary_type = summary_type
            self.summaryTypeChanged.emit()
            if self._extracted_text:
                self.summarize()

    @Slot()
    def summarize(self) -> None:
        summarizer = self._get_summarizer()
        if summarizer is None or not self._extracted_text or self._busy:
            return
        text = self._extracted_text
        summary_type = self._summary_type
        language = self._settings.output_language
        self._cancel_requested = False
        self._set_busy(True)
        self._set_status("Generating summary…", "ok")

        def work() -> StructuredSummary | None:
            try:
                return summarizer.summarize_structured(
                    text,
                    summary_type,
                    should_cancel=lambda: self._cancel_requested,
                    language=language,
                )
            except SummarizationCancelledError:
                return None

        def done(summary: Any) -> None:
            if summary is None:  # cancelled cleanly between chunks
                self._finish("Stopped", "warn")
                return
            self._last_summary = summary
            self._finish("Summary complete")
            self.summaryReady.emit(summary_to_variant(summary))

        self._run(work, done)

    @Slot()
    def cancelSummarize(self) -> None:
        """Request a clean stop; the worker aborts at the next chunk boundary."""
        if self._busy:
            self._cancel_requested = True
            self._set_status("Stopping…", "warn")

    @Slot()
    def regenerate(self) -> None:
        self.summarize()

    @Slot(str, bool)
    def saveSummary(self, file_path: str, as_docx: bool) -> None:
        # Write the summary already on screen — never re-run inference (that both
        # wasted compute and could save text different from what the user saw,
        # and a second concurrent run is unsafe on the shared llama context).
        summary = self._last_summary
        if summary is None:
            self.toast.emit("Generate a summary first")
            return
        path = _url_to_path(file_path)
        source = self._current_file
        summary_type = summary.summary_type
        self._set_status("Saving summary…", "ok")

        def work() -> None:
            if as_docx:
                write_summary_docx(path, source_name=source, summary=summary.text)
            else:
                write_summary_txt(
                    path, source_name=source, summary=summary.text, summary_type=summary_type
                )

        def done(_result: Any) -> None:
            self.toast.emit(f"Saved: {Path(path).name}")

        self._run(work, done)

    # -- settings ----------------------------------------------------------- #
    @Slot(bool)
    def toggleGpu(self, enabled: bool) -> None:
        self._settings.use_gpu = enabled
        save_settings(self._settings)  # GPU change persists immediately
        self.settingsChanged.emit()
        self.savedFlash.emit()
        self._arm_reload()

    @Slot(int)
    def setThreads(self, n_threads: int) -> None:
        self._settings.n_threads = max(1, n_threads)
        # Threads persist on reload (not on every slider tick); just arm reload.
        self.settingsChanged.emit()
        self._arm_reload()

    @Slot(str)
    def setAppearance(self, mode: str) -> None:
        self._settings.appearance = mode
        save_settings(self._settings)  # appearance persists + applies immediately
        self.settingsChanged.emit()
        self.savedFlash.emit()

    @Slot(str)
    def setOutputLanguage(self, language: str) -> None:
        """Set the summary's language. Applies to the next summary, not the model.

        No reload is armed: the language lives in the prompt, so the loaded
        model is unaffected and only the next run changes.
        """
        chosen = normalize_language(language)
        if chosen == self._settings.output_language:
            return
        self._settings.output_language = chosen
        save_settings(self._settings)  # language persists immediately
        self.settingsChanged.emit()
        self.savedFlash.emit()

    # -- batch -------------------------------------------------------------- #
    def _resolve_batch_inputs(self, folder: Path, out: Path) -> list[Path] | None:
        """Validate the batch folders and ensure the output dir exists.

        Returns the documents to process, or ``None`` after toasting why not —
        so a missing/empty folder is a friendly message, not an unhandled slot
        exception that would silently abort the whole operation.
        """
        if not folder.is_dir():
            self.toast.emit("That folder does not exist")
            return None
        files = document_parser.find_documents(folder)
        if not files:
            self.toast.emit("No supported documents in that folder")
            return None
        try:
            out.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.toast.emit("Could not create the output folder")
            return None
        return files

    @staticmethod
    def _batch_output_path(out: Path, stem: str, used: set[str]) -> Path:
        """A unique ``<stem>_summary.txt`` under ``out`` — disambiguating
        same-stem inputs (e.g. report.pdf + report.docx) so they don't silently
        overwrite each other."""
        name = f"{stem}_summary.txt"
        counter = 2
        while name in used:
            name = f"{stem}_{counter}_summary.txt"
            counter += 1
        used.add(name)
        return out / name

    @Slot(str, str)
    def batchProcess(self, folder_path: str, out_dir: str) -> None:
        """Summarize every supported document in a folder, writing .txt outputs.

        Maintains ``batchRows`` (a list of {name, status, tokens}) so the UI can
        show per-file QUEUED → PROCESSING → DONE/FAILED status as work proceeds.
        """
        summarizer = self._get_summarizer()
        if summarizer is None or self._busy:
            return
        folder = Path(_url_to_path(folder_path))
        out = Path(_url_to_path(out_dir))
        files = self._resolve_batch_inputs(folder, out)
        if files is None:
            return
        summary_type = self._summary_type
        language = self._settings.output_language
        total = len(files)
        self._batch_rows = [{"name": f.name, "status": "QUEUED", "tokens": 0} for f in files]
        self.batchRowsChanged.emit()
        self._set_busy(True)
        self._set_status(f"Processing 0/{total}…", "ok")

        def set_row(index: int, status: str, tokens: int = 0) -> None:
            rows = [dict(r) for r in self._batch_rows]
            rows[index]["status"] = status
            if tokens:
                rows[index]["tokens"] = tokens
            self._batch_rows = rows
            self.batchRowsChanged.emit()

        used_names: set[str] = set()

        def work() -> tuple[int, list[dict[str, str]]]:
            done_count = 0
            failures: list[dict[str, str]] = []
            for index, path in enumerate(files):
                set_row(index, "PROCESSING")
                self.batchProgress.emit(index, total, path.name)
                text, error = document_parser.extract_text(str(path))
                if error:
                    set_row(index, "FAILED")
                    failures.append({"name": path.name, "error": error})
                    continue
                try:
                    summary = summarizer.summarize(text, summary_type, language=language)
                    write_summary_txt(
                        self._batch_output_path(out, path.stem, used_names),
                        source_name=path.name,
                        summary=summary,
                        summary_type=summary_type,
                    )
                    set_row(index, "DONE", max(len(summary) // 4, 1))
                    done_count += 1
                except Exception as exc:  # one bad file must not abort the batch
                    set_row(index, "FAILED")
                    failures.append({"name": path.name, "error": str(exc)})
            return done_count, failures

        def done(result: Any) -> None:
            done_count, failures = result
            color = "ok" if not failures else "warn"
            self._finish(f"Batch complete: {done_count}/{total}", color)
            self.batchComplete.emit(done_count, total, failures, str(out))

        self._run(work, done)

    @Slot()
    def shutdown(self) -> None:
        log_info("UI shutting down; releasing model")
        self._set_summarizer(None)


def _auto_threads() -> int:
    cpu_count = os.cpu_count() or 8
    return max(4, cpu_count // 2)
