# SPDX-License-Identifier: MIT
"""Background workers for the Qt UI.

Long operations (model load, download, summarize, batch) run on a
``QThreadPool`` so the UI thread stays responsive. A worker never touches QML
directly — it reports back via signals, which Qt delivers on the receiver's
(UI) thread. Mirrors the threading discipline of the old CustomTkinter
controller (``self.after(0, ...)``), but with Qt's queued signal/slot delivery.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal

from docsummarizer.logger import log_error


class WorkerSignals(QObject):
    """Result channel for a :class:`Worker` (a QObject so signals are bound)."""

    done = Signal(object)
    failed = Signal(str)


class Worker(QRunnable):
    """Run ``fn(*args, **kwargs)`` off the UI thread, reporting via signals.

    Emits ``signals.done(result)`` on success or ``signals.failed(message)`` on
    any exception — the pool thread never crashes on a backend error.
    """

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:
            log_error(f"Worker failed: {exc!s}")
            self.signals.failed.emit(str(exc))
        else:
            self.signals.done.emit(result)
