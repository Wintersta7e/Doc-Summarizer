# SPDX-License-Identifier: MIT
"""Background workers for the Qt UI.

Long operations (model load, download, summarize, batch) run on a
``QThreadPool`` so the UI thread stays responsive. A worker never touches QML
directly — it reports back via signals, which Qt delivers on the receiver's
(UI) thread. Mirrors the threading discipline of the old CustomTkinter
controller (``self.after(0, ...)``), but with Qt's queued signal/slot delivery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, final

from PySide6.QtCore import QObject, QRunnable, Signal
from typing_extensions import override

from docsummarizer.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable


@final
class WorkerSignals(QObject):
    """Result channel for a :class:`Worker` (a QObject so signals are bound)."""

    done = Signal(object)
    failed = Signal(str)


@final
class Worker(QRunnable):
    """Run ``fn()`` off the UI thread, reporting via signals.

    Emits ``signals.done(result)`` on success or ``signals.failed(message)`` on
    any exception — the pool thread never crashes on a backend error.
    """

    def __init__(self, fn: Callable[[], object]) -> None:
        """Wrap ``fn``; nothing runs until the pool calls :meth:`run`."""
        super().__init__()
        self._fn = fn
        self.signals = WorkerSignals()

    @override
    def run(self) -> None:
        """Call ``fn`` and emit its result or its error message."""
        try:
            result = self._fn()
        except Exception as exc:  # the worker thread's edge: report, never crash the pool
            logger = get_logger()
            logger.exception("Worker failed")

            self.signals.failed.emit(str(exc))

        else:
            self.signals.done.emit(result)
