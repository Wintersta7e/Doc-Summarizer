# SPDX-License-Identifier: MIT
"""Tests for the thread-pool worker: it reports its result or its error by signal."""

from __future__ import annotations

from typing import NoReturn

from docsummarizer.ui.workers import Worker


def test_run_emits_the_result() -> None:
    worker = Worker(lambda: 42)
    results: list[object] = []
    failures: list[str] = []
    _ = worker.signals.done.connect(results.append)
    _ = worker.signals.failed.connect(failures.append)

    worker.run()

    assert results == [42]
    assert failures == []


def test_run_emits_the_error_message_instead_of_raising() -> None:
    def broken() -> NoReturn:
        msg = "no such file"
        raise OSError(msg)

    worker = Worker(broken)
    results: list[object] = []
    failures: list[str] = []
    _ = worker.signals.done.connect(results.append)
    _ = worker.signals.failed.connect(failures.append)

    worker.run()

    assert results == []
    assert failures == ["no such file"]
