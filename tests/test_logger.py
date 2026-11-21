# SPDX-License-Identifier: MIT
"""Tests for `docsummarizer.logger`."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from docsummarizer import logger as logger_module

if TYPE_CHECKING:
    from pathlib import Path


def test_get_log_directory_lands_under_isolated_tmp(tmp_path: Path) -> None:
    """`conftest._isolated_app_dirs` redirects platform env vars to tmp_path;
    `get_log_directory` must resolve under it (the platform branches
    themselves are covered in test_paths.py)."""
    d = logger_module.get_log_directory()
    assert str(d).startswith(str(tmp_path))
    assert d.name == "logs"
    assert d.exists()


def test_get_logger_returns_singleton() -> None:
    a = logger_module.get_logger()
    b = logger_module.get_logger()
    assert a is b


def test_get_logger_thread_safe() -> None:
    """Concurrent first-callers must not produce duplicate FileHandlers.

    Pre-fix: lazy init was `if _logger is None: _logger = setup_logger()`
    with no lock. Two simultaneous first calls could both pass the check
    and both add a FileHandler.
    """
    import threading

    barrier = threading.Barrier(8)
    results: list[logging.Logger] = []

    def worker():
        barrier.wait()
        results.append(logger_module.get_logger())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 8
    first = results[0]
    assert all(r is first for r in results)
    file_handlers = [h for h in first.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1


def test_system_info_has_expected_fields() -> None:
    info = logger_module.get_system_info()
    for key in ("platform", "architecture", "cpu_count", "python_version"):
        assert key in info, f"missing {key!r}"


def test_timer_logs_elapsed(caplog: pytest.LogCaptureFixture) -> None:
    logging.getLogger("DocSummarizer").addHandler(caplog.handler)
    logging.getLogger("DocSummarizer").setLevel(logging.DEBUG)

    with caplog.at_level(logging.INFO, logger="DocSummarizer"), logger_module.Timer("widget"):
        pass

    messages = "\n".join(r.getMessage() for r in caplog.records)
    assert "Starting: widget" in messages
    assert "Completed: widget" in messages


def test_timer_logs_failure_on_exception(caplog: pytest.LogCaptureFixture) -> None:
    logging.getLogger("DocSummarizer").addHandler(caplog.handler)
    logging.getLogger("DocSummarizer").setLevel(logging.DEBUG)

    with (
        caplog.at_level(logging.ERROR, logger="DocSummarizer"),
        pytest.raises(ValueError, match="boom"),
        logger_module.Timer("widget"),
    ):
        raise ValueError("boom")

    messages = "\n".join(r.getMessage() for r in caplog.records)
    assert "Failed: widget" in messages
