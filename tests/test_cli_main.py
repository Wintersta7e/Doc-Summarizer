# SPDX-License-Identifier: MIT
"""Tests for the CLI's setup paths and per-file outcomes.

``cli.main()`` runs against a fake model: these pin the exits a user sees when
the model is missing, the input is wrong or the model fails to load, and what
``summarize_file`` reports when one file fails to extract, summarize or write.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, NoReturn, cast

import pytest

from docsummarizer import cli

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from docsummarizer.model_manager import Summarizer

USAGE_ERROR = 2


class _FakeSummarizer:
    """Answers every document with a fixed summary, or raises ``error``."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error: Exception | None = error

    def summarize(self, text: str, **_options: str) -> str:
        if self.error is not None:
            raise self.error
        return f"summary of {text}"

    def close(self) -> None:
        pass


def _summarizer(error: Exception | None = None) -> Summarizer:
    return cast("Summarizer", cast("object", _FakeSummarizer(error)))


def _run_cli(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int | str | None:
    monkeypatch.setattr(sys, "argv", ["docsummarizer-cli", *argv])
    try:
        cli.main()
    except SystemExit as exc:
        return exc.code
    return 0


@pytest.fixture
def model_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model counts as downloaded and loads as the fake."""
    monkeypatch.setattr(cli, "ensure_model", lambda: True)

    def load(*_args: object, **_kwargs: object) -> _FakeSummarizer:
        return _FakeSummarizer()

    monkeypatch.setattr(cli, "Summarizer", load)


def _download(error: str | None, calls: list[str]) -> Callable[..., tuple[Path | None, str | None]]:
    def download(
        progress_callback: Callable[[float, str], None],
    ) -> tuple[Path | None, str | None]:
        calls.append("download")
        progress_callback(100.0, "done")
        return None, error

    return download


def test_print_progress_draws_the_bar_and_ends_the_line_at_100(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli.print_progress(50.0, "halfway")
    half = capsys.readouterr().out
    cli.print_progress(100.0, "done")
    full = capsys.readouterr().out

    assert half == "\r[" + "=" * 20 + "-" * 20 + "] 50.0% - halfway"
    assert full.endswith("] 100.0% - done\n")


def test_ensure_model_skips_the_download_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "is_model_downloaded", lambda: True)
    monkeypatch.setattr(cli, "download_model", _download(None, calls))

    assert cli.ensure_model()
    assert calls == []


def test_ensure_model_downloads_a_missing_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "is_model_downloaded", lambda: False)
    monkeypatch.setattr(cli, "download_model", _download(None, calls))

    assert cli.ensure_model()
    assert calls == ["download"]
    assert "Model downloaded successfully!" in capsys.readouterr().out


def test_ensure_model_reports_a_failed_download(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "is_model_downloaded", lambda: False)
    monkeypatch.setattr(cli, "download_model", _download("disk full", []))

    assert not cli.ensure_model()
    assert "Error: disk full" in capsys.readouterr().out


def test_main_exits_1_when_the_model_cannot_be_had(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "ensure_model", lambda: False)

    assert _run_cli(monkeypatch, "doc.txt") == 1


@pytest.mark.usefixtures("model_ready")
def test_download_only_stops_after_the_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run_cli(monkeypatch, "--download-only") == 0
    assert "Model is ready." in capsys.readouterr().out


@pytest.mark.usefixtures("model_ready")
def test_main_without_an_input_is_a_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run_cli(monkeypatch) == USAGE_ERROR


@pytest.mark.usefixtures("model_ready")
def test_main_refuses_a_missing_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run_cli(monkeypatch, str(tmp_path / "absent.txt")) == 1
    assert "Path does not exist" in capsys.readouterr().out


@pytest.mark.usefixtures("model_ready")
def test_main_refuses_a_folder_without_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run_cli(monkeypatch, str(tmp_path)) == 1
    assert "No supported documents found" in capsys.readouterr().out


def test_main_exits_1_when_the_model_fails_to_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "bad model file"
        raise RuntimeError(msg)

    doc = tmp_path / "doc.txt"
    _ = doc.write_text("some text", encoding="utf-8")
    monkeypatch.setattr(cli, "ensure_model", lambda: True)
    monkeypatch.setattr(cli, "Summarizer", broken)

    assert _run_cli(monkeypatch, str(doc)) == 1
    assert "Error loading model: bad model file" in capsys.readouterr().out


@pytest.mark.usefixtures("model_ready")
def test_a_console_run_prints_the_summary_and_the_language(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    doc = tmp_path / "doc.txt"
    _ = doc.write_text("some text", encoding="utf-8")

    assert _run_cli(monkeypatch, str(doc), "-l", "French") == 0
    out = capsys.readouterr().out
    assert "Summary language: French" in out
    assert "summary of some text" in out
    assert "Done. 1 written, 0 skipped, 0 failed." in out


@pytest.mark.usefixtures("model_ready")
def test_a_folder_run_into_its_own_folder_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _ = (tmp_path / "doc.txt").write_text("some text", encoding="utf-8")

    assert _run_cli(monkeypatch, str(tmp_path), "-o", str(tmp_path)) == USAGE_ERROR
    assert "Choose a different output folder" in capsys.readouterr().err


@pytest.mark.usefixtures("model_ready")
def test_a_single_input_into_a_new_folder_creates_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = tmp_path / "doc.txt"
    _ = doc.write_text("some text", encoding="utf-8")
    out = tmp_path / "new"

    assert _run_cli(monkeypatch, str(doc), "-o", str(out) + os.sep) == 0
    assert (out / "doc.txt.detailed.txt").is_file()


@pytest.mark.usefixtures("model_ready")
def test_an_output_folder_that_cannot_be_made_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    doc = tmp_path / "doc.txt"
    _ = doc.write_text("some text", encoding="utf-8")
    blocker = tmp_path / "blocker"
    _ = blocker.write_text("a file, not a folder", encoding="utf-8")

    out = str(blocker / "sub") + os.sep
    assert _run_cli(monkeypatch, str(doc), "-o", out) == USAGE_ERROR
    assert "could not create the output folder" in capsys.readouterr().err


def test_summarize_file_reports_an_extraction_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    doc = tmp_path / "empty.txt"
    _ = doc.write_text("   ", encoding="utf-8")

    assert cli.summarize_file(str(doc), _summarizer()) == "failed"
    assert "Error: No text could be extracted" in capsys.readouterr().out


def test_summarize_file_reports_a_failed_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    doc = tmp_path / "doc.txt"
    _ = doc.write_text("some text", encoding="utf-8")

    outcome = cli.summarize_file(str(doc), _summarizer(RuntimeError("out of memory")))

    assert outcome == "failed"
    assert "Error: out of memory" in capsys.readouterr().out


def test_summarize_file_reports_an_unwritable_target(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    doc = tmp_path / "doc.txt"
    _ = doc.write_text("some text", encoding="utf-8")
    target = tmp_path / "missing-folder" / "out.txt"

    assert cli.summarize_file(str(doc), _summarizer(), target) == "failed"
    assert f"Error: could not write {target}" in capsys.readouterr().out
