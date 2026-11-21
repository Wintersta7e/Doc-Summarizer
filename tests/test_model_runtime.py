# SPDX-License-Identifier: MIT
"""Tests for the summarizer's glue to the llama runtime, against a fake ``llama_cpp``.

The gate never installs the runtime extra, so a stand-in package takes its
place in ``sys.modules``. These pin what the model is loaded with, how Stop
installs and clears llama.cpp's abort callback (and degrades when a build
lacks it), how the context pointer and perf counters are found, and the
download's success and failure reports.
"""

from __future__ import annotations

import os
import sys
import types
from typing import TYPE_CHECKING, NoReturn, cast

import pytest

from docsummarizer import model_manager
from docsummarizer.model_manager import (
    SUMMARY_TYPE_BRIEF,
    CreateChatCompletionStreamResponse,
    SummarizationCancelledError,
    Summarizer,
)

if TYPE_CHECKING:
    import ctypes
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from pathlib import Path


class _CallbackType:
    """``ggml_abort_callback``: wraps a Python function, or none to clear."""

    def __init__(self, fn: Callable[[object], bool] | None = None) -> None:
        self.fn: Callable[[object], bool] | None = fn


class _AbortBinding:
    """The low-level binding's abort-callback setter; records each call."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail: bool = fail
        self.calls: list[tuple[object, _CallbackType, ctypes.c_void_p]] = []

    def __call__(self, ctx: object, callback: _CallbackType, user_data: ctypes.c_void_p) -> None:
        if self.fail:
            msg = "this build cannot set the callback"
            raise RuntimeError(msg)
        self.calls.append((ctx, callback, user_data))


class _FakeLlama:
    """``llama_cpp.Llama``: records its options and streams a fixed answer."""

    def __init__(self, **options: object) -> None:
        self.options: dict[str, object] = options
        self.ctx: object = object()
        self.on_chunk: Callable[[], None] = lambda: None
        self.fail_with: Exception | None = None

    def tokenize(self, text: bytes, *, add_bos: bool = True) -> list[int]:
        del add_bos
        return [0] * (len(text) // 3)

    def create_chat_completion(
        self, *, messages: Sequence[Mapping[str, str]], stream: bool = False, **kwargs: object
    ) -> Iterator[CreateChatCompletionStreamResponse]:
        del messages, kwargs
        assert stream, "a cancellable run streams"
        return self._chunks()

    def _chunks(self) -> Iterator[CreateChatCompletionStreamResponse]:
        self.on_chunk()
        if self.fail_with is not None:
            raise self.fail_with
        yield {"choices": [{"delta": {"content": "the summary"}, "finish_reason": None}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}


def _install_llama_cpp(
    monkeypatch: pytest.MonkeyPatch, setter: _AbortBinding | None = None
) -> types.ModuleType:
    """Put a fake ``llama_cpp`` in ``sys.modules``; returns its low-level module."""
    pkg = types.ModuleType("llama_cpp")
    low = types.ModuleType("llama_cpp.llama_cpp")
    if setter is not None:
        monkeypatch.setattr(low, "llama_set_abort_callback", setter, raising=False)
        monkeypatch.setattr(low, "ggml_abort_callback", _CallbackType, raising=False)
    monkeypatch.setattr(pkg, "Llama", _FakeLlama, raising=False)
    monkeypatch.setattr(pkg, "llama_cpp", low, raising=False)
    monkeypatch.setitem(sys.modules, "llama_cpp", pkg)
    return low


def _llm(summarizer: Summarizer) -> _FakeLlama:
    assert isinstance(summarizer.llm, _FakeLlama)
    return summarizer.llm


def test_the_model_loads_with_the_runtime_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _ = _install_llama_cpp(monkeypatch)
    model = tmp_path / "model.gguf"

    summarizer = Summarizer(model, n_ctx=4096, n_threads=3, n_gpu_layers=-1)

    assert _llm(summarizer).options == {
        "model_path": str(model),
        "n_ctx": 4096,
        "n_threads": 3,
        "n_threads_batch": 3,
        "n_gpu_layers": -1,
        "verbose": False,
    }


@pytest.mark.parametrize(("cores", "threads"), [(12, 6), (2, 4), (None, 4)])
def test_threads_default_to_half_the_cores_but_at_least_four(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cores: int | None, threads: int
) -> None:
    _ = _install_llama_cpp(monkeypatch)
    monkeypatch.setattr(os, "cpu_count", lambda: cores)

    assert Summarizer(tmp_path / "model.gguf").n_threads == threads


def test_stop_installs_the_abort_callback_for_one_completion_then_clears_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    setter = _AbortBinding()
    _ = _install_llama_cpp(monkeypatch, setter)
    stop = {"now": False}
    answers: list[bool] = []

    def check() -> bool:
        return stop["now"]

    with Summarizer(tmp_path / "model.gguf") as summarizer:
        llm = _llm(summarizer)

        def probe() -> None:
            fn = setter.calls[-1][1].fn
            assert fn is not None
            answers.append(fn(None))
            stop["now"] = True
            answers.append(fn(None))
            stop["now"] = False

        llm.on_chunk = probe
        result = summarizer.summarize("a short document", SUMMARY_TYPE_BRIEF, should_cancel=check)
        # Cleared when the completion ends, not only when the model closes.
        after_completion = list(setter.calls)

    assert result == "the summary"
    assert answers == [False, True]
    (installed_ctx, _cb, _data), (cleared_ctx, cleared, _none) = after_completion
    assert installed_ctx is cleared_ctx is llm.ctx
    assert cleared.fn is None


def test_a_cancel_check_that_raises_never_reaches_llama_cpp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    setter = _AbortBinding()
    _ = _install_llama_cpp(monkeypatch, setter)
    summarizer = Summarizer(tmp_path / "model.gguf")
    broken = {"now": False}

    def check() -> bool:
        if broken["now"]:
            msg = "the flag is gone"
            raise RuntimeError(msg)
        return False

    answers: list[bool] = []

    def probe() -> None:
        fn = setter.calls[-1][1].fn
        assert fn is not None
        broken["now"] = True
        answers.append(fn(None))
        broken["now"] = False

    _llm(summarizer).on_chunk = probe
    _ = summarizer.summarize("a short document", SUMMARY_TYPE_BRIEF, should_cancel=check)

    assert answers == [False]


def test_a_build_that_cannot_set_the_callback_still_streams(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    setter = _AbortBinding(fail=True)
    _ = _install_llama_cpp(monkeypatch, setter)
    summarizer = Summarizer(tmp_path / "model.gguf")

    result = summarizer.summarize(
        "a short document", SUMMARY_TYPE_BRIEF, should_cancel=lambda: False
    )

    assert result == "the summary"
    assert setter.calls == []


def test_a_failure_while_stopping_is_reported_as_cancelled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _ = _install_llama_cpp(monkeypatch, _AbortBinding())
    summarizer = Summarizer(tmp_path / "model.gguf")
    stop = {"now": False}
    llm = _llm(summarizer)
    llm.fail_with = RuntimeError("llama_decode returned 2")

    def abort() -> None:
        stop["now"] = True

    llm.on_chunk = abort

    with pytest.raises(SummarizationCancelledError):
        _ = summarizer.summarize(
            "a short document", SUMMARY_TYPE_BRIEF, should_cancel=lambda: stop["now"]
        )


def test_a_failure_without_stop_is_raised_as_itself(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _ = _install_llama_cpp(monkeypatch)
    summarizer = Summarizer(tmp_path / "model.gguf")
    _llm(summarizer).fail_with = RuntimeError("llama_decode returned 1")

    with pytest.raises(RuntimeError, match="llama_decode returned 1"):
        _ = summarizer.summarize(
            "a short document", SUMMARY_TYPE_BRIEF, should_cancel=lambda: False
        )


def test_a_closed_summarizer_refuses_to_summarize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _ = _install_llama_cpp(monkeypatch)
    summarizer = Summarizer(tmp_path / "model.gguf")
    summarizer.close()

    with pytest.raises(RuntimeError):
        _ = summarizer.summarize("a short document", SUMMARY_TYPE_BRIEF)
    with pytest.raises(RuntimeError):
        _ = summarizer._count_tokens("text")


class _BrokenProperty:
    """An object whose ``ctx`` attribute raises, as some bindings' properties do."""

    @property
    def ctx(self) -> NoReturn:
        msg = "no context on this build"
        raise RuntimeError(msg)


def _with_llm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, llm: object) -> Summarizer:
    _ = _install_llama_cpp(monkeypatch)
    summarizer = Summarizer(tmp_path / "model.gguf")
    summarizer.llm = cast("model_manager.ChatModel", llm)
    return summarizer


def test_the_context_falls_back_to_the_internal_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inner = object()
    llm = types.SimpleNamespace(ctx=None, _ctx=types.SimpleNamespace(ctx=inner))

    assert _with_llm(monkeypatch, tmp_path, llm)._llama_context() is inner


def test_a_context_property_that_raises_reads_as_no_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    llm = types.SimpleNamespace(_ctx=_BrokenProperty())
    summarizer = _with_llm(monkeypatch, tmp_path, llm)

    assert summarizer._llama_context() is None
    summarizer.llm = cast("model_manager.ChatModel", cast("object", _BrokenProperty()))
    assert summarizer._llama_context() is None


def test_perf_counters_are_none_without_the_binding_or_when_it_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    low = _install_llama_cpp(monkeypatch)
    summarizer = Summarizer(tmp_path / "model.gguf")

    assert summarizer.perf_snapshot() is None  # this build has no llama_perf_context

    def broken(_ctx: object) -> NoReturn:
        msg = "perf counters are off"
        raise RuntimeError(msg)

    monkeypatch.setattr(low, "llama_perf_context", broken, raising=False)
    assert summarizer.perf_snapshot() is None


def test_download_reports_progress_and_returns_the_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_download(*, repo_id: str, filename: str, local_dir: Path, tqdm_class: object) -> str:
        del repo_id
        assert tqdm_class is not None
        target = local_dir / filename
        _ = target.write_bytes(b"gguf")
        return str(target)

    monkeypatch.setattr(model_manager, "hf_hub_download", fake_download)
    progress: list[tuple[float, str]] = []

    path, error = model_manager.download_model(
        progress_callback=lambda pct, msg: progress.append((pct, msg))
    )

    assert error is None
    assert path.read_bytes() == b"gguf"
    assert progress[0][0] == 0.0
    assert progress[-1] == (100.0, "Download complete")


def test_a_failed_download_returns_its_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_download(**_options: object) -> NoReturn:
        msg = "connection reset"
        raise OSError(msg)

    monkeypatch.setattr(model_manager, "hf_hub_download", fake_download)
    progress: list[tuple[float, str]] = []

    path, error = model_manager.download_model(
        progress_callback=lambda pct, msg: progress.append((pct, msg))
    )

    assert error == "Failed to download model: connection reset"
    assert path == model_manager.get_model_path()
    assert not path.exists()
    assert progress[-1] == (0.0, error)
