# SPDX-License-Identifier: MIT
"""The hot paths stay linear: ten times the input costs at most twenty times the time.

Each path is timed on an input of n sentences (a ten-page report) and of 10 n,
best of five runs. A linear path takes about ten times as long; a
quadratic one takes a hundred times, so the bound of twenty catches a quadratic
blow-up on any machine whatever its speed. One timing repeats the call until the
n-sentence input takes at least 20 ms, so the timer's resolution does not decide
the ratio, and the larger input repeats it as often. The fast paths need no
input of hundreds of megabytes to reach 20 ms that way. Only the ratio is
asserted, never a wall time.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import pytest

from docsummarizer.document_parser import extract_from_docx, extract_from_txt
from docsummarizer.model_manager import _split_into_chunks, _split_into_chunks_with_offsets
from docsummarizer.provenance import locate_quote

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_LOG = logging.getLogger(__name__)
_GROWTH = 10
_MAX_SLOWDOWN = 20
_MIN_SECONDS = 0.020
_REPEATS = 5
# Big enough that a quadratic path shows; small enough that one still fails
# inside the test timeout instead of running for minutes.
_SENTENCES = 500
# Doublings of the calls per timing before a path counts as too fast to time.
_MAX_DOUBLINGS = 20
_CHUNK_CHARS = 4_000
_SENTENCES_PER_PARAGRAPH = 5


def _sentence(index: int) -> str:
    return f"Pilot {index} logs crossing {index * 7 % 101} by the light of lantern {index % 13}."


def _paragraphs(count: int) -> list[str]:
    """``count`` distinct sentences, grouped into paragraphs."""
    step = _SENTENCES_PER_PARAGRAPH
    return [
        " ".join(_sentence(index) for index in range(start, min(start + step, count)))
        for start in range(0, count, step)
    ]


def _text(count: int) -> str:
    return "\n\n".join(_paragraphs(count))


def _chunker(count: int, _folder: Path) -> Callable[[], object]:
    text = _text(count)
    return lambda: _split_into_chunks(text, _CHUNK_CHARS)


def _offset_chunker(count: int, _folder: Path) -> Callable[[], object]:
    text = _text(count)
    return lambda: _split_into_chunks_with_offsets(text, _CHUNK_CHARS)


def _quote_locator(count: int, _folder: Path) -> Callable[[], object]:
    text = _text(count)
    # The last sentence: every sentence before it is ranked first.
    quote = _sentence(count - 1)
    return lambda: locate_quote(quote, text)


def _txt_parser(count: int, folder: Path) -> Callable[[], object]:
    path = folder / f"{count}.txt"
    _ = path.write_text(_text(count), encoding="utf-8")
    return lambda: extract_from_txt(str(path))


def _docx_parser(count: int, folder: Path) -> Callable[[], object]:
    from docx import Document  # the parser imports it lazily as well

    path = folder / f"{count}.docx"
    document = Document()
    for paragraph in _paragraphs(count):
        _ = document.add_paragraph(paragraph)
    document.save(str(path))
    return lambda: extract_from_docx(str(path))


def _timed(run: Callable[[], object], calls: int) -> float:
    """Seconds ``calls`` runs take."""
    start = time.perf_counter()
    for _ in range(calls):
        _ = run()
    return time.perf_counter() - start


def _calls_to_time(run: Callable[[], object]) -> int:
    """The fewest calls, doubling from one, that take at least 20 ms.

    Returns:
        The number of calls per timing.
    """
    calls = 1
    for _ in range(_MAX_DOUBLINGS):
        if _timed(run, calls) >= _MIN_SECONDS:
            return calls
        calls *= 2
    pytest.fail(f"{calls} calls still take under 20 ms")


def _best_time(run: Callable[[], object], calls: int) -> float:
    """The fastest of five timings of ``calls`` runs, in seconds."""
    return min(_timed(run, calls) for _ in range(_REPEATS))


_PATHS = {
    "chunker": _chunker,
    "offset-chunker": _offset_chunker,
    "locate-quote": _quote_locator,
    "txt-parser": _txt_parser,
    "docx-parser": _docx_parser,
}


@pytest.mark.parametrize("name", _PATHS)
def test_ten_times_the_input_costs_at_most_twenty_times_the_time(name: str, tmp_path: Path) -> None:
    prepare = _PATHS[name]
    small_run = prepare(_SENTENCES, tmp_path)
    calls = _calls_to_time(small_run)
    small = _best_time(small_run, calls)

    large = _best_time(prepare(_SENTENCES * _GROWTH, tmp_path), calls)

    _LOG.info(
        "%s: %s calls, %.1f ms, then %.1f ms, ratio %.1f",
        name,
        calls,
        small * 1000,
        large * 1000,
        large / small,
    )
    assert large <= _MAX_SLOWDOWN * small
