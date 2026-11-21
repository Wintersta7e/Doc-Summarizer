# SPDX-License-Identifier: MIT
"""Each screen in each theme still looks the way its committed baseline does.

Every screen renders through ``scripts/render_qml.py``'s path (offscreen, the
software backend, the bundled fonts, 1240x820) and is compared with
``tests/visual/baselines/linux/<screen>-<theme>.png``. A pixel differs when any
channel is off by more than 8; a screen fails when more than 0.1 % of its
pixels differ. On failure the render and a diff image (differing pixels in red)
are written to the test's temporary directory and named in the message: a
render that is the new intended look is copied over its baseline.

The baselines were rendered with PySide6 6.12.0 on Linux; another Qt version or
platform rasterises text differently, so only Linux compares pixels. The two
facts a render reads from the machine, the CPU count and GPU support, are
pinned so the top bar and the Config screen read the same everywhere.
"""

from __future__ import annotations

import itertools
import logging
import os
import sys
from pathlib import Path

import PySide6
import pytest
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter

import docsummarizer.ui.bridge as bridge_module
from render_qml import SCREENS, THEMES, render

_LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="baselines are Linux renders; the Windows leg covers behaviour, not pixels",
)

_BASELINES = Path(__file__).parent / "visual" / "baselines" / "linux"
# The Qt binding the baselines were rendered with.
_BASELINE_PYSIDE6 = "6.12.0"
# A pixel differs when a channel is off by more than this, out of 255.
_CHANNEL_TOLERANCE = 8
# A screen fails when more than this share of its pixels differ.
_MAX_DIFFERING_SHARE = 0.001
# The machine every baseline shows: 8 CPUs (so 4 threads), no GPU offload.
_CPUS = 8
_FORMAT = QImage.Format.Format_RGBA8888
_CHANNEL_SHIFTS = (0, 8, 16, 24)
_CHANNEL_MASK = 0xFF
# Laid over a failing render so the differing pixels stand out.
_DIM = QColor(0, 0, 0, 192)


@pytest.fixture(scope="session")
def qapp() -> QGuiApplication:
    app = QGuiApplication.instance() or QGuiApplication([])
    assert isinstance(app, QGuiApplication)
    return app


@pytest.fixture
def same_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the CPU count and GPU support the screens show."""
    monkeypatch.setattr(os, "cpu_count", lambda: _CPUS)
    monkeypatch.setattr(bridge_module, "gpu_offload_supported", lambda: False)


def _pixel_differs(first: int, second: int) -> bool:
    return any(
        abs(((first >> shift) & _CHANNEL_MASK) - ((second >> shift) & _CHANNEL_MASK))
        > _CHANNEL_TOLERANCE
        for shift in _CHANNEL_SHIFTS
    )


def differing_pixels(actual: QImage, expected: QImage) -> list[int]:
    """The indexes, row by row, of the pixels where ``actual`` and ``expected`` differ.

    Returns:
        Every index when the sizes differ; none when the images match.
    """
    actual = actual.convertToFormat(_FORMAT)
    expected = expected.convertToFormat(_FORMAT)
    if actual.size() != expected.size():
        return list(range(actual.width() * actual.height()))
    # RGBA8888 rows are whole 32-bit words, so the buffer is one word a pixel.
    first, second = bytes(actual.constBits()), bytes(expected.constBits())
    if first == second:
        return []
    pairs = zip(memoryview(first).cast("I"), memoryview(second).cast("I"), strict=True)
    return [
        index
        for index, (one, other) in enumerate(pairs)
        if one != other and _pixel_differs(one, other)
    ]


def _diff_image(actual: QImage, differing: list[int]) -> QImage:
    """``actual`` dimmed, with each differing pixel in red."""
    diff = actual.convertToFormat(_FORMAT)
    painter = QPainter(diff)
    painter.fillRect(diff.rect(), _DIM)
    _ = painter.end()
    red = QColor("#ff0000")
    width = diff.width()
    for index in differing:
        diff.setPixelColor(index % width, index // width, red)
    return diff


def test_differing_pixels_counts_only_changes_past_the_tolerance() -> None:
    expected = QImage(4, 2, _FORMAT)
    expected.fill(QColor(100, 100, 100))
    actual = expected.copy()
    actual.setPixelColor(1, 0, QColor(100 + _CHANNEL_TOLERANCE, 100, 100))
    actual.setPixelColor(2, 1, QColor(100, 100, 101 + _CHANNEL_TOLERANCE))

    assert differing_pixels(expected, expected.copy()) == []
    assert differing_pixels(actual, expected) == [6]
    assert len(differing_pixels(actual.scaled(2, 2), expected)) == 2 * 2


@pytest.mark.usefixtures("qapp", "same_machine")
@pytest.mark.parametrize(("screen", "theme"), list(itertools.product(SCREENS, THEMES)))
def test_screen_matches_its_baseline(screen: str, theme: str, tmp_path: Path) -> None:
    name = f"{screen}-{theme}.png"
    image = render(screen, theme)
    baseline = QImage(str(_BASELINES / name))
    differing = differing_pixels(image, baseline)
    share = len(differing) / (image.width() * image.height())
    _LOG.info("%s: %s pixels differ (%.4f %%)", name, len(differing), share * 100)
    if share <= _MAX_DIFFERING_SHARE:
        return

    rendered, diff = tmp_path / name, tmp_path / f"{screen}-{theme}.diff.png"
    assert image.save(str(rendered))
    assert _diff_image(image, differing).save(str(diff))
    pytest.fail(
        f"{name}: {share:.2%} of pixels differ (limit {_MAX_DIFFERING_SHARE:.1%}); "
        f"render {rendered}, diff {diff}; baselines are PySide6 {_BASELINE_PYSIDE6}, "
        f"this run {PySide6.__version__}"
    )
