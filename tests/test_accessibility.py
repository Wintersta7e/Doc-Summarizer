# SPDX-License-Identifier: MIT
"""Every control is named, reachable by Tab, and readable in each theme.

Each screen loads in each theme the way ``scripts/render_qml.py`` renders it,
with accessibility switched on so the controls publish what a screen reader
reads. A control is a button, text field, combo box, slider or spin box, or a
``MouseArea`` with a click handler. Each one visible has a name and a role;
each one also enabled is reached by pressing Tab from the first item that
takes focus. The colour check reads the ``Theme`` singleton and holds each
text colour to WCAG AA (4.5:1) on every surface the screens draw, and each
control part (the focus ring) to 3:1.

``screen_view`` is the fixture keyboard tests build on: one loaded screen per
screen and theme, torn down after the test.
"""

from __future__ import annotations

import contextlib
import gc
import itertools
import logging
from typing import TYPE_CHECKING, cast

import pytest
from PySide6.QtCore import SIGNAL, QEvent, QEventLoop, QObject, Qt, QTimer
from PySide6.QtGui import QAccessible, QColor, QGuiApplication, QKeyEvent

from render_qml import SCREENS, THEMES, load_preview

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

    from PySide6.QtGui import QAccessibleInterface
    from PySide6.QtQuick import QQuickItem, QQuickView

_LOG = logging.getLogger(__name__)
# How long a screen runs before it is inspected: its first bindings and
# timers have fired.
_SETTLE_MS = 200
# A Tab walk longer than this has left the screen's focus cycle.
_MAX_TABS = 200
_CONTROL_CLASSES = frozenset(
    {"QQuickAbstractButton", "QQuickTextField", "QQuickComboBox", "QQuickSlider", "QQuickSpinBox"}
)
_CLICKED = SIGNAL("clicked(QQuickMouseEvent*)")
# WCAG 2.x AA: normal text, and the parts of a control (1.4.3 and 1.4.11).
_AA_TEXT = 4.5
_AA_CONTROL_PART = 3.0
# Theme tokens the screens use as text colours.
_TEXT_TOKENS = (
    "ink",
    "inkSoft",
    "text",
    "label",
    "label2",
    "faint",
    "dim",
    "accent",
    "accent2",
    "brass",
    "statusError",
    "navOff",
)
# Opaque surfaces text sits on, and translucent ones laid over them.
_SURFACES = ("pageBg", "pageTop", "pageBottom", "shellTop", "shellBottom", "block", "hexBg")
_TINTS = ("kpRest", "srcPane", "navOnBg")
# Text on an accent fill: the primary button and the selected segment.
_ON_ACCENT = (("inkOnAccent", "accent"), ("inkOnAccent", "accentDeep"))
# Control parts drawn in a token: the keyboard focus ring.
_CONTROL_PART_TOKENS = ("accent",)


@pytest.fixture(scope="session")
def qapp() -> QGuiApplication:
    app = QGuiApplication.instance() or QGuiApplication([])
    assert isinstance(app, QGuiApplication)
    return app


def _spin(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    _ = loop.exec()


_SCREEN_THEMES = list(itertools.product(SCREENS, THEMES))


@contextlib.contextmanager
def _loaded(screen: str, theme: str) -> Generator[QQuickView, None, None]:
    """``screen`` shown in ``theme`` with accessibility on, closed and freed on exit.

    Yields:
        The shown view.
    """
    # Controls publish their accessible names only while accessibility is on.
    QAccessible.setActive(True)
    view = load_preview(screen, theme)
    try:
        _spin(_SETTLE_MS)
        yield view
    finally:
        _ = view.close()
        del view
        _ = gc.collect()
        _spin(0)
        QAccessible.setActive(False)


@pytest.fixture(params=_SCREEN_THEMES, ids=["-".join(pair) for pair in _SCREEN_THEMES])
def screen_view(request: pytest.FixtureRequest, qapp: QGuiApplication) -> Iterator[QQuickView]:
    """One screen in one theme, loaded with accessibility on.

    Yields:
        The shown view; it is closed and freed after the test.
    """
    assert qapp is QGuiApplication.instance()
    with _loaded(*cast("tuple[str, str]", request.param)) as view:
        yield view


def _controls_under(item: QQuickItem) -> Iterator[QQuickItem]:
    """The controls in ``item``'s tree, not looking inside a Qt Quick control.

    A control's own parts (the text field inside a combo box) are announced and
    focused as the control itself, so they are not separate controls.

    Yields:
        Each control, parents before children.
    """
    if any(item.inherits(name) for name in _CONTROL_CLASSES):
        yield item
        return
    if item.inherits("QQuickMouseArea") and item.receivers(_CLICKED) > 0:
        yield item
    for child in item.childItems():
        yield from _controls_under(child)


def _controls(view: QQuickView) -> list[QQuickItem]:
    """The visible controls on the view."""
    return [item for item in _controls_under(view.rootObject()) if item.isVisible()]


def _describe(item: QQuickItem) -> str:
    owner = item.parentItem().metaObject().className()
    return f"{item.metaObject().className()} in {owner} '{_name(item)}'"


def _accessible(item: QQuickItem) -> QAccessibleInterface | None:
    # The stubs promise an interface; an item with no accessibility gets none.
    return cast("QAccessibleInterface | None", QAccessible.queryAccessibleInterface(item))


def _name(item: QQuickItem) -> str:
    accessible = _accessible(item)
    return "" if accessible is None else accessible.text(QAccessible.Text.Name).strip()


def _role(item: QQuickItem) -> QAccessible.Role:
    accessible = _accessible(item)
    return QAccessible.Role.NoRole if accessible is None else accessible.role()


def press(view: QQuickView, key: Qt.Key) -> None:
    """Press and release ``key`` on ``view``, as the keyboard does."""
    for kind in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
        _ = QGuiApplication.sendEvent(view, QKeyEvent(kind, key, Qt.KeyboardModifier.NoModifier))


def tab_order(view: QQuickView) -> list[QQuickItem]:
    """The items Tab visits, from the first that takes focus round to it again.

    Returns:
        Each focused item once, in Tab order; empty when nothing takes focus.
    """
    root = view.rootObject()
    first = root.nextItemInFocusChain(forward=True)
    if first == root:
        return []
    first.forceActiveFocus(Qt.FocusReason.TabFocusReason)
    order: list[QQuickItem] = []
    for _ in range(_MAX_TABS):
        # The stubs promise an item; a window with nothing focused has none.
        focused = cast("QQuickItem | None", view.activeFocusItem())
        if focused is None or focused in order:
            break
        order.append(focused)
        press(view, Qt.Key.Key_Tab)
    return order


def test_every_control_has_a_name_and_a_role(screen_view: QQuickView) -> None:
    controls = _controls(screen_view)
    unnamed = [
        _describe(item)
        for item in controls
        if not _name(item) or _role(item) == QAccessible.Role.NoRole
    ]

    _LOG.info("%s controls, unnamed: %s", len(controls), unnamed)
    assert controls
    assert unnamed == []


def test_tab_reaches_every_enabled_control(screen_view: QQuickView) -> None:
    order = tab_order(screen_view)
    unreachable = [
        _describe(item) for item in _controls(screen_view) if item.isEnabled() and item not in order
    ]

    _LOG.info("Tab order: %s", [_name(item) for item in order])
    assert unreachable == []


@pytest.mark.usefixtures("qapp")
@pytest.mark.parametrize("key", [Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter])
def test_a_focused_click_target_runs_its_action_from_the_keyboard(key: Qt.Key) -> None:
    with _loaded("summary", "dark") as view:
        theme = _theme(view)
        (toggle,) = [item for item in _controls(view) if _name(item) == "Switch to the light theme"]
        toggle.forceActiveFocus(Qt.FocusReason.TabFocusReason)

        press(view, key)

        assert theme.property("dark") is False
        assert _name(toggle) == "Switch to the dark theme"


def _theme(view: QQuickView) -> QObject:
    theme = view.engine().singletonInstance("App", "Theme")
    assert isinstance(theme, QObject)
    return theme


def _color(theme: QObject, token: str) -> QColor:
    value = cast("object", theme.property(token))
    # A token whose binding never ran reads as an invalid colour, drawn black:
    # QML takes a property named `on` + capital for a signal handler.
    assert isinstance(value, QColor), token
    assert value.isValid(), token
    return value


def _over(tint: QColor, base: QColor) -> QColor:
    """``tint`` composited over the opaque ``base``."""
    alpha = tint.alphaF()
    return QColor.fromRgbF(
        tint.redF() * alpha + base.redF() * (1 - alpha),
        tint.greenF() * alpha + base.greenF() * (1 - alpha),
        tint.blueF() * alpha + base.blueF() * (1 - alpha),
    )


def _luminance(color: QColor) -> float:
    def linear(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    red, green, blue = (linear(c) for c in (color.redF(), color.greenF(), color.blueF()))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast(first: QColor, second: QColor) -> float:
    """The WCAG contrast ratio of two opaque colours, 1 to 21.

    Returns:
        The lighter colour's relative luminance plus 0.05 over the darker's.
    """
    light, dark = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def test_contrast_of_two_colours_follows_wcag() -> None:
    black, white = QColor("#000000"), QColor("#ffffff")

    assert contrast(black, white) == pytest.approx(21)
    assert contrast(white, black) == pytest.approx(21)
    assert contrast(QColor("#777777"), white) == pytest.approx(4.48, abs=0.01)
    # QColor keeps 16 bits a channel.
    assert _over(QColor.fromRgbF(1, 1, 1, 0.5), black).redF() == pytest.approx(0.5, abs=1e-4)


def _surfaces(theme: QObject) -> dict[str, QColor]:
    surfaces = {token: _color(theme, token) for token in _SURFACES}
    for tint, base in itertools.product(_TINTS, _SURFACES):
        surfaces[f"{tint} over {base}"] = _over(_color(theme, tint), surfaces[base])
    return surfaces


@pytest.mark.usefixtures("qapp")
@pytest.mark.parametrize("theme_name", THEMES)
def test_theme_colours_meet_wcag_aa(theme_name: str) -> None:
    with _loaded("summary", theme_name) as view:
        theme = _theme(view)
        assert theme.property("dark") is (theme_name == "dark")
        surfaces = _surfaces(theme)
        pairs = [
            (token, surface, _AA_TEXT)
            for token, surface in itertools.product(_TEXT_TOKENS, surfaces)
        ]
        pairs += [
            (token, surface, _AA_CONTROL_PART)
            for token, surface in itertools.product(_CONTROL_PART_TOKENS, surfaces)
        ]
        failing = [
            f"{token} on {surface}: {ratio:.2f} < {minimum}"
            for token, surface, minimum in pairs
            if (ratio := contrast(_color(theme, token), surfaces[surface])) < minimum
        ]
        failing += [
            f"{token} on {fill}: {ratio:.2f} < {_AA_TEXT}"
            for token, fill in _ON_ACCENT
            if (ratio := contrast(_color(theme, token), _color(theme, fill))) < _AA_TEXT
        ]

    assert failing == []
