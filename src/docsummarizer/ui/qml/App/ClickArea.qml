import QtQuick
import App

// A click target that is also a keyboard control: Tab reaches it, Space or
// Enter activates it, a screen reader announces it by `name`, and a ring shows
// while it has focus. Handle `activated`, not `clicked`, so every way in runs
// the same action. A MouseArea's `enabled` only stops the mouse, so the
// keyboard and screen-reader paths check it too.
MouseArea {
    id: area
    // What a screen reader announces for this control.
    required property string name
    // Whether the target is drawn filled with the accent, for its focus ring.
    property bool onAccent: false
    signal activated

    function activate(): void {
        if (area.enabled)
            area.activated();
    }

    cursorShape: enabled ? Qt.PointingHandCursor : Qt.ArrowCursor
    activeFocusOnTab: area.enabled
    onClicked: area.activated()
    Keys.onSpacePressed: area.activate()
    Keys.onReturnPressed: area.activate()
    Keys.onEnterPressed: area.activate()
    Accessible.role: Accessible.Button
    Accessible.name: area.name
    Accessible.focusable: area.enabled
    Accessible.onPressAction: area.activate()

    FocusRing {
        visible: area.activeFocus
        onAccent: area.onAccent
    }
}
