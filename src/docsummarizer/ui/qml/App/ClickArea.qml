import QtQuick
import App

// A click target that is also a keyboard control: Tab reaches it, Space or
// Enter activates it, a screen reader announces it by `name`, and a ring shows
// while it has focus. Handle `activated`, not `clicked`, so every way in runs
// the same action.
MouseArea {
    id: area
    // What a screen reader announces for this control.
    required property string name
    signal activated

    cursorShape: enabled ? Qt.PointingHandCursor : Qt.ArrowCursor
    activeFocusOnTab: true
    onClicked: area.activated()
    Keys.onSpacePressed: area.activated()
    Keys.onReturnPressed: area.activated()
    Keys.onEnterPressed: area.activated()
    Accessible.role: Accessible.Button
    Accessible.name: area.name
    Accessible.focusable: true
    Accessible.onPressAction: area.activated()

    Rectangle {
        anchors.fill: parent
        anchors.margins: -2
        visible: area.activeFocus
        color: "transparent"
        radius: 3
        border.width: 1
        border.color: Theme.accent
    }
}
