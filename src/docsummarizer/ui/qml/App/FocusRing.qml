import QtQuick
import App

// The keyboard focus indicator: a 2 px ring just inside the control it fills,
// so no neighbour drawn over the control and no clipping parent hides it. It
// starts 1 px in, clear of the control's own border, and on an accent fill it
// draws in the ink made for that fill.
Rectangle {
    // Whether the control under the ring is filled with the accent.
    property bool onAccent: false

    anchors.fill: parent
    anchors.margins: 1
    color: "transparent"
    radius: 2
    border.width: 2
    border.color: onAccent ? Theme.inkOnAccent : Theme.accent
}
