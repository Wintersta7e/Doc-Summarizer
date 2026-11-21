pragma ComponentBehavior: Bound
import QtQuick
import App

// A segmented control (Brief / Detailed / Structured). Active segment fills
// with the accent; emits selected(value).
Row {
    id: seg
    property var options: []
    property string current: ""
    signal selected(string value)
    spacing: 0

    Repeater {
        model: seg.options
        delegate: Rectangle {
            id: segment
            required property var modelData
            width: 96
            height: 38
            radius: 2
            color: segment.modelData === seg.current ? Theme.accent : "transparent"
            border.width: 1
            border.color: Theme.line2
            Text {
                anchors.centerIn: parent
                text: String(segment.modelData).charAt(0).toUpperCase() + String(segment.modelData).slice(1)
                color: segment.modelData === seg.current ? Theme.onAccent : Theme.navOff
                font.family: Theme.ui
                font.pixelSize: 11
                font.letterSpacing: 0.6
            }
            MouseArea {
                anchors.fill: parent
                cursorShape: Qt.PointingHandCursor
                onClicked: seg.selected(segment.modelData)
            }
        }
    }
}
