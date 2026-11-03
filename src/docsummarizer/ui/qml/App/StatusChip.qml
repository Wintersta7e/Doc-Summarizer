import QtQuick
import App

// Batch status pill: DONE (accent) / PROCESSING (brass, pulsing) / QUEUED (dim) /
// FAILED (error) / SKIPPED (dim). Optional trailing meta (e.g. token count).
Rectangle {
    id: chip
    property string status: "QUEUED"
    property string meta: ""

    readonly property color hue: status === "DONE" ? Theme.accent : status === "PROCESSING" ? Theme.brass : status === "FAILED" ? Theme.statusError : Theme.dim

    implicitWidth: row.implicitWidth + 22
    implicitHeight: 22
    radius: 2
    color: "transparent"
    border.width: 1
    border.color: Qt.rgba(hue.r, hue.g, hue.b, 0.35)

    Row {
        id: row
        anchors.centerIn: parent
        spacing: 8
        Text {
            anchors.verticalCenter: parent.verticalCenter
            text: chip.status === "PROCESSING" ? "SUMMARIZING…" : chip.status
            color: chip.hue
            font.family: Theme.mono
            font.pixelSize: 9
            font.letterSpacing: 1
            SequentialAnimation on opacity {
                running: chip.status === "PROCESSING"
                loops: Animation.Infinite
                NumberAnimation {
                    to: 0.4
                    duration: 700
                }
                NumberAnimation {
                    to: 1.0
                    duration: 700
                }
            }
        }
        Text {
            anchors.verticalCenter: parent.verticalCenter
            visible: chip.meta !== ""
            text: chip.meta
            color: Theme.dim
            font.family: Theme.mono
            font.pixelSize: 9
        }
    }
}
