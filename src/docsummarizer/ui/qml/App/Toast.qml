import QtQuick
import App

// Bottom-center transient toast. Call show(message); it fades out after ~2.2s.
Rectangle {
    id: toast
    property string message: ""

    anchors.horizontalCenter: parent.horizontalCenter
    anchors.bottom: parent.bottom
    anchors.bottomMargin: 28
    width: row.implicitWidth + 36
    height: 40
    radius: 3
    color: Theme.shellTop
    border.width: 1
    border.color: Theme.line2
    opacity: 0
    visible: opacity > 0
    z: 100

    Behavior on opacity {
        NumberAnimation {
            duration: 300
        }
    }

    Row {
        id: row
        anchors.centerIn: parent
        spacing: 10
        Rectangle {
            width: 7
            height: 7
            radius: 3.5
            anchors.verticalCenter: parent.verticalCenter
            color: Theme.accent
        }
        Text {
            text: toast.message
            textFormat: Text.PlainText
            color: Theme.ink
            font.family: Theme.body
            font.pixelSize: 13
        }
    }

    Timer {
        id: hideTimer
        interval: 2200
        onTriggered: toast.opacity = 0
    }

    function show(msg) {
        toast.message = msg;
        toast.opacity = 1;
        hideTimer.restart();
    }
}
