import QtQuick
import QtQuick.Layouts
import App

// Persistent top bar: hex logo + identity + breathing secure badge + live
// model/compute readout + theme toggle. Model strings come from the bridge.
Rectangle {
    id: bar
    height: 66
    color: "transparent"

    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: 22
        anchors.rightMargin: 22
        spacing: 16

        // Hex logo mark (outer hex + accent inner hex).
        Item {
            Layout.preferredWidth: 30
            Layout.preferredHeight: 30
            Hex {
                anchors.fill: parent
                fill: Theme.hexBg
                stroke: Theme.ring
            }
            Hex {
                anchors.centerIn: parent
                size: 9
                fill: Theme.accent
                stroke: "transparent"
            }
        }

        ColumnLayout {
            spacing: 0
            Text {
                text: "DocSummarizer"
                color: Theme.ink
                font.family: Theme.serif
                font.pixelSize: 17
                font.weight: Font.DemiBold
            }
            Text {
                text: "ABSTRACT CONSOLE · v2.0.1"
                color: Theme.faint
                font.family: Theme.mono
                font.pixelSize: 9
                font.letterSpacing: 2
            }
        }

        Rectangle {
            Layout.preferredWidth: badge.implicitWidth + 28
            Layout.preferredHeight: 26
            radius: 2
            color: Qt.rgba(0.78, 0.66, 0.42, 0.05)
            border.width: 1
            border.color: Theme.brassRing
            Row {
                id: badge
                anchors.centerIn: parent
                spacing: 8
                Rectangle {
                    width: 6
                    height: 6
                    radius: 3
                    anchors.verticalCenter: parent.verticalCenter
                    color: Theme.brassDot
                    // Breathing pulse.
                    SequentialAnimation on opacity {
                        running: true
                        loops: Animation.Infinite
                        NumberAnimation {
                            to: 0.5
                            duration: 1600
                            easing.type: Easing.InOutSine
                        }
                        NumberAnimation {
                            to: 0.95
                            duration: 1600
                            easing.type: Easing.InOutSine
                        }
                    }
                }
                Text {
                    text: "SECURE · AIRGAPPED · LOCAL CORE"
                    color: Theme.brass
                    font.family: Theme.ui
                    font.pixelSize: 10
                    font.weight: Font.Medium
                    font.letterSpacing: 1.5
                }
            }
        }

        Item {
            Layout.fillWidth: true
        }

        // Right-side readout, per-segment coloring.
        Row {
            spacing: 0
            Layout.alignment: Qt.AlignVCenter
            Text {
                text: bridge.modelName.toUpperCase()
                color: Theme.faint
                font.family: Theme.mono
                font.pixelSize: 10
                font.letterSpacing: 1
            }
            Text {
                text: "   |   "
                color: Theme.dim
                font.family: Theme.mono
                font.pixelSize: 10
            }
            Text {
                text: bridge.computeLabel
                color: Theme.accent
                font.family: Theme.mono
                font.pixelSize: 10
                font.letterSpacing: 1
            }
            Text {
                text: "   |   "
                color: Theme.dim
                font.family: Theme.mono
                font.pixelSize: 10
            }
            Text {
                text: "ONLINE·LOCAL"
                color: Theme.accent
                font.family: Theme.mono
                font.pixelSize: 10
                font.letterSpacing: 1
            }
        }

        Rectangle {
            Layout.preferredWidth: 30
            Layout.preferredHeight: 30
            radius: 3
            color: "transparent"
            border.width: 1
            border.color: Theme.line2
            Text {
                anchors.centerIn: parent
                text: Theme.dark ? "☾" : "☀"
                color: Theme.label2
                font.pixelSize: 13
            }
            MouseArea {
                anchors.fill: parent
                cursorShape: Qt.PointingHandCursor
                onClicked: {
                    Theme.dark = !Theme.dark;
                    bridge.setAppearance(Theme.dark ? "Dark" : "Light");
                }
            }
        }
    }

    Rectangle {
        anchors.bottom: parent.bottom
        width: parent.width
        height: 1
        color: Theme.line2
    }
}
