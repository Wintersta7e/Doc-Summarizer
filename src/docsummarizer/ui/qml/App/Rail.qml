pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Layouts
import App

// LCARS-style left navigation rail. Emits navigate(name) on click.
Rectangle {
    id: rail
    width: 120
    color: "transparent"

    property string currentScreen: "summary"
    signal navigate(string name)

    readonly property var items: [
        {
            "name": "summary",
            "label": "Summary"
        },
        {
            "name": "extract",
            "label": "Extract"
        },
        {
            "name": "batch",
            "label": "Batch"
        },
        {
            "name": "config",
            "label": "Config"
        }
    ]

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 8

        // LCARS elbow accent block — the 20px bottom-left curve is the signature.
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 42
            topLeftRadius: 4
            topRightRadius: 4
            bottomRightRadius: 4
            bottomLeftRadius: 20
            gradient: Gradient {
                GradientStop {
                    position: 0.0
                    color: Theme.accent
                }
                GradientStop {
                    position: 1.0
                    color: Theme.accentDeep
                }
            }
            Text {
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                anchors.rightMargin: 9
                anchors.bottomMargin: 6
                text: "01"
                color: Theme.onAccent
                font.family: Theme.mono
                font.pixelSize: 9
                font.letterSpacing: 1
            }
        }

        Repeater {
            model: rail.items
            delegate: Rectangle {
                id: navItem
                required property var modelData
                Layout.fillWidth: true
                Layout.preferredHeight: 40
                radius: 4
                color: navItem.modelData.name === rail.currentScreen ? Theme.navOnBg : Theme.block
                border.width: 1
                border.color: navItem.modelData.name === rail.currentScreen ? Theme.navOnRing : "transparent"
                Text {
                    anchors.left: parent.left
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.leftMargin: 12
                    text: navItem.modelData.label
                    color: navItem.modelData.name === rail.currentScreen ? Theme.label2 : Theme.navOff
                    font.family: Theme.ui
                    font.pixelSize: 11
                    font.weight: Font.Medium
                    font.letterSpacing: 0.4
                }
                MouseArea {
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked: rail.navigate(navItem.modelData.name)
                }
            }
        }

        Item {
            Layout.fillHeight: true
        }

        // Decorative LCARS foot block.
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 34
            topLeftRadius: 18
            topRightRadius: 4
            bottomRightRadius: 4
            bottomLeftRadius: 4
            opacity: 0.7
            gradient: Gradient {
                GradientStop {
                    position: 0.0
                    color: Qt.rgba(0.23, 0.26, 0.31, 1.0)
                }
                GradientStop {
                    position: 1.0
                    color: Qt.rgba(0.17, 0.19, 0.24, 1.0)
                }
            }
            Rectangle {
                anchors.fill: parent
                color: "transparent"
                topLeftRadius: 18
                topRightRadius: 4
                bottomRightRadius: 4
                bottomLeftRadius: 4
                border.width: 1
                border.color: Theme.brassRing
            }
        }
    }
}
