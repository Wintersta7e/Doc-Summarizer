pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Layouts
import QtQuick.Controls.Basic
import App

// First-run overlay: a modal scrim over the whole shell with a centered panel
// that walks the user through the one-time local model download. The primary
// button cycles Begin Download → Downloading… → Enter Console, then emits
// enter() to dismiss into the console. Progress + status are driven by the
// bridge's progress(pct, msg) signal.
Item {
    id: screen

    // The ConsoleBridge, passed down from Main.qml.
    required property var bridge

    signal enter

    property bool shown: true
    property bool started: false
    property string statusLine: "Awaiting initialization."

    visible: screen.shown && opacity > 0
    opacity: screen.shown ? 1 : 0
    Behavior on opacity {
        NumberAnimation {
            duration: 180
        }
    }

    Connections {
        target: screen.bridge
        function onProgress(pct, msg) {
            screen.statusLine = msg;
        }
    }

    // Full-bleed dimming scrim. Swallows clicks so the console behind stays
    // inert while the overlay is up.
    Rectangle {
        anchors.fill: parent
        color: Theme.overlay
        MouseArea {
            anchors.fill: parent
            hoverEnabled: true
            preventStealing: true
        }
    }

    // Centered panel ----------------------------------------------------- //
    Rectangle {
        id: panel
        anchors.centerIn: parent
        width: 520
        implicitHeight: panelCol.implicitHeight + 56
        radius: 6
        color: Theme.shellTop
        border.width: 1
        border.color: Theme.line2

        ColumnLayout {
            id: panelCol
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.margins: 28
            spacing: 16

            Text {
                text: "INITIALIZE LOCAL CORE"
                color: Theme.label2
                font.family: Theme.mono
                font.pixelSize: 10
                font.letterSpacing: 2.4
            }

            Text {
                Layout.fillWidth: true
                text: "First-run setup"
                color: Theme.ink
                font.family: Theme.serif
                font.pixelSize: 32
                wrapMode: Text.WordWrap
            }

            Text {
                Layout.fillWidth: true
                text: "DocSummarizer runs entirely offline. Before the first summary, the " + "local language model is downloaded once and cached on this machine. " + "Nothing leaves your device after this step."
                color: Theme.text
                font.family: Theme.body
                font.pixelSize: 15
                lineHeight: 1.35
                wrapMode: Text.WordWrap
            }

            // Model card --------------------------------------------------- //
            Rectangle {
                Layout.fillWidth: true
                radius: 4
                color: Theme.block
                border.width: 1
                border.color: Theme.line
                implicitHeight: cardRow.implicitHeight + 28

                RowLayout {
                    id: cardRow
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.leftMargin: 16
                    anchors.rightMargin: 16
                    spacing: 12

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 5
                        Text {
                            Layout.fillWidth: true
                            text: screen.bridge.modelName
                            color: Theme.inkSoft
                            font.family: Theme.serif
                            font.pixelSize: 19
                            elide: Text.ElideRight
                        }
                        Text {
                            Layout.fillWidth: true
                            text: screen.bridge.modelQuant + " · " + screen.bridge.modelSizeGb + " GB · ONE-TIME"
                            color: Theme.faint
                            font.family: Theme.mono
                            font.pixelSize: 10
                            font.letterSpacing: 1.2
                        }
                    }

                    Text {
                        text: Math.round(screen.bridge.downloadPercent) + "%"
                        color: Theme.accent
                        font.family: Theme.mono
                        font.pixelSize: 26
                        Layout.alignment: Qt.AlignVCenter
                    }
                }
            }

            ProgressBar {
                Layout.fillWidth: true
                from: 0
                to: 1
                value: screen.bridge.downloadPercent / 100
            }

            Text {
                Layout.fillWidth: true
                text: screen.statusLine
                color: Theme.faint
                font.family: Theme.mono
                font.pixelSize: 11
                font.letterSpacing: 0.8
                wrapMode: Text.WordWrap
            }

            // Primary action --------------------------------------------- //
            Button {
                id: action
                Layout.fillWidth: true
                enabled: !(screen.started && screen.bridge.downloadPercent < 100)
                text: screen.bridge.downloadPercent >= 100 ? "Enter Console" : (screen.started ? "Downloading… " + Math.round(screen.bridge.downloadPercent) + "%" : "Begin Download")
                onClicked: {
                    if (!screen.started) {
                        screen.started = true;
                        screen.bridge.beginDownload();
                    } else if (screen.bridge.downloadPercent >= 100) {
                        screen.enter();
                    }
                }

                contentItem: Text {
                    text: action.text
                    color: action.enabled ? Theme.inkOnAccent : Theme.dim
                    font.family: Theme.ui
                    font.pixelSize: 13
                    font.letterSpacing: 0.8
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                background: Rectangle {
                    implicitHeight: 44
                    radius: 3
                    color: action.enabled ? Theme.accent : Theme.block
                    border.width: 1
                    border.color: action.enabled ? Theme.accentDeep : Theme.line2
                }
            }
        }
    }
}
