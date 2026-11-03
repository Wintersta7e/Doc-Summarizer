import QtQuick
import QtQuick.Layouts
import QtQuick.Controls.Basic
import QtQuick.Dialogs
import App

// Batch queue: pick a folder + output dir, process all supported docs, and watch
// per-file status (QUEUED → PROCESSING → DONE/FAILED/SKIPPED) plus an overall bar.
Item {
    id: screen

    property string folder: ""
    property string outDir: ""
    property int doneCount: 0
    property int totalCount: 0
    property real progress: 0

    Connections {
        target: bridge
        function onBatchProgress(done, total, name) {
            screen.totalCount = total;
            screen.doneCount = done;  // files completed before the in-flight one
            screen.progress = total > 0 ? done / total : 0;
        }
        function onBatchComplete(done, total, failures, outFolder) {
            screen.doneCount = done;
            screen.totalCount = total;
            screen.progress = 1;
        }
    }

    // Let the bridge convert URLs (cross-platform) rather than hand-rolling it.
    FolderDialog {
        id: folderDialog
        onAccepted: screen.folder = bridge.urlToPath(selectedFolder.toString())
    }
    FolderDialog {
        id: outputDialog
        onAccepted: screen.outDir = bridge.urlToPath(selectedFolder.toString())
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 24
        spacing: 16

        // -- Header -------------------------------------------------------- //
        RowLayout {
            Layout.fillWidth: true
            spacing: 16
            ColumnLayout {
                spacing: 5
                SectionLabel {
                    text: "BATCH QUEUE"
                }
                Text {
                    text: screen.folder !== "" ? screen.folder : "Choose a folder…"
                    color: Theme.ink
                    font.family: Theme.serif
                    font.pixelSize: 30
                    font.weight: Font.DemiBold
                }
                Text {
                    text: bridge.batchRows.length + " DOCUMENTS · OUTPUT → " + (screen.outDir !== "" ? screen.outDir : "—") + " · " + bridge.summaryType.toUpperCase()
                    color: Theme.faint
                    font.family: Theme.mono
                    font.pixelSize: 9
                    font.letterSpacing: 1
                }
            }
            Item {
                Layout.fillWidth: true
            }
            Text {
                text: screen.doneCount + " / " + (screen.totalCount > 0 ? screen.totalCount : bridge.batchRows.length)
                color: Theme.accent
                font.family: Theme.mono
                font.pixelSize: 11
                font.letterSpacing: 1
                Layout.alignment: Qt.AlignVCenter
            }
            ConsoleButton {
                text: "Add Folder"
                onClicked: folderDialog.open()
            }
            ConsoleButton {
                text: "Output…"
                onClicked: outputDialog.open()
            }
            ConsoleButton {
                text: bridge.busy ? "Processing…" : "Process All"
                primary: true
                enabled: screen.folder !== "" && screen.outDir !== "" && bridge.modelReady && !bridge.busy
                onClicked: bridge.batchProcess(screen.folder, screen.outDir)
            }
        }

        // -- Overall strip ------------------------------------------------- //
        RowLayout {
            Layout.fillWidth: true
            spacing: 12
            Text {
                text: "OVERALL"
                color: Theme.faint
                font.family: Theme.mono
                font.pixelSize: 10
                font.letterSpacing: 1.3
            }
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 3
                radius: 1.5
                color: Theme.block
                Rectangle {
                    width: parent.width * screen.progress
                    height: parent.height
                    radius: 1.5
                    gradient: Gradient {
                        orientation: Gradient.Horizontal
                        GradientStop {
                            position: 0
                            color: Theme.accentDeep
                        }
                        GradientStop {
                            position: 1
                            color: Theme.accent2
                        }
                    }
                    Behavior on width {
                        NumberAnimation {
                            duration: 500
                        }
                    }
                }
            }
            Text {
                text: Math.round(screen.progress * 100) + "%"
                color: Theme.accent
                font.family: Theme.mono
                font.pixelSize: 10
            }
        }

        // -- Rows ---------------------------------------------------------- //
        ListView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            spacing: 8
            model: bridge.batchRows
            delegate: Rectangle {
                required property var modelData
                required property int index
                width: ListView.view ? ListView.view.width : 0
                height: 64
                radius: 3
                color: Theme.kpRest
                border.width: 1
                border.color: Theme.line
                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 16
                    anchors.rightMargin: 16
                    spacing: 14
                    Hex {
                        size: 22
                        glyph: String(index + 1).padStart(2, "0")
                        glyphSize: 9
                    }
                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 3
                        Text {
                            Layout.fillWidth: true
                            text: modelData.name
                            color: Theme.inkSoft
                            font.family: Theme.body
                            font.pixelSize: 13
                            elide: Text.ElideRight
                        }
                        Text {
                            text: modelData.status === "DONE" ? modelData.tokens + " TOK" : modelData.status.toLowerCase()
                            color: Theme.faint
                            font.family: Theme.mono
                            font.pixelSize: 9
                            font.letterSpacing: 0.5
                        }
                    }
                    StatusChip {
                        status: modelData.status
                        meta: modelData.status === "DONE" ? modelData.tokens + " TOK" : ""
                    }
                }
            }
        }
    }

    // Empty hint when nothing queued yet.
    Text {
        anchors.centerIn: parent
        visible: bridge.batchRows.length === 0
        text: "Add a folder of documents to begin"
        color: Theme.faint
        font.family: Theme.mono
        font.pixelSize: 13
        font.letterSpacing: 1
    }
}
