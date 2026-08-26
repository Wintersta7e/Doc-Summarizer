import QtQuick
import QtQuick.Layouts
import QtQuick.Controls.Basic
import App

// Config view: model core status, compute (GPU + thread count), appearance, and
// runtime/reload controls. Mirrors SummaryScreen's import set and Theme/bridge
// idioms. The "SAVED TO DISK" badge briefly fades in whenever the bridge emits
// savedFlash, confirming settings were persisted.
Item {
    id: screen

    signal reinitialize

    function flashSaved() {
        savedBadge.opacity = 1;
        savedTimer.restart();
    }

    Connections {
        target: bridge
        function onSavedFlash() {
            screen.flashSaved();
        }
    }

    Timer {
        id: savedTimer
        interval: 1700
        repeat: false
        onTriggered: savedBadge.opacity = 0
    }

    Flickable {
        anchors.fill: parent
        contentWidth: width
        contentHeight: column.implicitHeight + 48
        clip: true

        ColumnLayout {
            id: column
            x: 24
            y: 24
            width: parent.width - 48
            spacing: 18

            // -- Header ---------------------------------------------------- //
            RowLayout {
                Layout.fillWidth: true
                spacing: 16
                ColumnLayout {
                    spacing: 4
                    Text {
                        text: "CONFIGURATION"
                        color: Theme.label
                        font.family: Theme.ui
                        font.pixelSize: 10
                        font.letterSpacing: 2.4
                    }
                    Text {
                        text: "System & Compute"
                        color: Theme.ink
                        font.family: Theme.serif
                        font.pixelSize: 28
                    }
                }
                Item {
                    Layout.fillWidth: true
                }
                Row {
                    id: savedBadge
                    spacing: 8
                    opacity: 0
                    Behavior on opacity {
                        NumberAnimation {
                            duration: 280
                        }
                    }
                    Rectangle {
                        width: 8
                        height: 8
                        radius: 4
                        anchors.verticalCenter: parent.verticalCenter
                        color: Theme.accent
                    }
                    Text {
                        anchors.verticalCenter: parent.verticalCenter
                        text: "SAVED TO DISK"
                        color: Theme.accent
                        font.family: Theme.mono
                        font.pixelSize: 10
                        font.letterSpacing: 1.4
                    }
                }
            }

            // -- Cards ----------------------------------------------------- //
            GridLayout {
                Layout.fillWidth: true
                columns: 2
                columnSpacing: 16
                rowSpacing: 16

                // (1) MODEL CORE --------------------------------------------- //
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: modelCol.implicitHeight + 36
                    radius: 4
                    color: Theme.block
                    border.width: 1
                    border.color: Theme.line
                    ColumnLayout {
                        id: modelCol
                        x: 18
                        y: 18
                        width: parent.width - 36
                        spacing: 8
                        Text {
                            text: "MODEL CORE"
                            color: Theme.label
                            font.family: Theme.ui
                            font.pixelSize: 10
                            font.letterSpacing: 2.2
                        }
                        Text {
                            Layout.fillWidth: true
                            text: bridge.modelName !== "" ? bridge.modelName : "No model selected"
                            color: Theme.ink
                            font.family: Theme.serif
                            font.pixelSize: 22
                            wrapMode: Text.WordWrap
                        }
                        Text {
                            Layout.fillWidth: true
                            text: bridge.modelQuant + " · " + bridge.modelSizeGb.toFixed(1) + " GB · CTX " + bridge.contextSize
                            color: Theme.text
                            font.family: Theme.mono
                            font.pixelSize: 11
                            font.letterSpacing: 0.6
                            wrapMode: Text.WordWrap
                        }
                        Text {
                            text: bridge.modelDownloaded ? "● DOWNLOADED · VERIFIED" : "● NOT DOWNLOADED"
                            color: bridge.modelDownloaded ? Theme.accent : Theme.brass
                            font.family: Theme.mono
                            font.pixelSize: 10
                            font.letterSpacing: 1
                        }
                        Button {
                            text: "Re-initialize"
                            onClicked: screen.reinitialize()
                        }
                    }
                }

                // (2) COMPUTE ------------------------------------------------ //
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: computeCol.implicitHeight + 36
                    radius: 4
                    color: Theme.block
                    border.width: 1
                    border.color: Theme.line
                    ColumnLayout {
                        id: computeCol
                        x: 18
                        y: 18
                        width: parent.width - 36
                        spacing: 10
                        Text {
                            text: "COMPUTE"
                            color: Theme.label
                            font.family: Theme.ui
                            font.pixelSize: 10
                            font.letterSpacing: 2.2
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12
                            Text {
                                text: "GPU OFFLOAD"
                                color: Theme.inkSoft
                                font.family: Theme.ui
                                font.pixelSize: 12
                                font.letterSpacing: 0.6
                            }
                            Item {
                                Layout.fillWidth: true
                            }
                            // GPU toggle pill
                            Rectangle {
                                id: gpuPill
                                width: 52
                                height: 26
                                radius: 13
                                opacity: bridge.gpuSupported ? 1.0 : 0.4
                                color: bridge.gpuEnabled ? Theme.accent : Theme.srcPane
                                border.width: 1
                                border.color: bridge.gpuEnabled ? Theme.accentDeep : Theme.line2
                                Rectangle {
                                    width: 20
                                    height: 20
                                    radius: 10
                                    y: 3
                                    x: bridge.gpuEnabled ? gpuPill.width - width - 3 : 3
                                    color: bridge.gpuEnabled ? Theme.onAccent : Theme.navOff
                                    Behavior on x {
                                        NumberAnimation {
                                            duration: 140
                                        }
                                    }
                                }
                                MouseArea {
                                    anchors.fill: parent
                                    enabled: bridge.gpuSupported
                                    cursorShape: bridge.gpuSupported ? Qt.PointingHandCursor : Qt.ArrowCursor
                                    onClicked: bridge.toggleGpu(!bridge.gpuEnabled)
                                }
                            }
                        }
                        Text {
                            Layout.fillWidth: true
                            text: !bridge.gpuSupported ? "GPU not available in this build (CPU-only). Inference runs on the CPU." : (bridge.gpuEnabled ? "Layers offloaded to the GPU on reload." : "Running on CPU. Enable to offload layers to the GPU.")
                            color: Theme.faint
                            font.family: Theme.body
                            font.pixelSize: 12
                            wrapMode: Text.WordWrap
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12
                            Text {
                                text: "THREADS"
                                color: Theme.inkSoft
                                font.family: Theme.ui
                                font.pixelSize: 12
                                font.letterSpacing: 0.6
                            }
                            Slider {
                                id: threadSlider
                                Layout.fillWidth: true
                                from: 2
                                to: bridge.cpuCount
                                stepSize: 1
                                value: bridge.threads
                                onMoved: bridge.setThreads(Math.round(value))
                            }
                            Text {
                                text: Math.round(threadSlider.value) + " / " + bridge.cpuCount
                                color: Theme.accent
                                font.family: Theme.mono
                                font.pixelSize: 11
                                font.letterSpacing: 0.6
                            }
                        }
                    }
                }

                // (3) APPEARANCE --------------------------------------------- //
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: appearanceCol.implicitHeight + 36
                    radius: 4
                    color: Theme.block
                    border.width: 1
                    border.color: Theme.line
                    ColumnLayout {
                        id: appearanceCol
                        x: 18
                        y: 18
                        width: parent.width - 36
                        spacing: 10
                        Text {
                            text: "APPEARANCE"
                            color: Theme.label
                            font.family: Theme.ui
                            font.pixelSize: 10
                            font.letterSpacing: 2.2
                        }
                        SegmentedControl {
                            options: ["System", "Light", "Dark"]
                            current: bridge.appearance
                            onSelected: value => {
                                bridge.setAppearance(value);
                                Theme.applyMode(value);
                            }
                        }
                    }
                }

                // (4) RUNTIME ------------------------------------------------ //
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: runtimeCol.implicitHeight + 36
                    radius: 4
                    color: Theme.block
                    border.width: 1
                    border.color: Theme.line
                    ColumnLayout {
                        id: runtimeCol
                        x: 18
                        y: 18
                        width: parent.width - 36
                        spacing: 10
                        Text {
                            text: "RUNTIME"
                            color: Theme.label
                            font.family: Theme.ui
                            font.pixelSize: 10
                            font.letterSpacing: 2.2
                        }
                        Text {
                            Layout.fillWidth: true
                            text: "Active: " + bridge.computeLabel + ". Changes take effect after a reload."
                            color: Theme.text
                            font.family: Theme.mono
                            font.pixelSize: 11
                            font.letterSpacing: 0.4
                            wrapMode: Text.WordWrap
                        }
                        Button {
                            text: "Reload Model"
                            enabled: bridge.reloadArmed && !bridge.busy
                            onClicked: bridge.reloadModel()
                        }
                    }
                }

                // (5) SUMMARY LANGUAGE --------------------------------------- //
                Rectangle {
                    Layout.fillWidth: true
                    Layout.columnSpan: 2
                    Layout.preferredHeight: languageCol.implicitHeight + 36
                    radius: 4
                    color: Theme.block
                    border.width: 1
                    border.color: Theme.line
                    ColumnLayout {
                        id: languageCol
                        x: 18
                        y: 18
                        width: parent.width - 36
                        spacing: 10
                        Text {
                            text: "SUMMARY LANGUAGE"
                            color: Theme.label
                            font.family: Theme.ui
                            font.pixelSize: 10
                            font.letterSpacing: 2.2
                        }
                        ComboBox {
                            id: languageBox
                            Layout.preferredWidth: 280
                            // "auto" is the stored value; only its label is dressed up.
                            readonly property var labels: bridge.outputLanguages.map(function (name) {
                                return name === "auto" ? "Auto — match the document" : name;
                            })
                            model: labels
                            currentIndex: bridge.outputLanguages.indexOf(bridge.outputLanguage)
                            // A language hand-edited into settings.json need not be
                            // on the list; show it rather than a blank box.
                            displayText: currentIndex >= 0 ? labels[currentIndex] : bridge.outputLanguage
                            onActivated: index => bridge.setOutputLanguage(bridge.outputLanguages[index])
                        }
                        Text {
                            Layout.fillWidth: true
                            text: bridge.outputLanguage === "auto" ? "Summaries follow the document's own language. Applies to the next summary." : "Summaries are written in " + bridge.outputLanguage + " whatever the document's language. Applies to the next summary."
                            color: Theme.faint
                            font.family: Theme.body
                            font.pixelSize: 12
                            wrapMode: Text.WordWrap
                        }
                    }
                }
            }
        }
    }
}
