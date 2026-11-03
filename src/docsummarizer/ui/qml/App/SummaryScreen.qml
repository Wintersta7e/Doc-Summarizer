import QtQuick
import QtQuick.Layouts
import QtQuick.Controls.Basic
import QtQuick.Dialogs
import App

// Summary view. Two states: an empty dropzone (no document) and the loaded
// two-pane view (source + distilled summary with click-to-trace provenance).
Item {
    id: screen

    property var summary: ({"summaryType": "detailed", "lead": "", "points": [], "sections": ({}), "text": ""})
    property int activePoint: -1
    property string activeSection: ""
    property int hlStart: -1
    property int hlEnd: -1

    function cssRgba(c, a) {
        return "rgba(" + Math.round(c.r * 255) + "," + Math.round(c.g * 255) + "," + Math.round(c.b * 255) + "," + a + ")";
    }
    function escapeHtml(s) {
        return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }
    function renderSource() {
        var raw = bridge.extractedText;
        if (raw === "")
            return "";
        if (screen.hlStart >= 0 && screen.hlEnd > screen.hlStart) {
            var pre = escapeHtml(raw.substring(0, screen.hlStart));
            var hit = escapeHtml(raw.substring(screen.hlStart, screen.hlEnd));
            var post = escapeHtml(raw.substring(screen.hlEnd));
            var style = "background-color:" + cssRgba(Theme.accent, 0.20) + ";color:" + cssRgba(Theme.accent2, 1.0);
            return pre + "<span style='" + style + "'>" + hit + "</span>" + post;
        }
        return escapeHtml(raw);
    }
    function trace(point) {
        if (point && point.hasCitation) {
            screen.hlStart = point.start;
            screen.hlEnd = point.end;
        } else {
            screen.hlStart = -1;
            screen.hlEnd = -1;
        }
        srcText.text = screen.renderSource();
        // Scroll the highlighted sentence into view — on a long document it is
        // usually off-screen, so without this the trace looks like it did nothing.
        if (screen.hlStart >= 0) {
            var r = srcText.positionToRectangle(screen.hlStart);
            var maxY = Math.max(0, srcText.implicitHeight - srcFlick.height);
            srcFlick.contentY = Math.max(0, Math.min(r.y - srcFlick.height / 3, maxY));
        }
    }
    function resetProvenance() {
        screen.activePoint = -1;
        screen.activeSection = "";
        screen.hlStart = -1;
        screen.hlEnd = -1;
        srcText.text = screen.renderSource();
    }

    // Per-summary-type compression label (matches the prototype's strip).
    function compressionLabel() {
        switch (screen.summary.summaryType) {
        case "brief":
            return "1 PARAGRAPH";
        case "structured":
            return (screen.summary.sections ? Object.keys(screen.summary.sections).length : 4) + " SECTIONS";
        default:
            return (screen.summary.points ? screen.summary.points.length : 3) + " POINTS";
        }
    }

    Connections {
        target: bridge
        function onSummaryReady(variant) {
            screen.summary = variant;
            screen.resetProvenance();
        }
        function onDocChanged() {
            screen.resetProvenance();
        }
    }

    FileDialog {
        id: openDialog
        title: "Select a document"
        nameFilters: ["Documents (*.pdf *.docx *.rtf *.txt *.md)", "All files (*)"]
        onAccepted: bridge.loadDocument(selectedFile.toString())
    }

    // ====================================================================== //
    //  Empty / file-drop state
    // ====================================================================== //
    DropArea {
        anchors.fill: parent
        enabled: !bridge.hasDoc
        onEntered: dropZone.dragging = true
        onExited: dropZone.dragging = false
        onDropped: drop => {
            dropZone.dragging = false;
            if (drop.hasUrls)
                bridge.loadDocument(drop.urls[0].toString());
        }

        Rectangle {
            id: dropZone
            visible: !bridge.hasDoc
            anchors.centerIn: parent
            width: 540
            height: 320
            radius: 3
            color: dragging ? Theme.navOnBg : "transparent"
            border.width: 1
            border.color: dragging ? Theme.accent : Theme.line2
            property bool dragging: false

            ColumnLayout {
                anchors.centerIn: parent
                width: parent.width - 80
                spacing: 14

                Hex {
                    Layout.alignment: Qt.AlignHCenter
                    size: 54
                    glyph: "↓"
                }
                Text {
                    Layout.alignment: Qt.AlignHCenter
                    text: "Drop a document to summarize"
                    color: Theme.ink
                    font.family: Theme.serif
                    font.pixelSize: 25
                    font.weight: Font.DemiBold
                }
                Text {
                    Layout.alignment: Qt.AlignHCenter
                    text: "or click to browse — PDF, DOCX, RTF, TXT, MD"
                    color: Theme.faint
                    font.family: Theme.body
                    font.pixelSize: 13
                }
                Text {
                    Layout.alignment: Qt.AlignHCenter
                    text: "SUPPORTED · PDF · DOCX · RTF · TXT · MD"
                    color: Theme.dim
                    font.family: Theme.mono
                    font.pixelSize: 9
                    font.letterSpacing: 1.5
                }
                ConsoleButton {
                    Layout.alignment: Qt.AlignHCenter
                    text: "Select File"
                    primary: true
                    onClicked: openDialog.open()
                }
                RowLayout {
                    Layout.alignment: Qt.AlignHCenter
                    spacing: 8
                    Rectangle {
                        width: 6
                        height: 6
                        radius: 3
                        color: Theme.brassDot
                    }
                    Text {
                        text: "Processed entirely on your machine — files never leave this device."
                        color: Theme.faint
                        font.family: Theme.body
                        font.pixelSize: 12
                    }
                }
            }
            MouseArea {
                anchors.fill: parent
                cursorShape: Qt.PointingHandCursor
                onClicked: openDialog.open()
            }
        }
    }

    // ====================================================================== //
    //  Loaded state
    // ====================================================================== //
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 24
        spacing: 14
        visible: bridge.hasDoc

        // -- Doc header ---------------------------------------------------- //
        RowLayout {
            Layout.fillWidth: true
            spacing: 16
            ColumnLayout {
                spacing: 5
                RowLayout {
                    spacing: 10
                    Rectangle {
                        radius: 2
                        color: Theme.accent
                        Layout.preferredHeight: 18
                        Layout.preferredWidth: tag.implicitWidth + 14
                        Text {
                            id: tag
                            anchors.centerIn: parent
                            text: "PDF"
                            color: Theme.onAccent
                            font.family: Theme.ui
                            font.pixelSize: 9
                            font.weight: Font.Medium
                            font.letterSpacing: 1
                        }
                    }
                    Text {
                        text: "FILE 0x7F · " + bridge.currentFileName.toUpperCase()
                        textFormat: Text.PlainText
                        color: Theme.faint
                        font.family: Theme.mono
                        font.pixelSize: 9
                        font.letterSpacing: 1.2
                    }
                    Text {
                        text: "✕ UNLOAD"
                        color: Theme.faint
                        font.family: Theme.mono
                        font.pixelSize: 9
                        font.letterSpacing: 1
                        MouseArea {
                            anchors.fill: parent
                            cursorShape: Qt.PointingHandCursor
                            onClicked: bridge.unloadDocument()
                        }
                    }
                }
                Text {
                    text: bridge.currentFileName
                    textFormat: Text.PlainText
                    color: Theme.ink
                    font.family: Theme.serif
                    font.pixelSize: 33
                    font.weight: Font.DemiBold
                }
                Text {
                    text: bridge.extractedText.split(/\s+/).filter(Boolean).length.toLocaleString(Qt.locale(), "f", 0) + " WORDS · " + bridge.extractedText.length.toLocaleString(Qt.locale(), "f", 0) + " CHARS · NOMINAL"
                    color: Theme.faint
                    font.family: Theme.mono
                    font.pixelSize: 9
                    font.letterSpacing: 1
                }
            }
            Item {
                Layout.fillWidth: true
            }
            SegmentedControl {
                options: bridge.summaryTypes
                current: bridge.summaryType
                onSelected: value => bridge.setSummaryType(value)
            }
        }

        // -- Compression strip --------------------------------------------- //
        RowLayout {
            Layout.fillWidth: true
            spacing: 12
            Text {
                text: "COMPRESSION"
                color: Theme.label2
                font.family: Theme.ui
                font.pixelSize: 9
                font.letterSpacing: 1.6
                font.weight: Font.Medium
            }
            Text {
                text: bridge.extractedText.split(/\s+/).filter(Boolean).length.toLocaleString(Qt.locale(), "f", 0) + "  →  " + screen.compressionLabel()
                color: Theme.text
                font.family: Theme.mono
                font.pixelSize: 9
                font.letterSpacing: 1
            }
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 2
                radius: 1
                color: Theme.block
                Rectangle {
                    width: parent.width * 0.05
                    height: parent.height
                    radius: 1
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
                }
            }
            Text {
                text: screen.summary.summaryType === "brief" ? "−99%" : screen.summary.summaryType === "structured" ? "−94%" : "−96%"
                color: Theme.accent
                font.family: Theme.mono
                font.pixelSize: 9
                font.letterSpacing: 1
            }
        }

        // -- Two panes ----------------------------------------------------- //
        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 16

            // Source pane (45/55 split via preferred-width ratio).
            CutPanel {
                Layout.preferredWidth: 45
                Layout.fillWidth: true
                Layout.fillHeight: true
                cut: 16
                fill: Theme.srcPane
                stroke: Theme.line
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 10
                    SectionLabel {
                        text: "SOURCE · ABSTRACT"
                    }
                    Flickable {
                        id: srcFlick
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        contentWidth: width
                        contentHeight: srcText.implicitHeight
                        ScrollBar.vertical: ScrollBar {}
                        TextEdit {
                            id: srcText
                            width: parent.width
                            readOnly: true
                            wrapMode: Text.WordWrap
                            selectByMouse: false
                            textFormat: TextEdit.RichText
                            color: Theme.text
                            font.family: Theme.body
                            font.pixelSize: 15
                            text: screen.renderSource()
                        }
                    }
                }
            }

            // Summary pane
            CutPanel {
                Layout.preferredWidth: 55
                Layout.fillWidth: true
                Layout.fillHeight: true
                cut: 16
                fill: "transparent"
                stroke: Theme.line
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 10
                    opacity: bridge.busy ? 0 : 1
                    RowLayout {
                        Layout.fillWidth: true
                        SectionLabel {
                            text: "DISTILLED · " + String(screen.summary.summaryType).toUpperCase()
                        }
                        Item {
                            Layout.fillWidth: true
                        }
                        Text {
                            visible: screen.summary.summaryType !== "brief"
                            text: screen.activePoint >= 0 ? "POINT " + (screen.activePoint + 1) + " → SOURCE" : "POINT n → SOURCE"
                            color: screen.activePoint >= 0 ? Theme.accent : Theme.dim
                            font.family: Theme.mono
                            font.pixelSize: 9
                            font.letterSpacing: 1
                        }
                    }

                    // Lead (Detailed + Brief)
                    Text {
                        visible: String(screen.summary.lead) !== ""
                        Layout.fillWidth: true
                        text: screen.summary.lead
                        textFormat: Text.MarkdownText
                        wrapMode: Text.WordWrap
                        color: Theme.ink
                        font.family: Theme.serif
                        font.pixelSize: screen.summary.summaryType === "brief" ? 21 : 19
                        lineHeight: 1.35
                    }

                    // Detailed: key-point rows with hex markers
                    ListView {
                        visible: screen.summary.summaryType === "detailed"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        spacing: 8
                        model: screen.summary.points
                        delegate: Rectangle {
                            required property var modelData
                            required property int index
                            width: ListView.view ? ListView.view.width : 0
                            height: kpRow.implicitHeight + 22
                            radius: 3
                            color: index === screen.activePoint ? Theme.navOnBg : Theme.kpRest
                            border.width: 1
                            border.color: index === screen.activePoint ? Theme.navOnRing : Theme.line
                            RowLayout {
                                id: kpRow
                                x: 12
                                y: 11
                                width: parent.width - 24
                                spacing: 12
                                Hex {
                                    size: 24
                                    glyph: String(index + 1).padStart(2, "0")
                                    glyphSize: 10
                                    accentInner: index === screen.activePoint
                                }
                                ColumnLayout {
                                    Layout.fillWidth: true
                                    spacing: 3
                                    Text {
                                        Layout.fillWidth: true
                                        text: modelData.text
                                        textFormat: Text.MarkdownText
                                        color: Theme.inkSoft
                                        font.family: Theme.ui
                                        font.pixelSize: 14
                                        font.weight: Font.Medium
                                        wrapMode: Text.WordWrap
                                    }
                                    Text {
                                        visible: modelData.hasCitation
                                        text: "▸ traces to source"
                                        color: Theme.accent
                                        font.family: Theme.mono
                                        font.pixelSize: 9
                                        font.letterSpacing: 0.8
                                    }
                                }
                            }
                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: {
                                    screen.activePoint = index;
                                    screen.trace(modelData);
                                }
                            }
                        }
                    }

                    // Structured: PURPOSE / METHOD / RESULTS / CONCLUSIONS
                    ColumnLayout {
                        visible: screen.summary.summaryType === "structured"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        spacing: 12
                        Repeater {
                            model: ["PURPOSE", "METHOD", "RESULTS", "CONCLUSIONS"]
                            delegate: Rectangle {
                                required property string modelData
                                property var pts: (screen.summary.sections && screen.summary.sections[modelData]) ? screen.summary.sections[modelData] : []
                                property var firstPt: pts.length > 0 ? pts[0] : null
                                property bool traceable: firstPt && firstPt.hasCitation
                                visible: pts.length > 0
                                Layout.fillWidth: true
                                Layout.preferredHeight: secCol.implicitHeight + 22
                                radius: 3
                                color: screen.activeSection === modelData ? Theme.navOnBg : Theme.kpRest
                                Rectangle {
                                    visible: traceable
                                    width: 2
                                    height: parent.height
                                    color: screen.activeSection === modelData ? Theme.accent : Theme.line2
                                }
                                ColumnLayout {
                                    id: secCol
                                    x: 14
                                    y: 11
                                    width: parent.width - 28
                                    spacing: 4
                                    Text {
                                        text: modelData
                                        color: Theme.label2
                                        font.family: Theme.ui
                                        font.pixelSize: 9
                                        font.letterSpacing: 1.6
                                        font.weight: Font.Medium
                                    }
                                    Repeater {
                                        model: parent.parent.pts
                                        delegate: Text {
                                            required property var modelData
                                            Layout.fillWidth: true
                                            text: modelData.text
                                            textFormat: Text.MarkdownText
                                            color: Theme.inkSoft
                                            font.family: Theme.body
                                            font.pixelSize: 13
                                            lineHeight: 1.45
                                            wrapMode: Text.WordWrap
                                        }
                                    }
                                }
                                MouseArea {
                                    anchors.fill: parent
                                    enabled: traceable
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: {
                                        screen.activeSection = modelData;
                                        screen.trace(firstPt);
                                    }
                                }
                            }
                        }
                        Item {
                            Layout.fillHeight: true
                        }
                    }

                    // Trace hint
                    Text {
                        visible: screen.summary.summaryType !== "brief"
                        text: "TRACE — Select a point to follow it back to its source."
                        color: Theme.dim
                        font.family: Theme.mono
                        font.pixelSize: 9
                        font.letterSpacing: 0.8
                    }
                }

                // "Summarizing…" shimmer while a summary generates.
                Skeleton {
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.top: parent.top
                    anchors.leftMargin: 16
                    anchors.rightMargin: 16
                    anchors.topMargin: 52
                    visible: bridge.busy
                }
            }
        }

        // -- Footer -------------------------------------------------------- //
        RowLayout {
            Layout.fillWidth: true
            spacing: 12
            RowLayout {
                spacing: 8
                Rectangle {
                    width: 7
                    height: 7
                    radius: 3.5
                    color: Theme.statusColorFor(bridge.statusColor)
                }
                Text {
                    text: bridge.busy ? "GENERATING…" : "GENERATED LOCALLY"
                    color: Theme.faint
                    font.family: Theme.mono
                    font.pixelSize: 9
                    font.letterSpacing: 1
                }
            }
            Item {
                Layout.fillWidth: true
            }
            ConsoleButton {
                text: "Stop"
                visible: bridge.busy
                onClicked: bridge.cancelSummarize()
            }
            ConsoleButton {
                text: "Copy"
                enabled: !bridge.busy
                onClicked: {
                    copyHelper.text = screen.summary.text;
                    copyHelper.selectAll();
                    copyHelper.copy();
                    bridge.toast("Summary copied");
                }
            }
            ConsoleButton {
                text: "Regenerate"
                enabled: bridge.canSummarize && !bridge.busy
                onClicked: bridge.regenerate()
            }
            ConsoleButton {
                text: "Save Summary"
                primary: true
                enabled: bridge.canSummarize && !bridge.busy
                onClicked: saveDialog.open()
            }
        }
    }

    TextEdit {
        id: copyHelper
        textFormat: TextEdit.PlainText
        visible: false
    }

    FileDialog {
        id: saveDialog
        title: "Save summary"
        fileMode: FileDialog.SaveFile
        defaultSuffix: "txt"  // so a name typed without an extension still saves
        nameFilters: ["Text (*.txt)", "Word document (*.docx)"]
        onAccepted: bridge.saveSummary(selectedFile.toString(), selectedFile.toString().endsWith(".docx"))
    }
}
