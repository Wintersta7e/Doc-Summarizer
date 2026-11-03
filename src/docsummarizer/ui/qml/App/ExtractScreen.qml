import QtQuick
import QtQuick.Layouts
import QtQuick.Controls.Basic
import App

// Raw extracted text: line-numbered feed with live search (highlight + count +
// step-to-match). Matches the prototype's Extract view.
Item {
    id: screen

    property string query: ""
    property int activeMatch: 0

    function lines() {
        return bridge.extractedText.split("\n");
    }
    function wordCount() {
        return bridge.extractedText.split(/\s+/).filter(Boolean).length;
    }

    // [{line, count}] and total matches, recomputed when query/text changes.
    property var matchLines: []
    property int matchCount: 0
    function recomputeMatches() {
        var q = screen.query.toLowerCase();
        var ml = [];
        var total = 0;
        if (q !== "") {
            var ls = screen.lines();
            for (var i = 0; i < ls.length; i++) {
                var c = ls[i].toLowerCase().split(q).length - 1;
                if (c > 0) {
                    ml.push(i);
                    total += c;
                }
            }
        }
        screen.matchLines = ml;
        screen.matchCount = total;
        screen.activeMatch = 0;  // a fresh search starts stepping at the first match
    }
    onQueryChanged: recomputeMatches()
    Connections {
        target: bridge
        function onDocChanged() {
            screen.recomputeMatches();
        }
    }

    function escapeHtml(s) {
        return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }
    function cssRgba(c, a) {
        return "rgba(" + Math.round(c.r * 255) + "," + Math.round(c.g * 255) + "," + Math.round(c.b * 255) + "," + a + ")";
    }
    function highlight(line) {
        var q = screen.query;
        if (q === "")
            return escapeHtml(line);
        var lc = line.toLowerCase();
        var ql = q.toLowerCase();
        var out = "";
        var i = 0;
        var idx;
        var style = "background-color:" + cssRgba(Theme.accent, 0.10) + ";color:" + cssRgba(Theme.accent2, 1.0);
        while ((idx = lc.indexOf(ql, i)) >= 0) {
            out += escapeHtml(line.substring(i, idx));
            out += "<span style='" + style + "'>" + escapeHtml(line.substring(idx, idx + q.length)) + "</span>";
            i = idx + q.length;
        }
        out += escapeHtml(line.substring(i));
        return out;
    }
    function stepMatch() {
        if (screen.matchLines.length === 0)
            return;
        // Jump to the current match, then advance — so the first press lands on
        // match 0 instead of skipping it.
        feed.positionViewAtIndex(screen.matchLines[screen.activeMatch], ListView.Center);
        screen.activeMatch = (screen.activeMatch + 1) % screen.matchLines.length;
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 24
        spacing: 14

        // -- Header -------------------------------------------------------- //
        RowLayout {
            Layout.fillWidth: true
            spacing: 16
            ColumnLayout {
                spacing: 5
                SectionLabel {
                    text: "RAW EXTRACTION"
                }
                Text {
                    text: bridge.currentFileName !== "" ? bridge.currentFileName : "No document loaded"
                    textFormat: Text.PlainText
                    color: Theme.ink
                    font.family: Theme.serif
                    font.pixelSize: 30
                    font.weight: Font.DemiBold
                }
                Text {
                    text: screen.wordCount().toLocaleString(Qt.locale(), "f", 0) + " WORDS · " + bridge.extractedText.length.toLocaleString(Qt.locale(), "f", 0) + " CHARS · UTF-8 · pypdf"
                    color: Theme.faint
                    font.family: Theme.mono
                    font.pixelSize: 9
                    font.letterSpacing: 1
                }
            }
            Item {
                Layout.fillWidth: true
            }
            // Search field with glyph + inline count.
            Rectangle {
                Layout.preferredWidth: 240
                Layout.preferredHeight: 32
                radius: 2
                color: "transparent"
                border.width: 1
                border.color: Theme.line2
                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 10
                    anchors.rightMargin: 10
                    spacing: 6
                    Text {
                        text: "⌕"
                        color: Theme.faint
                        font.pixelSize: 13
                    }
                    TextInput {
                        id: searchInput
                        Layout.fillWidth: true
                        color: Theme.ink
                        font.family: Theme.body
                        font.pixelSize: 12
                        clip: true
                        onTextChanged: screen.query = text
                        Text {
                            anchors.verticalCenter: parent.verticalCenter
                            visible: searchInput.text === ""
                            text: "search text…"
                            color: Theme.dim
                            font.family: Theme.body
                            font.pixelSize: 12
                        }
                    }
                    Text {
                        visible: screen.query !== ""
                        text: screen.matchCount === 0 ? "NO MATCH" : screen.matchCount + (screen.matchCount === 1 ? " MATCH" : " MATCHES")
                        color: screen.matchCount === 0 ? Theme.dim : Theme.accent
                        font.family: Theme.mono
                        font.pixelSize: 9
                        font.letterSpacing: 0.5
                    }
                }
            }
            ConsoleButton {
                text: "▸"
                enabled: screen.matchCount > 0
                onClicked: screen.stepMatch()
            }
            ConsoleButton {
                text: "Copy All"
                onClicked: {
                    copyHelper.text = bridge.extractedText;
                    copyHelper.selectAll();
                    copyHelper.copy();
                    bridge.toast("Extracted text copied");
                }
            }
        }

        // -- Numbered feed ------------------------------------------------- //
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            radius: 3
            color: Theme.srcPane
            border.width: 1
            border.color: Theme.line
            ListView {
                id: feed
                anchors.fill: parent
                anchors.margins: 16
                clip: true
                spacing: 2
                ScrollBar.vertical: ScrollBar {
                    policy: ScrollBar.AlwaysOn
                }
                model: screen.lines()
                delegate: RowLayout {
                    required property var modelData
                    required property int index
                    width: feed.width
                    spacing: 14
                    Text {
                        Layout.preferredWidth: 30
                        horizontalAlignment: Text.AlignRight
                        text: String(index + 1).padStart(3, "0")
                        color: Theme.dim
                        font.family: Theme.mono
                        font.pixelSize: 10
                    }
                    Text {
                        Layout.fillWidth: true
                        textFormat: Text.RichText
                        text: screen.highlight(modelData)
                        color: Theme.text
                        font.family: Theme.body
                        font.pixelSize: 13
                        lineHeight: 1.5
                        wrapMode: Text.WordWrap
                    }
                }
            }
        }
    }

    TextEdit {
        id: copyHelper
        textFormat: TextEdit.PlainText
        visible: false
    }
}
