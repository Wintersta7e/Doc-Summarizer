import QtQuick
import QtQuick.Layouts
import App

// Eyebrow/section label: a short accent tick + uppercase Chakra-Petch text.
RowLayout {
    id: sectionLabel
    property alias text: label.text
    property color tickColor: Theme.accent
    spacing: 8

    Rectangle {
        Layout.preferredWidth: 2
        Layout.preferredHeight: 12
        color: sectionLabel.tickColor
    }
    Text {
        id: label
        color: Theme.label2
        font.family: Theme.ui
        font.pixelSize: 9
        font.letterSpacing: 1.8
        font.weight: Font.Medium
    }
}
