import QtQuick
import QtQuick.Controls.Basic
import App

// Console button: ghost (transparent + 1px ring) by default, or `primary` (accent
// gradient fill). Replaces the bare gray Controls.Basic chrome.
Button {
    id: ctrl
    property bool primary: false

    implicitHeight: 34
    leftPadding: 18
    rightPadding: 18

    background: Rectangle {
        radius: 2
        color: "transparent"
        border.width: ctrl.primary ? 0 : 1
        border.color: Theme.line2
        opacity: ctrl.enabled ? (ctrl.down ? 0.85 : 1.0) : 0.4

        GlowEffect {
            anchors.fill: parent
            anchors.margins: -7
            visible: ctrl.primary
            z: -1
            intensity: 0.4
        }

        // Accent gradient fill for the primary variant (child rect avoids the
        // conditional-gradient-null gotcha on the parent).
        Rectangle {
            anchors.fill: parent
            radius: 2
            visible: ctrl.primary
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
        }

        FocusRing {
            visible: ctrl.visualFocus
            onAccent: ctrl.primary
        }
    }
    contentItem: Text {
        text: ctrl.text
        color: ctrl.primary ? Theme.inkOnAccent : Theme.label
        font.family: Theme.ui
        font.pixelSize: 11
        font.weight: Font.Medium
        font.letterSpacing: 0.6
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
    }
}
