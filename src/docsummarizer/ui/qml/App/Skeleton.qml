pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Layouts
import App

// "Summarizing…" loading state: a pulsing accent dot + label over animated
// placeholder bars with a shimmer sweep.
ColumnLayout {
    id: skel
    spacing: 12

    RowLayout {
        spacing: 8
        Rectangle {
            width: 7
            height: 7
            radius: 3.5
            color: Theme.accent
            SequentialAnimation on opacity {
                running: skel.visible
                loops: Animation.Infinite
                NumberAnimation {
                    to: 0.3
                    duration: 700
                }
                NumberAnimation {
                    to: 1.0
                    duration: 700
                }
            }
        }
        Text {
            text: "SUMMARIZING…"
            color: Theme.accent
            font.family: Theme.mono
            font.pixelSize: 10
            font.letterSpacing: 1
        }
    }

    Repeater {
        model: 5
        delegate: Rectangle {
            id: bar
            required property int index
            Layout.fillWidth: true
            Layout.rightMargin: [0, 60, 20, 120, 40][bar.index]
            Layout.preferredHeight: 14
            radius: 3
            color: Theme.block
            clip: true
            Rectangle {
                height: parent.height
                width: parent.width * 0.4
                gradient: Gradient {
                    orientation: Gradient.Horizontal
                    GradientStop {
                        position: 0.0
                        color: "transparent"
                    }
                    GradientStop {
                        position: 0.5
                        color: Theme.line2
                    }
                    GradientStop {
                        position: 1.0
                        color: "transparent"
                    }
                }
                NumberAnimation on x {
                    running: skel.visible
                    loops: Animation.Infinite
                    from: -bar.width * 0.4
                    to: bar.width
                    duration: 1250
                }
            }
        }
    }
}
