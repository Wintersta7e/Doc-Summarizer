pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import QtQuick.Shapes
import App

// Application shell: persistent top bar + LCARS rail + a screen Loader, with the
// first-run download overlay on top. The 1240px design width is the minimum;
// the window is resizable.
ApplicationWindow {
    id: win
    visible: true
    width: 1240
    height: 760
    minimumWidth: 1000
    minimumHeight: 640
    title: "DocSummarizer — Abstract Console"
    color: Theme.pageBg

    // NB: not "screen" — ApplicationWindow already defines a `screen` (QScreen).
    // The ConsoleBridge, handed in by the loader; every screen gets it from here.
    required property var bridge

    property string activeScreen: "summary"
    property bool forceFirstRun: false

    // Restore the persisted appearance on launch (was previously always dark).
    Component.onCompleted: Theme.applyMode(win.bridge.appearance)

    // Radial page vignette (lighter at top-center, darkening to the edges).
    Shape {
        anchors.fill: parent
        preferredRendererType: Shape.CurveRenderer
        ShapePath {
            strokeWidth: -1
            fillGradient: RadialGradient {
                centerX: win.width / 2
                centerY: -win.height * 0.18
                centerRadius: win.width * 0.95
                focalX: win.width / 2
                focalY: -win.height * 0.18
                GradientStop {
                    position: 0.0
                    color: Theme.pageTop
                }
                GradientStop {
                    position: 0.56
                    color: Theme.pageBg
                }
                GradientStop {
                    position: 1.0
                    color: Theme.pageBottom
                }
            }
            startX: 0
            startY: 0
            PathLine {
                x: win.width
                y: 0
            }
            PathLine {
                x: win.width
                y: win.height
            }
            PathLine {
                x: 0
                y: win.height
            }
        }
    }

    // Faint 40px instrument grid.
    Canvas {
        id: gridCanvas
        anchors.fill: parent
        onWidthChanged: requestPaint()
        onHeightChanged: requestPaint()
        Connections {
            target: Theme
            function onDarkChanged() {
                gridCanvas.requestPaint();
            }
        }
        onPaint: {
            var ctx = getContext("2d");
            ctx.clearRect(0, 0, width, height);
            ctx.strokeStyle = Theme.dark ? "rgba(111,195,216,0.03)" : "rgba(40,80,100,0.05)";
            ctx.lineWidth = 1;
            for (let x = 40; x < width; x += 40) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, height);
                ctx.stroke();
            }
            for (let y = 40; y < height; y += 40) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0
        // Inert behind the modal first-run overlay.
        enabled: !firstRun.holdsFocus

        TopBar {
            Layout.fillWidth: true
            bridge: win.bridge
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            Rail {
                Layout.fillHeight: true
                currentScreen: win.activeScreen
                onNavigate: name => win.activeScreen = name
            }

            // StackLayout (not a Loader) so each screen stays alive across
            // navigation — switching tabs no longer discards the generated
            // summary, search state, or scroll position.
            StackLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                currentIndex: {
                    switch (win.activeScreen) {
                    case "extract":
                        return 1;
                    case "batch":
                        return 2;
                    case "config":
                        return 3;
                    default:
                        return 0;
                    }
                }
                SummaryScreen {
                    bridge: win.bridge
                }
                ExtractScreen {
                    bridge: win.bridge
                }
                BatchScreen {
                    bridge: win.bridge
                }
                ConfigScreen {
                    bridge: win.bridge
                    onReinitialize: win.forceFirstRun = true
                }
            }
        }
    }

    // One-time model download, shown on first run or when re-initialized.
    FirstRunOverlay {
        id: firstRun
        anchors.fill: parent
        bridge: win.bridge
        shown: !win.bridge.modelDownloaded || win.forceFirstRun
        visible: shown
        onEnter: {
            win.forceFirstRun = false;
            win.bridge.checkModel();
        }
    }

    Toast {
        id: toast
    }
    Connections {
        target: win.bridge
        function onToast(message) {
            toast.show(message);
        }
    }
}
