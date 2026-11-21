pragma Singleton
import QtQuick

// Design tokens for the "Abstract Console" UI, translated from the prototype's
// CSS custom properties. `dark` flips the whole palette. Decorative glows /
// scanlines are deferred to the visual-polish pass; these are the load-bearing
// color/type tokens every screen binds to.
QtObject {
    id: theme
    property bool dark: true

    // Apply a persisted appearance mode ("System" | "Light" | "Dark").
    function applyMode(mode) {
        if (mode === "Light")
            dark = false;
        else if (mode === "Dark")
            dark = true;
        else
            dark = (Application.styleHints.colorScheme === Qt.ColorScheme.Dark);  // System → follow OS
    }

    // Surfaces
    readonly property color pageBg: dark ? "#080d14" : "#eef2f6"
    readonly property color pageTop: dark ? "#0d1721" : "#ffffff"
    readonly property color pageBottom: dark ? "#060a0f" : "#e3e9ef"
    readonly property color shellTop: dark ? "#0b121a" : "#ffffff"
    readonly property color shellBottom: dark ? "#080d13" : "#f6f9fb"
    readonly property color block: dark ? "#0e1820" : "#eaeff3"
    readonly property color kpRest: dark ? Qt.rgba(1, 1, 1, 0.012) : Qt.rgba(0.08, 0.16, 0.24, 0.015)
    readonly property color hexBg: dark ? "#0a1c22" : "#e2edf1"
    readonly property color srcPane: dark ? Qt.rgba(0, 0, 0, 0.18) : Qt.rgba(0.118, 0.353, 0.471, 0.04)
    readonly property color overlay: dark ? Qt.rgba(0.016, 0.027, 0.043, 0.78) : Qt.rgba(0.882, 0.91, 0.933, 0.82)

    // Lines / rings
    readonly property color line: dark ? Qt.rgba(0.435, 0.765, 0.847, 0.09) : Qt.rgba(0.157, 0.314, 0.392, 0.12)
    readonly property color line2: dark ? Qt.rgba(0.435, 0.765, 0.847, 0.13) : Qt.rgba(0.157, 0.314, 0.392, 0.18)
    readonly property color ring: dark ? Qt.rgba(0.435, 0.765, 0.847, 0.14) : Qt.rgba(0.157, 0.314, 0.392, 0.18)

    // Text
    readonly property color ink: dark ? "#eef5f6" : "#16242c"
    readonly property color inkSoft: dark ? "#dfecef" : "#26343d"
    readonly property color text: dark ? "#92a4ae" : "#52626c"
    readonly property color label: dark ? "#7b94a0" : "#586a74"
    readonly property color label2: dark ? "#9ad6e6" : "#1f8aa6"
    readonly property color faint: dark ? "#566c76" : "#728791"
    readonly property color dim: dark ? "#46606c" : "#9aadb7"

    // Accents
    readonly property color accent: dark ? "#6fc3d8" : "#1f8aa6"
    readonly property color accent2: dark ? "#9ad6e6" : "#2aa6c4"
    readonly property color accentDeep: dark ? "#56a9be" : "#18748c"
    readonly property color onAccent: dark ? "#06222a" : "#ffffff"
    readonly property color brass: dark ? "#d8bd86" : "#8c6f33"
    readonly property color brassDot: dark ? "#c7a96b" : "#b08d45"
    readonly property color brassRing: dark ? Qt.rgba(0.78, 0.66, 0.42, 0.32) : Qt.rgba(0.59, 0.47, 0.22, 0.36)

    // Status colors (keyed by ConsoleBridge.statusColor: ok|warn|error)
    readonly property color statusOk: accent
    readonly property color statusWarn: brass
    readonly property color statusError: dark ? "#e0726a" : "#c0392b"

    // Nav rail
    readonly property color navOnBg: dark ? Qt.rgba(0.435, 0.765, 0.847, 0.10) : Qt.rgba(0.122, 0.541, 0.651, 0.10)
    readonly property color navOnRing: dark ? Qt.rgba(0.435, 0.765, 0.847, 0.34) : Qt.rgba(0.122, 0.541, 0.651, 0.34)
    readonly property color navOff: dark ? "#6c828c" : "#5e7682"

    // Type families. Vendored woff/ttf are bundled in Phase 4; until then these
    // resolve to the nearest installed face (Qt falls back gracefully).
    readonly property string serif: "Cormorant Garamond"
    readonly property string ui: "Chakra Petch"
    readonly property string mono: "Share Tech Mono"
    readonly property string body: "Saira"

    function statusColorFor(key) {
        if (key === "error")
            return statusError;
        if (key === "warn")
            return statusWarn;
        return statusOk;
    }
}
