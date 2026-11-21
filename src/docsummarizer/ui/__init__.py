# SPDX-License-Identifier: MIT
"""Qt/QML user interface for DocSummarizer (the "Abstract Console" redesign).

`app.py` boots a `QQmlApplicationEngine`, `bridge.py` exposes the backend to
QML via a `ConsoleBridge` QObject, and `workers.py` runs long operations off
the UI thread. QML sources live in `qml/`. Importing this package does not
require llama-cpp (the model is built lazily inside a worker).
"""
