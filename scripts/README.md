# scripts/

Developer / QA helpers (not shipped in the app).

| Script | What it does | How to run |
|--------|--------------|------------|
| `check.py` | The one gate: runs `format`, `lint`, `types`, `arch`, `qml` and `test`, or `all` of them in that order, stopping at the first red step. Tools come from the active environment (`uv sync --locked --extra dev --extra gui`); jscpd runs through `npx` at its pinned version, so Node is the one prerequisite outside the lock. | `python scripts/check.py [format\|lint\|types\|arch\|qml\|test\|all]` |
| `render_qml.py` | Renders a QML screen to a PNG **offscreen** (no display needed), seeded with sample state, so the Qt/QML UI can be reviewed against the design screenshots. | `python scripts/render_qml.py [summary\|extract\|batch\|config\|firstrun] [dark\|light] [out.png]` |
| `_preview.qml` | Item-rooted mirror of `Main.qml`'s shell used only by `render_qml.py` (an `ApplicationWindow` can't be grabbed the same way). | — |
| `eval/run.py` | Runs the evaluation tiers, rescoring, comparisons and checks against a local corpus and model; see `DEVELOPMENT.md`. | `python scripts/eval/run.py --eval-dir <eval-dir> --tier smoke --model <gguf>` |
