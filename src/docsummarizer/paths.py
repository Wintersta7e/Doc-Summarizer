# SPDX-License-Identifier: MIT
"""Platform-specific application data paths.

Resolves the per-user app-data directory for DocSummarizer on each
supported platform, then appends a caller-supplied subdirectory. Both
`logger` and `model_manager` need this — keeping the platform logic
here means a future tweak (e.g. honoring a `DOCSUMMARIZER_HOME` env
var) only has to land in one place.
"""

import os
import sys
from pathlib import Path

_APP_NAME = "DocSummarizer"


def app_data_dir(subpath: str) -> Path:
    """Return the per-user data dir for `subpath`, creating it if missing.

    - Windows: `%LOCALAPPDATA%\\DocSummarizer\\<subpath>`
    - macOS:   `~/Library/Application Support/DocSummarizer/<subpath>`
    - Linux:   `$XDG_DATA_HOME/DocSummarizer/<subpath>` (fallback
      `~/.local/share/DocSummarizer/<subpath>`)
    """
    if sys.platform == "win32":
        # `or` (not get's default) so a set-but-empty var doesn't resolve to cwd.
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")

    target = base / _APP_NAME / subpath
    target.mkdir(parents=True, exist_ok=True)
    return target
