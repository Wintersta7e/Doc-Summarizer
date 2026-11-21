# SPDX-License-Identifier: MIT
"""Platform-specific application data paths.

Resolves the per-user app-data directory for DocSummarizer on each
supported platform, then appends a caller-supplied subdirectory. Both
`logger` and `model_manager` need this — keeping the platform logic
here means a future tweak (e.g. honoring a `DOCSUMMARIZER_HOME` env
var) only has to land in one place.
"""

import contextlib
import os
import sys
from collections.abc import Generator
from pathlib import Path

_APP_NAME = "DocSummarizer"
# Roots that replace the per-user dir while a run must leave the user's app data
# alone (the smoke test); the innermost wins. A plain list, not a context
# variable, so worker threads the run starts resolve the same root.
_redirects: list[Path] = []


@contextlib.contextmanager
def redirected_app_data(root: Path) -> Generator[None, None, None]:
    """Resolve every app-data dir under `root` instead of the per-user dir, inside the block.

    Yields:
        Nothing; the redirect holds until the block exits.
    """
    _redirects.append(root)
    try:
        yield
    finally:
        _ = _redirects.pop()


def app_data_dir(subpath: str) -> Path:
    r"""Return the per-user data dir for `subpath`, creating it if missing.

    - Windows: `%LOCALAPPDATA%\DocSummarizer\<subpath>`
    - macOS:   `~/Library/Application Support/DocSummarizer/<subpath>`
    - Linux:   `$XDG_DATA_HOME/DocSummarizer/<subpath>` (fallback
      `~/.local/share/DocSummarizer/<subpath>`)
    - inside `redirected_app_data(root)`: `<root>/<subpath>`
    """
    if _redirects:
        root = _redirects[-1]
    elif sys.platform == "win32":
        # `or` (not get's default) so a set-but-empty var doesn't resolve to cwd.
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        root = base / _APP_NAME
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support" / _APP_NAME
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
        root = base / _APP_NAME

    target = root / subpath
    target.mkdir(parents=True, exist_ok=True)
    return target
