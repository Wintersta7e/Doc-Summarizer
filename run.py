#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""DocSummarizer - Offline Document Summarization Tool
Main entry point.
"""

import sys
from pathlib import Path


class _NullWriter:
    """Discard writes when stdout/stderr is unavailable (Windows GUI mode)."""

    def write(self, _text: str) -> int:
        return len(_text)

    def flush(self) -> None:
        pass


# Windows GUI processes can have sys.stdout / sys.stderr set to None. Libraries
# like huggingface_hub write to stdout during downloads and would crash with
# "'NoneType' object has no attribute 'write'".
if sys.stdout is None:
    sys.stdout = _NullWriter()
if sys.stderr is None:
    sys.stderr = _NullWriter()


try:
    from docsummarizer.ui.app import main
except ImportError:
    # Running from source without `pip install -e .`. Make src/ importable.
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from docsummarizer.ui.app import main


if __name__ == "__main__":
    main()
