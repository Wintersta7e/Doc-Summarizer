# SPDX-License-Identifier: MIT
from pathlib import Path

def hf_hub_download(
    *,
    repo_id: str,
    filename: str,
    local_dir: str | Path | None = None,
    # A tqdm subclass; typed as a class only, since tqdm is imported lazily.
    tqdm_class: type[object] | None = None,
) -> str: ...
