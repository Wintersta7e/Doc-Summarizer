# SPDX-License-Identifier: MIT
"""Settings files written by earlier releases still load.

Each ``fixtures/upgrade/<tag>/settings.json`` was written by that release's own
``save_settings``, run from a checkout of the tag: its defaults with every
field it knew changed once. The current code must read each file without a
warning, keep every value the file holds, and give fields added since then
their defaults.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
from pathlib import Path
from typing import cast

import pytest

from docsummarizer.settings import Settings, load_settings, settings_path

_FIXTURES = Path(__file__).parent / "fixtures" / "upgrade"

# Every release that shipped a settings file; each later release adds its own.
_RELEASES = ("v2.0.0", "v2.0.1")


@pytest.mark.parametrize("release", _RELEASES)
def test_release_settings_still_load(release: str, caplog: pytest.LogCaptureFixture) -> None:
    """A release's settings file loads with its values kept and new fields defaulted.

    Mutations caught: a loader that drops a field it still knows (the value
    falls back to its default), one that rejects a value an earlier release
    wrote, and one that no longer reads the file at all.
    """
    fixture = _FIXTURES / release / "settings.json"
    written = cast("dict[str, object]", json.loads(fixture.read_text(encoding="utf-8")))
    defaults = dataclasses.asdict(Settings())
    assert set(written) <= set(defaults), "a field this release wrote is no longer known"
    assert all(written[name] != defaults[name] for name in written), "fixture holds a default"
    _ = shutil.copyfile(fixture, settings_path())

    with caplog.at_level(logging.WARNING):
        loaded = dataclasses.asdict(load_settings())

    assert [record.getMessage() for record in caplog.records] == []
    assert loaded == defaults | written
