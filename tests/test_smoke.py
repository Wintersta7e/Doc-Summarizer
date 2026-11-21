# SPDX-License-Identifier: MIT
"""The GUI entry's ``--smoke-test`` run and the CLI's ``--version``.

A packaged build runs ``--smoke-test`` to prove it starts: the engine, every
screen, the bundled fonts, the settings file and the optional model runtime,
with no model loaded. These tests run that path in-process (so it counts toward
coverage), once through ``run.py`` in a fresh process, and break each part to
show the run then exits non-zero.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.metadata
import importlib.util
import logging
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from PySide6.QtGui import QGuiApplication
from typing_extensions import override

from docsummarizer import cli
from docsummarizer.settings import settings_path
from docsummarizer.ui import app as app_mod

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

_LOG = logging.getLogger(__name__)
_REPO = Path(__file__).resolve().parents[1]
_VERSION = importlib.metadata.version("docsummarizer")
_SMOKE_OK = f"DocSummarizer {_VERSION} smoke ok\n"
# A fresh interpreter importing PySide6 and loading every screen; seconds, not minutes.
_SUBPROCESS_TIMEOUT_S = 60


@pytest.fixture
def qapp() -> QGuiApplication:
    """The process-wide Qt application the engine needs.

    Returns:
        The existing application, or a new one.
    """
    existing = QGuiApplication.instance()
    return existing if isinstance(existing, QGuiApplication) else QGuiApplication([])


class _Runtime(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Stands in for the optional ``llama_cpp`` package: records each import, or fails it."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error: Exception | None = error
        self.imports: int = 0

    @override
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        return importlib.util.spec_from_loader(fullname, self) if fullname == "llama_cpp" else None

    @override
    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return None

    @override
    def exec_module(self, module: ModuleType) -> None:
        self.imports += 1
        if self.error is not None:
            raise self.error


def _install_runtime(monkeypatch: pytest.MonkeyPatch, runtime: _Runtime) -> None:
    monkeypatch.delitem(sys.modules, "llama_cpp", raising=False)
    monkeypatch.setattr(sys, "meta_path", [runtime, *sys.meta_path])


def _copy_qml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the app at a copy of its QML tree, so a test can break one file of it.

    Returns:
        The copy's ``App`` directory.
    """
    qml_dir = tmp_path / "qml"
    _ = shutil.copytree(app_mod._QML_DIR, qml_dir)
    monkeypatch.setattr(app_mod, "_QML_DIR", qml_dir)
    monkeypatch.setattr(app_mod, "_MAIN_QML", qml_dir / "App" / "Main.qml")
    return qml_dir / "App"


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    _ = path.write_text(text.replace(old, new, 1), encoding="utf-8")


@pytest.mark.usefixtures("qapp")
def test_smoke_test_prints_the_version_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert app_mod.main(["DocSummarizer", "--smoke-test"]) == 0

    out, err = capsys.readouterr()
    assert out == _SMOKE_OK
    assert err == ""
    assert settings_path().is_file()


@pytest.mark.usefixtures("qapp")
def test_smoke_test_imports_the_model_runtime_when_it_is_installed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = _Runtime()
    _install_runtime(monkeypatch, runtime)

    assert app_mod.main(["DocSummarizer", "--smoke-test"]) == 0

    assert runtime.imports == 1
    assert capsys.readouterr().out == _SMOKE_OK


@pytest.mark.usefixtures("qapp")
@pytest.mark.parametrize(
    "error",
    [
        OSError("the shared library failed to load"),
        ModuleNotFoundError("a dependency of the runtime is missing", name="numpy"),
    ],
    ids=["library", "dependency"],
)
def test_smoke_test_fails_when_the_model_runtime_does_not_load(
    error: Exception, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_runtime(monkeypatch, _Runtime(error))

    assert app_mod.main(["DocSummarizer", "--smoke-test"]) == 1

    out, err = capsys.readouterr()
    assert out == ""
    assert str(error) in err


@pytest.mark.usefixtures("qapp")
def test_smoke_test_fails_when_a_screen_does_not_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    app_dir = _copy_qml(tmp_path, monkeypatch)
    _edit(app_dir / "ConfigScreen.qml", "import App\n", "import App\nimport NoSuchModule\n")

    assert app_mod.main(["DocSummarizer", "--smoke-test"]) == 1

    out, err = capsys.readouterr()
    _LOG.info("smoke run with a broken screen: %s", err)
    assert out == ""
    assert "Failed to load QML" in err


@pytest.mark.usefixtures("qapp")
def test_smoke_test_fails_on_a_qml_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    app_dir = _copy_qml(tmp_path, monkeypatch)
    _edit(
        app_dir / "Main.qml",
        "Component.onCompleted: Theme.applyMode(win.bridge.appearance)",
        'Component.onCompleted: { Theme.applyMode(win.bridge.appearance); console.warn("probe") }',
    )

    assert app_mod.main(["DocSummarizer", "--smoke-test"]) == 1

    out, err = capsys.readouterr()
    assert out == ""
    assert "probe" in err


@pytest.mark.usefixtures("qapp")
def test_smoke_test_fails_without_the_bundled_fonts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(app_mod, "register_fonts", list)

    assert app_mod.main(["DocSummarizer", "--smoke-test"]) == 1

    out, err = capsys.readouterr()
    assert out == ""
    assert "no bundled fonts" in err


def test_run_py_runs_the_smoke_test_in_a_fresh_process(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "XDG_DATA_HOME": str(tmp_path),
        "LOCALAPPDATA": str(tmp_path),
        "HOME": str(tmp_path),
        "USERPROFILE": str(tmp_path),
        "QT_QPA_PLATFORM": "offscreen",
        "QT_QUICK_BACKEND": "software",
    }
    done = subprocess.run(
        [sys.executable, str(_REPO / "run.py"), "--smoke-test"],
        capture_output=True,
        text=True,
        env=env,
        timeout=_SUBPROCESS_TIMEOUT_S,
        check=False,
    )

    _LOG.info("run.py --smoke-test stderr: %s", done.stderr)
    assert (done.returncode, done.stdout) == (0, _SMOKE_OK)


def test_run_py_exits_with_the_status_main_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    failed = 3
    monkeypatch.setattr(app_mod, "main", lambda: failed)

    with pytest.raises(SystemExit) as exit_info:
        _ = runpy.run_path(str(_REPO / "run.py"), run_name="__main__")

    assert exit_info.value.code == failed


def test_cli_version_prints_the_package_version(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["docsummarizer-cli", "--version"])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"DocSummarizer {_VERSION}\n"
