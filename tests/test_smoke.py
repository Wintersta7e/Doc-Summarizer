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
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from PySide6.QtCore import qInstallMessageHandler
from PySide6.QtGui import QGuiApplication
from typing_extensions import override

from docsummarizer import cli
from docsummarizer.settings import Settings, save_settings, settings_path
from docsummarizer.ui import app as app_mod

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import ModuleType

    from PySide6.QtCore import QMessageLogContext, QtMsgType

# What the stand-in application's event loop returns.
_EXEC_STATUS = 7

_LOG = logging.getLogger(__name__)
_REPO = Path(__file__).resolve().parents[1]
_VERSION = importlib.metadata.version("docsummarizer")
_SMOKE_OK = f"DocSummarizer {_VERSION} smoke ok\n"
# A fresh interpreter importing PySide6 and loading every screen; seconds, not minutes.
_SUBPROCESS_TIMEOUT_S = 60
# A settings file as a user's earlier session left it, in a layout save_settings
# never writes (one line), so any write-back shows in its bytes.
_SAVED_SETTINGS = (
    b'{"n_threads": 6, "use_gpu": true, "appearance": "Dark", "output_language": "German"}'
)
_SAVED_MTIME_NS = 1_600_000_000_000_000_000


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


@pytest.mark.usefixtures("qapp")
def test_smoke_result_receives_the_ok_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = tmp_path / "result.txt"

    status = app_mod.main(["DocSummarizer", "--smoke-test", "--smoke-result", str(result)])

    assert status == 0
    assert result.read_text(encoding="utf-8") == _SMOKE_OK
    assert capsys.readouterr().out == _SMOKE_OK


@pytest.mark.usefixtures("qapp")
def test_smoke_result_is_the_proof_when_there_is_no_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A windowed Windows build starts with no stdout at all.
    monkeypatch.setattr(sys, "stdout", None)
    result = tmp_path / "result.txt"

    status = app_mod.main(["DocSummarizer", "--smoke-result", str(result), "--smoke-test"])

    assert status == 0
    assert result.read_text(encoding="utf-8") == _SMOKE_OK


@pytest.mark.usefixtures("qapp")
def test_a_failed_smoke_run_writes_no_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(app_mod, "register_fonts", list)
    result = tmp_path / "result.txt"

    status = app_mod.main(["DocSummarizer", "--smoke-test", "--smoke-result", str(result)])

    assert status == 1
    assert not result.exists()
    assert "no bundled fonts" in capsys.readouterr().err


def test_smoke_result_without_a_path_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    status = app_mod.main(["DocSummarizer", "--smoke-test", "--smoke-result"])

    assert status == app_mod.USAGE_ERROR
    assert "--smoke-result needs a path" in capsys.readouterr().err


def _saved_settings() -> Path:
    """Write the settings file an earlier session left: compact, non-default and old.

    Returns:
        The settings file.
    """
    path = settings_path()
    _ = path.write_bytes(_SAVED_SETTINGS)
    os.utime(path, ns=(_SAVED_MTIME_NS, _SAVED_MTIME_NS))
    return path


def _files_under(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))


def _assert_untouched(saved: Path) -> None:
    """The saved settings file is byte-for-byte and time-for-time what it was, and alone."""
    assert saved.read_bytes() == _SAVED_SETTINGS
    assert saved.stat().st_mtime_ns == _SAVED_MTIME_NS
    assert _files_under(saved.parents[1]) == ["config", "config/settings.json"]


@pytest.mark.usefixtures("qapp")
def test_smoke_test_works_in_a_temp_dir_and_leaves_the_app_data_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    saved = _saved_settings()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    written: list[Path] = []

    def spy(settings: Settings) -> None:
        written.append(settings_path())
        save_settings(settings)

    monkeypatch.setattr(app_mod, "save_settings", spy)

    assert app_mod.main(["DocSummarizer", "--smoke-test"]) == 0

    assert capsys.readouterr().out == _SMOKE_OK
    _assert_untouched(saved)
    assert len(written) == 1
    assert written[0].is_relative_to(scratch)
    assert list(scratch.iterdir()) == []


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
        "Component.onCompleted: { Theme.applyMode(win.bridge.appearance); "
        'console.warn("probe one"); console.warn("probe two") }',
    )

    assert app_mod.main(["DocSummarizer", "--smoke-test"]) == 1

    out, err = capsys.readouterr()
    assert out == ""
    # One line per warning, each led by its category.
    probes = [line.partition(": ") for line in err.splitlines() if "probe" in line]
    assert [message for _, _, message in probes] == ["probe one", "probe two"]
    assert len({category for category, _, _ in probes}) == 1


@pytest.mark.usefixtures("qapp")
def test_smoke_test_fails_without_the_bundled_fonts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(app_mod, "register_fonts", list)

    assert app_mod.main(["DocSummarizer", "--smoke-test"]) == 1

    out, err = capsys.readouterr()
    assert out == ""
    assert "no bundled fonts" in err


@pytest.mark.parametrize(
    ("error", "installed"),
    [(None, True), (ModuleNotFoundError("no llama_cpp", name="llama_cpp"), False)],
    ids=["installed", "not-installed"],
)
def test_the_runtime_check_tells_an_installed_runtime_from_a_missing_one(
    monkeypatch: pytest.MonkeyPatch, error: Exception | None, *, installed: bool
) -> None:
    _install_runtime(monkeypatch, _Runtime(error))

    assert app_mod._import_model_runtime() is installed


@pytest.mark.usefixtures("qapp")
def test_smoke_test_fails_when_the_settings_do_not_round_trip(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    reads = iter([Settings(n_threads=2), Settings(n_threads=3)])

    def load_settings() -> Settings:
        return next(reads)

    monkeypatch.setattr(app_mod, "load_settings", load_settings)

    assert app_mod.main(["DocSummarizer", "--smoke-test"]) == 1

    out, err = capsys.readouterr()
    assert out == ""
    assert "settings did not round-trip through" in err


def _earlier_handler(_mode: QtMsgType, _context: QMessageLogContext, _message: str) -> None:
    """A Qt message handler installed before the smoke run."""


@pytest.mark.usefixtures("qapp")
def test_smoke_test_puts_back_the_qt_message_handler_it_replaced(
    capsys: pytest.CaptureFixture[str],
) -> None:
    original = qInstallMessageHandler(_earlier_handler)
    try:
        assert app_mod.main(["DocSummarizer", "--smoke-test"]) == 0
    finally:
        current = qInstallMessageHandler(original)

    assert current is _earlier_handler
    assert capsys.readouterr().out == _SMOKE_OK


@pytest.mark.usefixtures("qapp")
def test_the_smoke_run_gives_qt_the_command_line_without_its_own_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[list[str]] = []
    running = app_mod._application

    def application(args: list[str]) -> QGuiApplication:
        seen.append(args)
        return running(args)

    monkeypatch.setattr(app_mod, "_application", application)
    result = tmp_path / "result.txt"
    args = ["DocSummarizer", "--smoke-test", "--smoke-result", str(result), "-style", "Basic"]

    assert app_mod.main(args) == 0

    assert seen == [["DocSummarizer", "-style", "Basic"]]
    assert capsys.readouterr().out == _SMOKE_OK


def _font_dir_at_startup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str | None]:
    """Run the smoke test; return ``QT_QPA_FONTDIR`` as the app saw it at creation."""
    seen: list[str | None] = []
    running = app_mod._application

    def application(args: list[str]) -> QGuiApplication:
        seen.append(os.environ.get("QT_QPA_FONTDIR"))
        return running(args)

    monkeypatch.setattr(app_mod, "_application", application)
    result = tmp_path / "result.txt"
    assert app_mod.main(["DocSummarizer", "--smoke-test", "--smoke-result", str(result)]) == 0
    return seen


def test_the_smoke_run_gives_offscreen_qt_the_bundled_fonts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows PySide6 ships no ``lib/fonts``; offscreen Qt warns without a font folder.

    The smoke test fails on any Qt warning, so on a Windows runner it failed
    though the native platform starts clean.
    """
    monkeypatch.setenv("QT_QPA_FONTDIR", "restored-at-teardown")
    monkeypatch.delenv("QT_QPA_FONTDIR")

    (font_dir,) = _font_dir_at_startup(monkeypatch, tmp_path)

    assert font_dir is not None
    assert sorted(Path(font_dir).glob("*.ttf"))


def test_the_smoke_run_keeps_a_font_folder_already_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QT_QPA_FONTDIR", str(tmp_path))

    assert _font_dir_at_startup(monkeypatch, tmp_path) == [str(tmp_path)]


@pytest.mark.usefixtures("qapp")
def test_the_smoke_run_leaves_the_app_logger_only_a_null_handler(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert app_mod.main(["DocSummarizer", "--smoke-test"]) == 0

    handlers = logging.getLogger("DocSummarizer").handlers
    assert handlers
    assert all(type(handler) is logging.NullHandler for handler in handlers)
    assert capsys.readouterr().out == _SMOKE_OK


@pytest.mark.usefixtures("qapp")
def test_main_reads_sys_argv_when_given_no_command_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["DocSummarizer", "--smoke-test"])

    assert app_mod.main() == 0

    assert capsys.readouterr().out == _SMOKE_OK


class _Recorder:
    """Stands in for a Qt object: records each method called on it, with its arguments."""

    def __init__(self, *args: object) -> None:
        self.args: tuple[object, ...] = args
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def __getattr__(self, name: str) -> Callable[..., None]:
        def method(*args: object) -> None:
            self.calls.append((name, args))

        return method

    @staticmethod
    def instance() -> None:
        """No application is running yet."""

    def exec(self) -> int:
        self.calls.append(("exec", ()))
        return _EXEC_STATUS


def test_a_new_application_is_named_for_the_app(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_mod, "QGuiApplication", _Recorder)

    app = app_mod._application(["DocSummarizer", "-style", "Basic"])

    assert isinstance(app, _Recorder)
    assert app.args == (["DocSummarizer", "-style", "Basic"],)
    assert app.calls == [
        ("setApplicationName", ("DocSummarizer",)),
        ("setOrganizationName", ("DocSummarizer",)),
    ]


def test_main_opens_the_window_checks_the_model_and_runs_the_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, bridge = _Recorder(), _Recorder()
    applications: list[list[str]] = []
    engines: list[object] = []
    hooked: list[object] = []

    def application(args: list[str]) -> _Recorder:
        applications.append(args)
        return app

    def create_engine(application: object) -> tuple[object, _Recorder]:
        engines.append(application)
        return object(), bridge

    monkeypatch.setattr(app_mod, "_application", application)
    monkeypatch.setattr(app_mod, "create_engine", create_engine)
    monkeypatch.setattr(app_mod, "_install_excepthook", hooked.append)

    assert app_mod.main(["DocSummarizer"]) == _EXEC_STATUS

    assert applications == [["DocSummarizer"]]
    assert engines == [app]
    assert hooked == [bridge]
    assert bridge.calls == [("checkModel", ())]
    assert app.calls == [("exec", ())]


def test_run_py_runs_the_smoke_test_in_a_fresh_process(tmp_path: Path) -> None:
    saved = _saved_settings()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    env = {
        **os.environ,
        "XDG_DATA_HOME": str(tmp_path),
        "LOCALAPPDATA": str(tmp_path),
        "HOME": str(tmp_path),
        "USERPROFILE": str(tmp_path),
        "TMPDIR": str(scratch),
        "TEMP": str(scratch),
        "TMP": str(scratch),
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
    _assert_untouched(saved)
    assert list(scratch.iterdir()) == []


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
