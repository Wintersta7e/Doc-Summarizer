"""Tests for the evaluation runner under ``scripts/eval``.

Everything here runs without a model: the case matrix, its budgets and the
package's configuration are checked as data, and the runner against fakes.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
import sys
from collections import namedtuple
from pathlib import Path
from typing import Any

import psutil
import pytest
from eval import cases, run

import docsummarizer
from docsummarizer import model_manager, paths
from docsummarizer.model_manager import (
    SUMMARY_TYPE_BRIEF,
    SUMMARY_TYPE_DETAILED,
    SUMMARY_TYPES,
    Summarizer,
)

_REPO = Path(__file__).resolve().parents[1]


def test_tier_matrices_match_the_spec() -> None:
    """The tiers hold exactly the cases the evaluation procedure names, in order."""
    for name, tier in cases.TIERS.items():
        print(f"tier {name}: {len(tier)} cases")

    smoke = cases.TIERS["smoke"]
    assert len(smoke) == 17
    auto_docs = ("zh-lit-s", "zh-mix-s", "de-lit-s", "en-lit-s", "zh-lit-m")
    auto = [
        cases.Case(doc, summary_type, "auto")
        for doc in auto_docs
        for summary_type in (SUMMARY_TYPE_BRIEF, SUMMARY_TYPE_DETAILED)
    ]
    explicit = [
        cases.Case("de-lit-s", SUMMARY_TYPE_DETAILED, "Spanish"),
        cases.Case("zh-lit-s", SUMMARY_TYPE_DETAILED, "German"),
        cases.Case("en-lit-s", SUMMARY_TYPE_DETAILED, "Chinese"),
        cases.Case("en-lit-s", SUMMARY_TYPE_DETAILED, "Traditional Chinese"),
    ]
    m12a = [
        cases.Case("zh-lit-s", SUMMARY_TYPE_BRIEF, "auto", m12a=True),
        cases.Case("zh-lit-s", SUMMARY_TYPE_DETAILED, "auto", m12a=True),
        cases.Case("zh-lit-m", SUMMARY_TYPE_DETAILED, "auto", m12a=True),
    ]
    assert list(smoke) == auto + explicit + m12a

    lang = cases.TIERS["lang"]
    assert len(lang) == 37
    lang_auto = {
        (case.doc, case.summary_type, case.split)
        for case in lang
        if case.language == "auto" and case.doc in cases.LANGUAGE_SET_IDS
    }
    assert lang_auto == {
        (doc, summary_type, split)
        for doc in cases.LANGUAGE_SET_IDS
        for summary_type in SUMMARY_TYPES
        for split in (False, True)
    }
    assert len(cases.LANGUAGE_SET_IDS) == 6
    assert cases.Case("zht-lit-s", SUMMARY_TYPE_DETAILED, "German") in lang

    full = cases.TIERS["full"]
    assert set(smoke) | set(lang) <= set(full)
    corpus_auto = {
        cases.Case(doc, summary_type, "auto")
        for doc in cases.CORPUS_IDS
        for summary_type in SUMMARY_TYPES
    }
    assert len(corpus_auto) == 33
    assert corpus_auto <= set(full)

    for name, tier in cases.TIERS.items():
        ids = [case.id for case in tier]
        assert len(ids) == len(set(ids)), name
        assert all(case.doc in cases.DOCUMENTS for case in tier), name
        assert all(case.summary_type in SUMMARY_TYPES for case in tier), name
        assert set(cases.TIER_TYPES[name]) >= {case.summary_type for case in tier}, name

    assert cases.Case("zh-lit-s", "brief", "auto").id == "zh-lit-s/brief/auto"
    split_id = cases.Case("fr-lit-s", "detailed", "auto", split=True).id
    assert split_id == "fr-lit-s/detailed/auto@split"
    m12a_id = cases.Case("zh-lit-m", "detailed", "auto", m12a=True).id
    assert m12a_id == "zh-lit-m/detailed/auto#m12a"


def test_split_documents_exceed_the_split_budget() -> None:
    """Every document of a split case is longer than one part at ``SPLIT_N_CTX``.

    Catches a split context whose budget a ~2-3k-token document fits in one
    part, so the split pass measures nothing multi-part.
    """
    max_tokens = inspect.signature(Summarizer.summarize).parameters["max_tokens"].default
    prose = cases.SPLIT_N_CTX - max_tokens - model_manager._SCAFFOLD_TOKENS
    json_budget = cases.SPLIT_N_CTX - max_tokens - model_manager._STRUCTURED_SCAFFOLD_TOKENS
    print(f"split budgets at {cases.SPLIT_N_CTX}: prose {prose}, json {json_budget}")

    split_docs = sorted({case.doc for case in cases.TIERS["lang"] if case.split})
    budget = max(prose, json_budget)
    for doc in split_docs:
        margin = cases.DOCUMENTS[doc].tokens - budget
        print(f"{doc}: margin {margin}")
        assert margin > 0, doc
    print(f"documents checked: {len(split_docs)}")
    assert split_docs
    assert cases.SPLIT_N_CTX == 4096


def test_no_real_path_in_harness_files() -> None:
    """No harness file names a real machine's paths."""
    needles = ["/" + "mnt/", "C:" + "\\", "/" + "home/", "\\" + "Users" + "\\"]
    files = sorted(
        [*(_REPO / "scripts" / "eval").glob("*.py"), *(_REPO / "tests").glob("test_eval_*.py")]
    )
    hits = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        hits.extend(f"{path.name}: {needle}" for needle in needles if needle in text)
    print(f"files scanned: {len(files)}; hits: {hits}")
    assert files
    assert hits == []


def test_language_classifier_is_a_core_dependency() -> None:
    """The language classifier is a core dependency, declared once.

    Catches the classifier kept in a temporary extra that a later change must
    delete.
    """
    tomllib = pytest.importorskip("tomllib")
    config = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    project = config["project"]
    matched = [
        entry
        for entry in project["dependencies"]
        if re.split(r"[<>=!~ \[]", entry, maxsplit=1)[0] == "py3langid"
    ]
    print(f"core entry: {matched}")
    assert "eval" not in project.get("optional-dependencies", {})
    assert len(matched) == 1
    overrides = config["tool"]["mypy"]["overrides"]
    modules = [table.get("module", []) for table in overrides]
    listed = sum(([m] if isinstance(m, str) else m).count("py3langid.*") for m in modules)
    assert listed == 1


def test_gated_set_is_derived(tmp_path: Path) -> None:
    """The files whose change needs a measured run follow the pipeline's imports.

    Catches a hand-listed set (roots only), so a newly imported module escapes
    the gate.
    """
    package = Path(docsummarizer.__file__).parent
    real = run.derive_gated(package)
    print(f"gated files: {len(real.files)} {real.files}; missing roots: {real.missing_roots}")
    assert len(real.files) > 0
    assert {"model_manager.py", "provenance.py"} <= set(real.files)
    assert not set(run.NOT_GATED) & set(real.files)
    for root in ("native.py", "hardware.py"):
        if not (package / root).exists():
            assert root in real.missing_roots

    synthetic = tmp_path / "pkg"
    synthetic.mkdir()
    (synthetic / "model_manager.py").write_text(
        "from . import logger\n\n\ndef load():\n    from .extra import x\n\n    return x\n",
        encoding="utf-8",
    )
    (synthetic / "extra.py").write_text("x = 1\n", encoding="utf-8")
    (synthetic / "logger.py").write_text("", encoding="utf-8")
    (synthetic / "provenance.py").write_text("", encoding="utf-8")
    gated = run.derive_gated(synthetic)
    print(f"synthetic gated: {gated.files}")
    assert "extra.py" in gated.files
    assert "logger.py" not in gated.files
    assert gated.missing_roots == ("native.py", "hardware.py")

    before = run.gated_hash(synthetic, gated.files)
    (synthetic / "extra.py").write_text("x = 2\n", encoding="utf-8")
    after_extra = run.gated_hash(synthetic, gated.files)
    assert after_extra != before
    (synthetic / "logger.py").write_text("y = 1\n", encoding="utf-8")
    assert run.gated_hash(synthetic, run.derive_gated(synthetic).files) == after_extra


_Times = namedtuple("_Times", "user system")
_Map = namedtuple("_Map", "path")
_OWN_PID = 999


class _FakeProcess:
    """A process as ``psutil`` reports it, with an invented executable name."""

    def __init__(
        self, pid: int, exe: str, cpu: float, maps: tuple[str, ...] = (), denied: bool = False
    ) -> None:
        self.pid = pid
        self._exe = exe
        self._cpu = cpu
        self._maps = maps
        self._denied = denied

    def memory_maps(self) -> list[Any]:
        if self._denied:
            raise psutil.AccessDenied
        return [_Map(path) for path in self._maps]

    def exe(self) -> str:
        return self._exe

    def cpu_times(self) -> Any:
        return _Times(self._cpu / 2, self._cpu / 2)


def _processes(at_end: bool, only: tuple[int, ...] | None = None) -> list[_FakeProcess]:
    table = [
        (101, "/usr/bin/inkwell-mapper", (2.0, 7.5), ("/x/libllama.so",), False),
        (102, "/opt/x/quill-llama-runner", (0.0, 5.0), (), False),
        (103, "/usr/bin/tallow-idle", (3.0, 3.0), ("/x/libggml-base.so",), False),
        (104, "/usr/bin/sealed-box", (1.0, 1.0), (), True),
        (105, "/usr/bin/kettle-clean", (0.0, 50.0), ("/x/libc.so",), False),
        (_OWN_PID, "/usr/bin/python3", (0.0, 60.0), ("/x/libggml.so",), False),
    ]
    return [
        _FakeProcess(pid, exe, cpu[1] if at_end else cpu[0], maps, denied)
        for pid, exe, cpu, maps, denied in table
        if only is None or pid in only
    ]


def test_contention_counts_llama_maps_and_access_denied() -> None:
    """A model process marks a run only when it used the CPU during that run.

    Catches a presence count (an idle match marks the run), and ``unknown``
    counted toward contention.
    """
    start = run.take_snapshot(_processes(False), _OWN_PID, tasklist=None)
    end = run.take_snapshot(_processes(True), _OWN_PID, tasklist=None)
    result = run.count_contention(start, end)
    print(f"full table: {result}")
    assert result.local.count == 2
    assert result.local.unknown == 1
    assert set(result.local.busy) == {"inkwell-mapper", "quill-llama-runner"}
    assert result.local.idle == ("tallow-idle",)
    assert _OWN_PID not in end.local
    assert result.host is None
    assert result.contended

    idle_only = (103, 104)
    start = run.take_snapshot(_processes(False, idle_only), _OWN_PID, tasklist=None)
    end = run.take_snapshot(_processes(True, idle_only), _OWN_PID, tasklist=None)
    quiet = run.count_contention(start, end)
    print(
        f"idle only: count={quiet.local.count} unknown={quiet.local.unknown} idle={quiet.local.idle}"
    )
    assert quiet.local.count == 0
    assert quiet.local.unknown == 1
    assert quiet.local.idle == ("tallow-idle",)
    assert not quiet.contended


def _tasklist(cpu_time: str) -> str:
    return (
        f'"orchard-llama-host.exe","4242","Console","1","120,000 K","Running","N/A","{cpu_time}","N/A"\n'
        '"ledger-sync.exe","5151","Console","1","20,000 K","Running","N/A","0:00:02","N/A"\n'
    )


def test_contention_counts_host_side(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Windows side of a WSL run is counted by the same CPU rule.

    Catches the host side left out, so a busy model process on the Windows
    side of a WSL run goes uncounted.
    """
    assert run.parse_tasklist(_tasklist("0:00:02")) == {4242: ("orchard-llama-host.exe", 2.0)}
    start = run.take_snapshot([], _OWN_PID, tasklist=lambda: _tasklist("0:00:02"))
    assert start.host_state == "ok"

    busy = run.count_contention(
        start, run.take_snapshot([], _OWN_PID, tasklist=lambda: _tasklist("0:00:07"))
    )
    print(f"(a) host advanced 5 s: {busy.host}")
    assert busy.host is not None
    assert busy.host.count == 1
    assert busy.contended

    idle = run.count_contention(
        start, run.take_snapshot([], _OWN_PID, tasklist=lambda: _tasklist("0:00:02"))
    )
    print(f"(b) host idle: {idle.host}")
    assert idle.host is not None
    assert idle.host.count == 0
    assert idle.host.idle == ("orchard-llama-host.exe",)
    assert not idle.contended

    blind = run.take_snapshot([], _OWN_PID, tasklist=lambda: None)
    unknown = run.count_contention(blind, blind)
    print(f"(c) no interop: host_state={blind.host_state} host={unknown.host}")
    assert blind.host_state == "unknown"
    assert unknown.host is not None
    assert unknown.host.count is None
    assert not unknown.contended

    def refuse() -> str:
        raise AssertionError("tasklist read off WSL")

    monkeypatch.setattr(run, "under_wsl", lambda: False)
    monkeypatch.setattr(run, "read_tasklist", refuse)
    off_wsl = run.snapshot_now()
    print(f"(d) off WSL: host_state={off_wsl.host_state}")
    assert off_wsl.host_state == "n/a"
    assert run.count_contention(off_wsl, off_wsl).host is None


def test_bootstrap_points_app_data_at_the_eval_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """App data and the imported package both come from the eval dir's run.

    Catches a module-level ``from eval import cases``, which resolves
    ``docsummarizer`` before ``bootstrap()`` runs.
    """
    monkeypatch.setattr(sys, "path", list(sys.path))
    app_data = tmp_path / "appdata"
    run.bootstrap(app_data)
    assert paths.app_data_dir("logs").is_relative_to(app_data)
    assert Path(docsummarizer.__file__).resolve().is_relative_to(_REPO / "src")

    tree = ast.parse((_REPO / "scripts" / "eval" / "run.py").read_text(encoding="utf-8"))
    banned = {"eval", "cases", "metrics", "docsummarizer"}
    counted = 0
    hits = []
    for node in tree.body:
        if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING":
            continue
        if isinstance(node, ast.Import):
            counted += 1
            hits += [a.name for a in node.names if a.name.split(".")[0] in banned]
        elif isinstance(node, ast.ImportFrom):
            counted += 1
            module = node.module or ""
            if node.level or module.split(".")[0] in banned:
                hits.append("." * node.level + module)
    print(f"module-level imports counted: {counted}; banned: {hits}")
    assert counted > 0
    assert hits == []


def _write_manifest(eval_dir: Path, docs: list[dict[str, Any]]) -> Path:
    corpus = eval_dir / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    manifest = corpus / "manifest.json"
    manifest.write_text(json.dumps({"documents": docs}), encoding="utf-8")
    return manifest


def _manifest_entry(eval_dir: Path, doc_id: str, text: str, **fields: Any) -> dict[str, Any]:
    corpus = eval_dir / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / f"{doc_id}.txt").write_text(text, encoding="utf-8")
    entry = {
        "id": doc_id,
        "file": f"{doc_id}.txt",
        "label": cases.DOCUMENTS[doc_id].label,
        "source": "project-written",
        "licence": "project",
        "tokens": len(text) // 3,
        "parts": 1,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    entry.update(fields)
    return entry


def test_manifest_absent_unreadable_and_mismatch(tmp_path: Path) -> None:
    def refused(eval_dir: Path, verify: bool = False) -> str:
        def attempt() -> None:
            documents = run.load_manifest(eval_dir)
            if verify:
                run.verify_documents(documents.values())

        with pytest.raises(run.ManifestError) as caught:
            attempt()
        print(f"refused: {caught.value}")
        return str(caught.value)

    assert "absent" in refused(tmp_path / "none")

    as_dir = tmp_path / "as-dir"
    (as_dir / "corpus" / "manifest.json").mkdir(parents=True)
    assert "unreadable" in refused(as_dir)

    bad_json = tmp_path / "bad-json"
    (bad_json / "corpus").mkdir(parents=True)
    (bad_json / "corpus" / "manifest.json").write_text("{not json", encoding="utf-8")
    assert "unreadable" in refused(bad_json)

    text = "Die Bäckerei im Dorf öffnet jeden Morgen um fünf Uhr."
    missing = tmp_path / "missing"
    entry = _manifest_entry(missing, "de-lit-s", text)
    _write_manifest(missing, [entry])
    (missing / "corpus" / "de-lit-s.txt").unlink()
    assert "absent: de-lit-s" in refused(missing, verify=True)

    changed = tmp_path / "changed"
    _write_manifest(changed, [_manifest_entry(changed, "de-lit-s", text, sha256="0" * 64)])
    assert "sha256: de-lit-s" in refused(changed, verify=True)

    relabelled = tmp_path / "relabelled"
    _write_manifest(relabelled, [_manifest_entry(relabelled, "de-lit-s", text, label="fr")])
    assert "label: de-lit-s" in refused(relabelled)

    no_script = tmp_path / "no-script"
    _write_manifest(
        no_script, [_manifest_entry(no_script, "zh-lit-s", "村里的面包店每天清晨开门。")]
    )
    assert "script: zh-lit-s" in refused(no_script)

    good = tmp_path / "good"
    _write_manifest(good, [_manifest_entry(good, "de-lit-s", text)])
    documents = run.load_manifest(good)
    run.verify_documents(documents.values())
    assert documents["de-lit-s"].label == "de"
    assert len(run.corpus_hash(documents.values())) == 64


def test_git_state_distinguishes_missing_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    no_repo = run.git_state(tmp_path)
    print(f"not a repo: {no_repo}")
    assert no_repo.commit is None
    assert no_repo.error
    assert "not a git repository" in no_repo.error

    monkeypatch.setattr(run.shutil, "which", lambda name: None)
    no_git = run.git_state(tmp_path)
    print(f"no git: {no_git}")
    assert no_git.commit is None
    assert no_git.error == "git not found"
