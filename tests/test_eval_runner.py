"""Tests for the evaluation runner under ``scripts/eval``.

Everything here runs without a model: the case matrix, its budgets and the
package's configuration are checked as data, and the runner against fakes.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
from collections import namedtuple
from pathlib import Path
from typing import Any

import psutil
import pytest
from eval import cases, metrics, run

import docsummarizer
from docsummarizer import model_manager, paths, provenance
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
    start = run.take_snapshot(_processes(False), _OWN_PID, host_processes=None)
    end = run.take_snapshot(_processes(True), _OWN_PID, host_processes=None)
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
    start = run.take_snapshot(_processes(False, idle_only), _OWN_PID, host_processes=None)
    end = run.take_snapshot(_processes(True, idle_only), _OWN_PID, host_processes=None)
    quiet = run.count_contention(start, end)
    print(
        f"idle only: count={quiet.local.count} unknown={quiet.local.unknown} idle={quiet.local.idle}"
    )
    assert quiet.local.count == 0
    assert quiet.local.unknown == 1
    assert quiet.local.idle == ("tallow-idle",)
    assert not quiet.contended


def _host_table(cpu_seconds: str) -> str:
    """The host's process table as the Windows query prints it: a header, then rows.

    The last row is a protected process, whose CPU cell is empty.
    """
    return (
        '"Id","ProcessName","CPU"\n'
        f'"4242","orchard-llama-host","{cpu_seconds}"\n'
        '"5151","ledger-sync","2.5"\n'
        '"6363","sealed-llama-guard",\n'
    )


def test_host_process_table_reads_float_seconds() -> None:
    """CPU is float seconds; the header and an empty CPU cell are skipped.

    Catches a parser still reading an ``H:MM:SS`` clock, which reads no row
    of the host's table, so the host side counts nothing.
    """
    parsed = run.parse_host_processes(_host_table("2.25"))
    print(f"parsed host table: {parsed}")
    assert parsed == {4242: ("orchard-llama-host", 2.25)}


def test_contention_counts_host_side(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Windows side of a WSL run is counted by the same CPU rule.

    Catches the host side left out, so a busy model process on the Windows
    side of a WSL run goes uncounted.
    """
    start = run.take_snapshot([], _OWN_PID, host_processes=lambda: _host_table("2.0"))
    assert start.host_state == "ok"

    busy = run.count_contention(
        start, run.take_snapshot([], _OWN_PID, host_processes=lambda: _host_table("7.25"))
    )
    print(f"(a) host advanced 5 s: {busy.host}")
    assert busy.host is not None
    assert busy.host.count == 1
    assert busy.contended

    idle = run.count_contention(
        start, run.take_snapshot([], _OWN_PID, host_processes=lambda: _host_table("2.0"))
    )
    print(f"(b) host idle: {idle.host}")
    assert idle.host is not None
    assert idle.host.count == 0
    assert idle.host.idle == ("orchard-llama-host",)
    assert not idle.contended

    blind = run.take_snapshot([], _OWN_PID, host_processes=lambda: None)
    unknown = run.count_contention(blind, blind)
    print(f"(c) no interop: host_state={blind.host_state} host={unknown.host}")
    assert blind.host_state == "unknown"
    assert unknown.host is not None
    assert unknown.host.count is None
    assert not unknown.contended

    def refuse() -> str:
        raise AssertionError("host process table read off WSL")

    monkeypatch.setattr(run, "under_wsl", lambda: False)
    monkeypatch.setattr(run, "read_host_processes", refuse)
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


# --------------------------------------------------------------------------- #
# The case loop, against a fake model
# --------------------------------------------------------------------------- #

_DOC_MARKER = "Document:\n"
_SINGLE_TEXT = (
    "The river ferry leaves the wooden pier at six every morning. "
    "Its captain checks the ropes and the engine before anyone boards. "
    "Students ride it to the school on the far bank."
)
_MULTI_TEXT = " ".join(
    f"In week {i} the school garden grew beans, carrots and tall yellow sunflowers."
    for i in range(120)
)
_LONG_RESPONSE = "\n  " + ("The ferry crosses the river twice an hour. " * 120)[:5000] + "  \n"
_PROSE = "The village bakery opens early every morning and sells fresh bread to everyone."


def _document_of(messages: list[dict[str, str]]) -> str:
    user = messages[-1]["content"]
    body = user.split(_DOC_MARKER, 1)[-1]
    return body.rsplit("\n\n", 1)[0]


def _default_respond(messages: list[dict[str, str]], kwargs: dict[str, Any]) -> str:
    """Valid Detailed/Structured JSON when asked for JSON, else prose."""
    if "response_format" in kwargs:
        sentences = provenance.split_sentences(_document_of(messages))
        doc = _document_of(messages)
        quote = doc[sentences[0][0] : sentences[0][1]] if sentences else ""
        point = {"text": "The ferry runs every morning across the river.", "quote": quote}
        sections = {name: [point] for name in model_manager._STRUCTURED_SECTIONS}
        lead = "A ferry serves the village and its school every single day."
        return json.dumps({"lead": lead, "points": [point], "sections": sections})
    if _SINGLE_TEXT in messages[-1]["content"]:
        return _LONG_RESPONSE
    return _PROSE


class _EvalLLM:
    """A model stand-in: records ``reset`` / ``set_seed`` and every completion in ``events``."""

    def __init__(self, respond: Any = None) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.completions = 0
        self.respond = respond or _default_respond

    def tokenize(self, text: bytes, add_bos: bool = True) -> list[int]:
        return [0] * (len(text) // 3)

    def reset(self) -> None:
        self.events.append(("reset",))

    def set_seed(self, seed: int) -> None:
        self.events.append(("set_seed", seed))

    def create_chat_completion(self, *, messages, stream=False, **kwargs):
        self.completions += 1
        self.events.append(("call",))
        text = self.respond(messages, kwargs)
        if stream:
            return self._stream(text)
        return {
            "choices": [
                {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
            ]
        }

    def _stream(self, text: str):
        half = len(text) // 2
        for piece in (text[:half], text[half:]):
            yield {"choices": [{"delta": {"content": piece}, "finish_reason": None}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}


def _shell(fake: Any, n_ctx: int) -> Summarizer:
    """A Summarizer around ``fake`` without running ``__init__`` (no llama-cpp needed)."""
    s = object.__new__(Summarizer)
    s.llm = fake
    s.n_ctx = n_ctx
    return s


@pytest.fixture
def eval_llm(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Every ``build_summarizer`` returns a shell around one shared fake, at the asked context."""
    fake = _EvalLLM()
    built: list[int] = []

    def build(model: Path, *, n_ctx: int, n_threads: int | None, n_gpu_layers: int) -> Summarizer:
        built.append(n_ctx)
        return _shell(fake, n_ctx)

    monkeypatch.setattr(run, "build_summarizer", build)
    monkeypatch.setattr(run, "snapshot_now", _empty_snapshot)
    monkeypatch.setattr(sys, "path", list(sys.path))
    fake.built = built
    return fake


def _empty_snapshot() -> Any:
    return run.ContentionSnapshot({}, 0, None, "n/a")


def test_case_loop_captures_every_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every completion of every case run is kept in full, before any clean-up.

    Catches capturing the post-processed return of ``_complete``, or only the
    final call.
    """
    import socket

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    snapshots: list[int] = []

    def counted() -> Any:
        snapshots.append(1)
        return _empty_snapshot()

    monkeypatch.setattr(run, "snapshot_now", counted)

    fake = _EvalLLM()
    summarizer = _shell(fake, 2048)
    specs = [
        run.RunSpec(cases.Case(doc, summary_type, "auto"), 0)
        for doc in ("single", "multi")
        for summary_type in SUMMARY_TYPES
    ]
    texts = {"single": _SINGLE_TEXT, "multi": _MULTI_TEXT}
    targets = {spec.case.id: metrics.Target("en", None) for spec in specs}
    runs = run.run_cases(summarizer, specs, texts, targets)

    segments: list[list[tuple[Any, ...]]] = []
    for event in fake.events:
        if event == ("reset",):
            segments.append([])
        segments[-1].append(event)
    assert len(segments) == len(specs)
    for case_run, segment in zip(runs, segments, strict=True):
        made = sum(1 for event in segment if event == ("call",))
        print(
            f"{case_run.case}: captured {len(case_run.calls)}, made {made}, parts {case_run.parts}"
        )
        assert segment[:2] == [("reset",), ("set_seed", 0)]
        assert len(case_run.calls) == made
        assert case_run.error is None
        assert case_run.repetition == 0
        assert case_run.contention["contended"] is False
        assert case_run.contended is False
    total = sum(len(case_run.calls) for case_run in runs)
    print(f"captured {total} calls, the fake made {fake.completions}")
    assert total == fake.completions
    assert len(snapshots) == 2 * len(specs)

    contents = [call.content for case_run in runs for call in case_run.calls]
    assert _LONG_RESPONSE in contents
    assert all(case_run.parts > 1 for case_run in runs if case_run.case.startswith("multi/"))
    assert all(case_run.parts == 1 for case_run in runs if case_run.case.startswith("single/"))

    sources = sorted((_REPO / "scripts" / "eval").glob("*.py"))
    sites = sum(len(re.findall(r"\bSummarizer\(", p.read_text(encoding="utf-8"))) for p in sources)
    print(f"Summarizer construction sites in scripts/eval: {sites}")
    assert sites == 1


_EN_UNITS = (
    "The village bakery opens every morning at five o'clock. "
    "The baker first heats the oven and then kneads the dough."
)
_DE_UNITS = (
    "Die Bäckerei im Dorf öffnet jeden Morgen um fünf Uhr. "
    "Der Bäcker heizt zuerst den Ofen und knetet dann den Teig."
)


def _call(seq: int, content: str, finish: str = "stop", json_call: bool = False) -> Any:
    kwargs = {"response_format": {"type": "json_object"}} if json_call else {}
    return run.CallRecord(seq, [], kwargs, False, content, finish, None, None, None)


def test_score_case_run_scores_every_call() -> None:
    brief = cases.Case("de-lit-s", SUMMARY_TYPE_BRIEF, "auto")
    result = model_manager.StructuredSummary(SUMMARY_TYPE_BRIEF, _DE_UNITS, [], None, _DE_UNITS)
    calls = [_call(0, _EN_UNITS), _call(1, _DE_UNITS, finish="length"), _call(2, "")]
    scored = run.score_case_run(
        brief, result, calls, _DE_UNITS, metrics.Target("de", None), parts=2
    )
    print(f"brief: {scored}")
    assert scored["M2"]["passed"] is False
    assert {off["call"] for off in scored["M2"]["off_target"]} == {0}
    assert scored["M1"]["violations"] == 1
    assert scored["M1"]["failed_calls"] == [2]
    assert scored["M4"]["violations"] == 1
    assert scored["M4"]["truncated_calls"] == [1]
    assert scored["M7"] is None
    assert scored["cell"] == ["de", "prose", "multi"]

    document = "Die Fähre fährt jeden Morgen um sechs Uhr über den Fluss. Sie ist sehr alt."
    quote = "Die Fähre fährt jeden Morgen um sechs Uhr über den Fluss."
    point_text = "Die Fähre fährt jeden Morgen früh über den breiten Fluss."
    payload = {"lead": _DE_UNITS, "points": [{"text": point_text, "quote": quote}]}
    span = provenance.SourceSpan(0, len(quote), quote, 1.0)
    detailed_result = model_manager.StructuredSummary(
        SUMMARY_TYPE_DETAILED,
        _DE_UNITS,
        [model_manager.SummaryPoint(point_text, span)],
        None,
        "rendered text is never scored",
    )
    detailed = cases.Case("de-lit-s", SUMMARY_TYPE_DETAILED, "auto")
    json_calls = [
        _call(0, "not json", json_call=True),
        _call(1, json.dumps(payload), json_call=True),
    ]
    scored = run.score_case_run(
        detailed, detailed_result, json_calls, document, metrics.Target("de", None), parts=1
    )
    print(f"detailed: {scored}")
    assert scored["M2"]["unparsed_calls"] == 1
    assert scored["M2"]["passed"] is True
    assert scored["M7"]["points"] == 1
    assert scored["M7"]["cited"] == 1
    assert scored["M7"]["verbatim"] == 1
    assert scored["cell"] == ["de", "json", "single"]


def test_m12a_rerun_compares_with_the_first_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run, "snapshot_now", _empty_snapshot)
    first = cases.Case("zh-lit-s", SUMMARY_TYPE_BRIEF, "auto")
    rerun = cases.Case("zh-lit-s", SUMMARY_TYPE_BRIEF, "auto", m12a=True)
    specs = [run.RunSpec(first, 0), run.RunSpec(rerun, 0)]
    texts = {"zh-lit-s": "村里的面包店每天清晨五点开门。店主会先把炉子烧热。然后开始揉面。"}
    targets = {case.id: metrics.Target("zh", None) for case in (first, rerun)}

    same = run.run_cases(_shell(_EvalLLM(), 8192), specs, texts, targets)
    print(f"identical: {same[1].metrics['M12a']}")
    assert same[1].metrics["M12a"]["identical"] is True
    assert same[1].metrics["M12a"]["violations"] == 0
    assert same[0].metrics["M12a"] is None

    answers: list[str] = []

    def drifting(messages: list[dict[str, str]], kwargs: dict[str, Any]) -> str:
        answers.append(_PROSE if not answers else _PROSE.replace("early", "earlz"))
        return answers[-1]

    drifted = run.run_cases(_shell(_EvalLLM(drifting), 8192), specs, texts, targets)
    m12a = drifted[1].metrics["M12a"]
    print(f"first_diff: {m12a['first_diff']}")
    assert m12a["identical"] is False
    assert m12a["first_diff"] == _PROSE.index("early") + 4
    assert m12a["violations"] == 1


# --------------------------------------------------------------------------- #
# The tier CLI
# --------------------------------------------------------------------------- #

_SENTENCES = {
    "zh": "村里的面包店每天清晨五点开门。",
    "de": "Die Fähre fährt jeden Morgen um sechs Uhr über den breiten Fluss. ",
    "en": "The river ferry leaves the wooden pier at six every morning. ",
    "fr": "Le bac quitte le ponton en bois chaque matin à six heures. ",
    "es": "El transbordador sale del muelle de madera cada mañana a las seis. ",
    "ru": "Паром каждое утро отходит от деревянной пристани в шесть часов. ",
    "ja": "村のパン屋は毎朝五時に店を開けます。",
}
_SMOKE_DOCS = ("zh-lit-s", "zh-mix-s", "de-lit-s", "en-lit-s", "zh-lit-m")


def _corpus(tmp_path: Path, ids: tuple[str, ...], long: tuple[str, ...] = ()) -> Path:
    """An eval dir with project-written documents and their manifest.

    zh texts are long enough to split at ``--n-ctx 2048``; a ``long`` document
    also splits at ``SPLIT_N_CTX``.
    """
    entries = []
    for doc_id in ids:
        shape = cases.DOCUMENTS[doc_id]
        repeats = 160 if doc_id in long else (40 if shape.label == "zh" else 3)
        text = _SENTENCES[shape.label] * repeats
        script = shape.script or "Simplified"
        fields = {"script": script} if shape.label == "zh" else {}
        entries.append(_manifest_entry(tmp_path, doc_id, text, **fields))
    _write_manifest(tmp_path, entries)
    return tmp_path


def _gguf(tmp_path: Path) -> Path:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF stand-in")
    return model


def _results(eval_dir: Path) -> list[dict[str, Any]]:
    files = sorted((eval_dir / "results").iterdir())
    assert len(files) == 1, files
    return [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]


def _case_runs(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [line for line in lines if line["kind"] == "case_run"]


def test_cli_smoke_tier_writes_a_complete_results_file(
    eval_llm: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    eval_dir = _corpus(tmp_path / "eval", _SMOKE_DOCS)
    argv = ["--eval-dir", str(eval_dir), "--tier", "smoke", "--model", str(_gguf(tmp_path))]
    code = run.main([*argv, "--n-ctx", "2048"])
    out = capsys.readouterr().out
    print(out)
    assert code == 0

    files = sorted((eval_dir / "results").iterdir())
    print(f"results files: {[f.name for f in files]}")
    assert len(files) == 1
    assert re.fullmatch(r"\d{8}T\d{6}Z-smoke-[0-9a-f]+\.jsonl", files[0].name)
    lines = _results(eval_dir)
    header = lines[0]
    assert header["kind"] == "header"
    assert set(run.HEADER_FIELDS) <= set(header), set(run.HEADER_FIELDS) - set(header)
    assert len(_case_runs(lines)) == 17
    assert sum(1 for line in lines if line["kind"] == "call") == eval_llm.completions
    assert header["n_ctx"] == 2048
    assert header["tier"] == "smoke"

    for column in ("M1", "M2", "M4", "M7", "M12a"):
        assert column in out
    assert re.search(r"cases=17 case_runs=17 calls=[1-9]\d* scored_units=\d+", out)
    assert not (tmp_path / "DocSummarizer").exists()
    assert list((eval_dir / "appdata").rglob("*.log"))


def test_cli_refuses_zero_and_bad_selectors(
    eval_llm: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    eval_dir = _corpus(tmp_path / "eval", _SMOKE_DOCS)
    model = _gguf(tmp_path)
    base = ["--eval-dir", str(eval_dir), "--model", str(model), "--n-ctx", "2048"]

    def refused(argv: list[str]) -> str:
        code = run.main(argv)
        out = capsys.readouterr().out
        print(f"{argv[len(base) :]} -> {code}: {out.strip()}")
        assert code == 2
        return out

    assert "nope" in refused([*base, "--tier", "smoke", "--case", "nope"])
    assert "seed 0 only until the seed API lands" in refused(
        [*base, "--tier", "smoke", "--seeds", "1"]
    )

    with pytest.raises(SystemExit) as parse_error:
        run.main([*base, "--tier", "full"])
    err = capsys.readouterr().err
    print(f"--tier full -> {parse_error.value.code}: {err.strip()}")
    assert parse_error.value.code == 2
    assert all(tier in err for tier in run.RUNNABLE_TIERS)

    empty = tmp_path / "empty"
    empty.mkdir()
    absent = ["--eval-dir", str(empty), "--model", str(model), "--tier", "smoke"]
    assert "absent" in refused(absent)

    doc = tmp_path / "f.txt"
    doc.write_text(_SENTENCES["de"] * 3, encoding="utf-8")
    assert "--label" in refused([*base, "--tier", "smoke", "--doc", str(doc)])

    def no_tokens(text: bytes, add_bos: bool = True) -> list[int]:
        raise RuntimeError("tokenizer unavailable")

    monkeypatch.setattr(eval_llm, "tokenize", no_tokens)
    assert "calls=0" in refused([*base, "--tier", "smoke"])


def test_cli_case_and_doc_restrict_the_run(
    eval_llm: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = _gguf(tmp_path)

    def selected(name: str, tier: str, ids: tuple[str, ...], *extra: str) -> list[dict[str, Any]]:
        eval_dir = _corpus(tmp_path / name, ids)
        argv = ["--eval-dir", str(eval_dir), "--model", str(model), "--tier", tier, *extra]
        assert run.main([*argv, "--n-ctx", "2048"]) == 0
        capsys.readouterr()
        lines = _results(eval_dir)
        print(f"{extra}: {[line['case'] for line in _case_runs(lines)]}")
        return lines

    by_doc = _case_runs(selected("doc", "lang", cases.LANGUAGE_SET_IDS, "--case", "de-cred-s"))
    assert len(by_doc) == 6
    assert {line["case"] for line in by_doc} == {
        f"de-cred-s/{t}/auto{suffix}" for t in SUMMARY_TYPES for suffix in ("", "@split")
    }

    exact = ("--case", "de-cred-s/brief/auto")
    one = _case_runs(selected("one", "lang", cases.LANGUAGE_SET_IDS, *exact))
    assert [line["case"] for line in one] == ["de-cred-s/brief/auto"]
    twin = (*exact, "--case", "de-cred-s/brief/auto@split")
    both = _case_runs(selected("twin", "lang", cases.LANGUAGE_SET_IDS, *twin))
    assert [line["case"] for line in both] == [
        "de-cred-s/brief/auto",
        "de-cred-s/brief/auto@split",
    ]

    adhoc = tmp_path / "f.txt"
    adhoc.write_text(_SENTENCES["de"] * 3, encoding="utf-8")
    sha = hashlib.sha256(adhoc.read_bytes()).hexdigest()
    lines = selected("adhoc", "smoke", _SMOKE_DOCS, "--doc", str(adhoc), "--label", "de")
    runs = _case_runs(lines)
    assert [line["case"] for line in runs] == [
        f"{sha[:12]}/{t}/auto" for t in (SUMMARY_TYPE_BRIEF, SUMMARY_TYPE_DETAILED)
    ]
    expected = hashlib.sha256(f"{sha[:12]}:{sha}\n".encode()).hexdigest()
    assert lines[0]["corpus_hash"] == expected
    assert lines[0]["documents"][sha[:12]]["path"] == str(adhoc.resolve())


def test_repeat_scopes_to_cold_auto_runs(
    eval_llm: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only the seed-0 ``auto`` runs that timing reads are repeated.

    Catches repeating the whole tier, so warm, explicit-language and M12a runs
    are paid for N times.
    """
    smoke = cases.TIERS["smoke"]
    runs = run.expand_runs(smoke, seeds=(0, 1), repeat=3)
    auto = [c for c in smoke if c.language == "auto" and not c.m12a]
    seed0_auto = [r for r in runs if r.seed == 0 and r.case in auto]
    seed1 = [r for r in runs if r.seed == 1]
    explicit = [r for r in runs if r.case.language != "auto"]
    m12a = [r for r in runs if r.case.m12a]
    print(
        f"total {len(runs)}: seed-0 auto {len(seed0_auto)}, seed-1 auto {len(seed1)}, "
        f"explicit {len(explicit)}, m12a {len(m12a)}"
    )
    assert len(runs) == 47
    assert (len(seed0_auto), len(seed1), len(explicit), len(m12a)) == (30, 10, 4, 3)
    assert runs[-3:] == m12a
    assert all(r.seed == 0 and r.repetition == 0 for r in [*explicit, *m12a])
    for case in auto:
        own = [(r.seed, r.repetition) for r in runs if r.case == case]
        start = next(i for i, r in enumerate(runs) if r.case == case)
        assert own == [(0, 0), (0, 1), (0, 2), (1, 0)]
        assert [(r.seed, r.repetition) for r in runs[start : start + 4]] == own

    eval_dir = _corpus(tmp_path / "eval", _SMOKE_DOCS)
    argv = ["--eval-dir", str(eval_dir), "--model", str(_gguf(tmp_path)), "--tier", "smoke"]
    assert run.main([*argv, "--n-ctx", "2048", "--repeat", "3"]) == 0
    out = capsys.readouterr().out
    lines = _results(eval_dir)
    print(f"case_run lines: {len(_case_runs(lines))}")
    assert len(_case_runs(lines)) == 37
    assert lines[0]["repeat"] == 3
    assert lines[0]["repeat_scope"] == run.REPEAT_SCOPE
    assert "cases=17 case_runs=37" in out

    with pytest.raises(SystemExit) as parse_error:
        run.main([*argv, "--repeat", "0"])
    assert parse_error.value.code == 2


def test_lang_tier_runs_the_split_pass(
    eval_llm: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    eval_dir = _corpus(tmp_path / "eval", cases.LANGUAGE_SET_IDS, long=("de-cred-s",))
    argv = ["--eval-dir", str(eval_dir), "--model", str(_gguf(tmp_path)), "--tier", "lang"]
    selection = ["--case", "de-cred-s", "--case", "fr-lit-s"]
    assert run.main([*argv, *selection, "--n-ctx", "2048"]) == 0
    out = capsys.readouterr().out
    print(out)
    print(f"builds: {eval_llm.built}")
    assert eval_llm.built == [2048, cases.SPLIT_N_CTX]

    lines = _results(eval_dir)
    header = lines[0]
    assert header["split_n_ctx"] == cases.SPLIT_N_CTX
    split_runs = [line for line in _case_runs(lines) if line["case"].endswith("@split")]
    assert len(split_runs) == 6
    assert {line["n_ctx"] for line in split_runs} == {cases.SPLIT_N_CTX}
    expected = [f"fr-lit-s/{t}/auto@split" for t in SUMMARY_TYPES]
    assert header["not_split"] == expected
    for case_id in expected:
        assert f"not split: {case_id}" in out
    assert all(line["parts"] > 1 for line in split_runs if line["case"].startswith("de-cred-s"))


# --------------------------------------------------------------------------- #
# Rescore by replay
# --------------------------------------------------------------------------- #


def _source(eval_dir: Path, model: Path, *extra: str) -> Path:
    """Run the smoke tier at ``--n-ctx 2048`` and return its results file."""
    argv = ["--eval-dir", str(eval_dir), "--model", str(model), "--tier", "smoke"]
    assert run.main([*argv, "--n-ctx", "2048", *extra]) == 0
    files = sorted((eval_dir / "results").glob("*.jsonl"))
    assert len(files) == 1, files
    return files[0]


def _rescored(eval_dir: Path) -> Path:
    files = sorted((eval_dir / "results").glob("*-rescore.jsonl"))
    assert len(files) == 1, files
    return files[0]


def _rescore(eval_dir: Path, source: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = run.main(["--eval-dir", str(eval_dir), "--rescore", str(source)])
    out = capsys.readouterr().out
    print(out)
    return code, out


def test_rescore_reproduces_every_metric(
    eval_llm: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Replaying a file through the tree that wrote it reproduces every metric and result.

    Catches a rescore that scores differently from the run: a lost call, another
    ``n_ctx`` or parts count, or M12a compared with the wrong run.
    """
    eval_dir = _corpus(tmp_path / "eval", _SMOKE_DOCS)
    source = _source(eval_dir, _gguf(tmp_path))
    capsys.readouterr()
    code, out = _rescore(eval_dir, source, capsys)
    assert code == 0

    before = run.read_results(source)
    after = run.read_results(_rescored(eval_dir))
    assert len(after.case_runs) == len(before.case_runs) == 17
    assert any(line["parts"] > 1 for line in before.case_runs)
    assert any(line["metrics"]["M12a"] for line in before.case_runs)
    for old, new in zip(before.case_runs, after.case_runs, strict=True):
        assert new["case"] == old["case"]
        assert new["metrics"] == old["metrics"], old["case"]
        assert new["result"] == old["result"], old["case"]
        assert new["run_stats"] == old["run_stats"]
        assert new["repetition"] == old["repetition"]
    assert after.calls == before.calls

    header = after.header
    for name in run.HEADER_FIELDS:
        assert header[name] == before.header[name], name
    git = run.git_state(run._REPO)
    assert set(header["scored_by"]) == {"commit", "dirty", "gated_hash"}
    assert header["scored_by"]["commit"] == git.commit
    assert header["rescored_from"] == str(source)
    assert header["arms"] == list(run.RESCORE_ARMS)
    assert f"replayed case_runs=17 calls={len(before.calls)}" in out


_ARMS_SENTENCES = (
    "The village bakery opens at five every morning.",
    "The baker heats the old stone oven long before dawn.",
)
_ARMS_LONG = (
    "On the first warm day of spring the children of the school garden planted long rows "
    "of beans, carrots, onions and tall yellow sunflowers beside the old stone wall."
)
_ARMS_UNRELATED = "The ferry leaves the wooden pier at noon."
_ARMS_DOC = " ".join((*_ARMS_SENTENCES, _ARMS_LONG, _ARMS_UNRELATED))
_ARMS_DROPPED = " ".join(_ARMS_LONG.split()[::2])  # every second word dropped
_ARMS_POINTS = (
    ("The bakery opens at five.", _ARMS_SENTENCES[0]),
    ("The oven is heated before dawn.", _ARMS_SENTENCES[1]),
    ("Children planted a spring garden by the wall.", _ARMS_DROPPED),
)
_ARMS_CASE = "en-lit-s/detailed/auto"


def _arms_respond(messages: list[dict[str, str]], kwargs: dict[str, Any]) -> str:
    if "response_format" in kwargs:
        points = [{"text": text, "quote": quote} for text, quote in _ARMS_POINTS]
        lead = "A village bakery, its oven and a school garden in spring."
        return json.dumps({"lead": lead, "points": points})
    return _PROSE


def _arms_corpus(eval_dir: Path) -> Path:
    _write_manifest(eval_dir, [_manifest_entry(eval_dir, "en-lit-s", _ARMS_DOC)])
    return eval_dir


def _arms_source(eval_llm: Any, tmp_path: Path) -> Path:
    """A one-case run whose three points cite two verbatim quotes and one loose one."""
    eval_llm.respond = _arms_respond
    return _source(_arms_corpus(tmp_path / "eval"), _gguf(tmp_path), "--case", _ARMS_CASE)


def test_rescore_prints_coverage_only_and_production_arms(
    eval_llm: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Both arms print per cell and in the aggregate, with the cited share the strict check costs.

    Acceptance: the coverage-only arm is the located set before the strict check,
    the production arm the set after it. Mutation caught: the coverage-only arm
    replayed without its patch, so before and after print the same share.
    """
    assert provenance.is_verified_match(_ARMS_DROPPED, _ARMS_LONG) is False
    assert provenance.locate_quote(_ARMS_DROPPED, _ARMS_DOC) is None
    with run.accept_every_match():
        located = provenance.locate_quote(_ARMS_DROPPED, _ARMS_DOC)
    print(f"located under accept_every_match: {located}")
    assert located is not None
    assert located.quote == _ARMS_LONG

    source = _arms_source(eval_llm, tmp_path)
    capsys.readouterr()

    def arms(eval_dir: Path) -> tuple[str, dict[str, Any]]:
        code, out = _rescore(eval_dir, source, capsys)
        assert code == 0
        lines = run.read_results(_rescored(eval_dir)).case_runs
        assert all("m7_coverage_only" in line for line in lines)
        return out, next(line for line in lines if line["case"] == _ARMS_CASE)

    out, line = arms(tmp_path / "eval")
    cited = (line["m7_coverage_only"]["cited"], line["metrics"]["M7"]["cited"])
    print(f"cited (coverage_only, production): {cited}")
    assert line["metrics"]["M7"]["points"] == 3
    assert cited == (3, 2)
    for scope in ("en/json/single", "aggregate"):
        assert f"M7 {scope} coverage_only cited=3/3 tight=3/3 whole_document=0/3" in out
        assert f"M7 {scope} production cited=2/3 tight=2/3 whole_document=0/3" in out
        assert f"verbatim=2/3\nM7 {scope} drop cited=1/3" in out

    monkeypatch.setattr(provenance, "is_verified_match", lambda quote, span_text: False)
    out, line = arms(_arms_corpus(tmp_path / "rejecting"))
    cited = (line["m7_coverage_only"]["cited"], line["metrics"]["M7"]["cited"])
    print(f"rejecting everything, cited (coverage_only, production): {cited}")
    assert cited == (3, 0)
    assert "M7 aggregate drop cited=3/3" in out


def test_rescore_refuses_a_different_request_sequence(
    eval_llm: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Replay stops when this tree would ask the model something the stored run did not.

    Catches a replay that hands out the stored answers without comparing requests.
    """
    eval_dir = _corpus(tmp_path / "eval", _SMOKE_DOCS)
    source = _source(eval_dir, _gguf(tmp_path), "--case", "zh-lit-m")
    capsys.readouterr()
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    calls = [i for i, record in enumerate(records) if record["kind"] == "call"]

    def rewritten(name: str, kept: list[dict[str, Any]]) -> Path:
        path = tmp_path / name
        path.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
        return path

    edited = json.loads(json.dumps(records))
    target = edited[calls[1]]
    target["messages"][0]["content"] += " Answer in verse."
    code, out = _rescore(eval_dir, rewritten("edited.jsonl", edited), capsys)
    assert code == 2
    assert f"replay mismatch: seq {target['seq']} field messages" in out

    dropped = [record for i, record in enumerate(records) if i != calls[-1]]
    code, out = _rescore(eval_dir, rewritten("dropped.jsonl", dropped), capsys)
    assert code == 2
    assert f"replay mismatch: seq {records[calls[-1]]['seq']} field call count" in out


class _ImportRecorder:
    """A ``sys.meta_path`` finder that records imports of ``llama_cpp`` and finds nothing."""

    def __init__(self) -> None:
        self.names: list[str] = []

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
        if name.partition(".")[0] == "llama_cpp":
            self.names.append(name)


def test_rescore_loads_no_model(
    eval_llm: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    eval_dir = _corpus(tmp_path / "eval", _SMOKE_DOCS)
    source = _source(eval_dir, _gguf(tmp_path), "--case", "zh-lit-m")
    capsys.readouterr()

    def no_model(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a rescore built a model")

    recorder = _ImportRecorder()
    monkeypatch.setattr(run, "build_summarizer", no_model)
    monkeypatch.delitem(sys.modules, "llama_cpp", raising=False)
    monkeypatch.setattr(sys, "meta_path", [recorder, *sys.meta_path])
    code, _ = _rescore(eval_dir, source, capsys)
    print(f"llama_cpp imports: {recorder.names}")
    assert code == 0
    assert recorder.names == []


def test_rescore_audits_every_citation(
    eval_llm: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every production citation must verifiably hold its quote and be the document's text.

    Catches grounding that attaches a span the quote is not in.
    """
    source = _arms_source(eval_llm, tmp_path)
    capsys.readouterr()
    code, out = _rescore(tmp_path / "eval", source, capsys)
    assert code == 0
    assert "audit: 2 citations, 0 failed" in out  # the production arm's two, not the three

    start = _ARMS_DOC.index(_ARMS_UNRELATED)
    unrelated = provenance.SourceSpan(start, start + len(_ARMS_UNRELATED), _ARMS_UNRELATED, 1.0)
    # The grounding step reads the name it imported from provenance.
    monkeypatch.setattr(model_manager, "locate_quote", lambda *args, **kwargs: unrelated)
    code, out = _rescore(_arms_corpus(tmp_path / "misplaced"), source, capsys)
    assert code == 1
    for index in range(3):
        assert f"audit: {_ARMS_CASE} point {index}" in out
    assert "audit: 3 citations, 3 failed" in out


def test_runner_logs_stay_in_the_eval_dir(
    eval_llm: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source(_corpus(tmp_path / "eval", _SMOKE_DOCS), _gguf(tmp_path), "--case", "zh-lit-m")
    eval_dir = _corpus(tmp_path / "sub", _SMOKE_DOCS)
    inherited = tmp_path / "inherited"
    env = {**os.environ, "XDG_DATA_HOME": str(inherited), "LOCALAPPDATA": str(inherited)}
    argv = ["scripts/eval/run.py", "--eval-dir", str(eval_dir), "--rescore", str(source)]
    done = subprocess.run(
        [sys.executable, "-I", *argv],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    print(done.stdout, done.stderr)
    assert done.returncode == 0
    logs = list((eval_dir / "appdata").rglob("docsummarizer_*.log"))
    print(f"logs: {logs}")
    assert logs
    assert not inherited.exists() or not any(inherited.rglob("*"))


# --------------------------------------------------------------------------- #
# Compare
# --------------------------------------------------------------------------- #

_COMPARE_FIELDS = ("machine_id", "backend", "metric_version", "corpus_hash", "seeds", "cases")
_COMPARE_CASES = tuple(f"doc{i}/detailed/auto" for i in range(17))
_DIFFERENT: dict[str, Any] = {
    "machine_id": "another machine",
    "backend": "gpu:another",
    "metric_version": 2,
    "corpus_hash": "d" * 64,
    "seeds": [0, 1],
    "cases": list(_COMPARE_CASES[:3]),
}


def _scored_line(case: str, m4: int, repetition: int) -> dict[str, Any]:
    scores = {
        "cell": ["en", "json", "single"],
        "M1": {"violations": 0},
        "M2": {"violations": 0},
        "M4": {"violations": m4},
        "M7": {"tight": 2, "points": 3},
        "M12a": None,
    }
    line = {"kind": "case_run", "case": case, "seed": 0, "repetition": repetition}
    return {**line, "metrics": scores}


def _compare_file(
    path: Path,
    *,
    repeat: int = 1,
    flip: str | None = None,
    cases_: tuple[str, ...] = _COMPARE_CASES,
    **fields: Any,
) -> Path:
    header = {
        "kind": "header",
        "machine_id": "a machine",
        "backend": "cpu",
        "metric_version": 1,
        "corpus_hash": "c" * 64,
        "seeds": [0],
        "cases": list(_COMPARE_CASES),
        "repeat": repeat,
        **fields,
    }
    with path.open("w", encoding="utf-8") as fh:
        run.write_line(fh, header)
        for case in cases_:
            for repetition in range(repeat):
                run.write_line(fh, _scored_line(case, int(case == flip), repetition))
    return path


def _compared(
    tmp_path: Path, b: Path, c: Path, capsys: pytest.CaptureFixture[str]
) -> tuple[int, str]:
    code = run.main(["--eval-dir", str(tmp_path), "--compare", str(b), str(c)])
    out = capsys.readouterr().out
    print(f"--compare {b.name} {c.name} -> {code}\n{out}")
    return code, out


@pytest.mark.parametrize("name", _COMPARE_FIELDS)
def test_compare_refuses_incomparable_arms(
    name: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two runs that measured different things are never compared.

    Catches a field dropped from the comparable set, so a verdict between runs
    of different seeds, machines or documents reads as a change.
    """
    b = _compare_file(tmp_path / "b.jsonl")
    c = _compare_file(tmp_path / "c.jsonl", **{name: _DIFFERENT[name]})
    code, out = _compared(tmp_path, b, c, capsys)
    assert code == 2
    assert f"differ in {name}" in out


def test_compare_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    empty = _compare_file(tmp_path / "empty.jsonl", cases_=())
    code, out = _compared(tmp_path, empty, empty, capsys)
    assert code == 2
    assert "zero case runs" in out

    b = _compare_file(tmp_path / "b.jsonl")
    flipped = _compare_file(tmp_path / "flipped.jsonl", flip=_COMPARE_CASES[4])
    code, out = _compared(tmp_path, b, flipped, capsys)
    assert code == 1
    assert f"regression: {_COMPARE_CASES[4]} seed 0 M4" in out

    for name, candidate in (
        ("same", _compare_file(tmp_path / "same.jsonl")),
        ("repeated", _compare_file(tmp_path / "repeated.jsonl", repeat=3)),
    ):
        code, out = _compared(tmp_path, b, candidate, capsys)
        print(f"{name}: {code}")
        assert code == 0
        assert "B 17 case runs, C 17 case runs" in out
        assert "aggregate" in out
        assert "blocks a merge" in out


# --------------------------------------------------------------------------- #
# Checks, the language self-test and the contention count
# --------------------------------------------------------------------------- #


class _CountedLLM(_EvalLLM):
    """An ``_EvalLLM`` with llama.cpp's cumulative counters, ``n_tokens`` and blocking ``usage``.

    Each call charges its whole prompt to ``n_p_eval``, except an exact repeat
    of the previous call's messages, which evaluates one token (a prefix hit).
    """

    def __init__(self, completion_offset: int = 0) -> None:
        super().__init__()
        self.ctx = object()
        self.n_tokens = 0
        self.perf = {"n_p_eval": 0, "n_eval": 0, "t_p_eval_ms": 0.0, "t_eval_ms": 0.0}
        self.completion_offset = completion_offset
        self._previous: Any = None

    def create_chat_completion(self, *, messages, stream=False, **kwargs):
        prompt = sum(len(self.tokenize(m["content"].encode())) for m in messages)
        evaluated = 1 if messages == self._previous else prompt
        self._previous = messages
        text = self.respond(messages, kwargs)
        out = len(self.tokenize(text.encode()))
        self.perf["n_p_eval"] += evaluated
        self.perf["n_eval"] += out
        self.perf["t_p_eval_ms"] += 10.0 * evaluated
        self.perf["t_eval_ms"] += 50.0 * out
        self.n_tokens = prompt + out
        response = super().create_chat_completion(messages=messages, stream=stream, **kwargs)
        if stream:
            return response
        usage = {"prompt_tokens": prompt, "completion_tokens": out + self.completion_offset}
        return {**response, "usage": usage}


def _counted(monkeypatch: pytest.MonkeyPatch, fake: _CountedLLM) -> None:
    """Serve ``fake`` from ``build_summarizer`` behind a ``llama_cpp`` module reading its counters."""
    import types

    pkg = types.ModuleType("llama_cpp")
    low = types.ModuleType("llama_cpp.llama_cpp")
    low.llama_perf_context = lambda _ctx: types.SimpleNamespace(**fake.perf)
    pkg.llama_cpp = low
    monkeypatch.setitem(sys.modules, "llama_cpp", pkg)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_cpp", low)

    def build(model: Path, *, n_ctx: int, n_threads: int | None, n_gpu_layers: int) -> Summarizer:
        return _shell(fake, n_ctx)

    monkeypatch.setattr(run, "build_summarizer", build)
    monkeypatch.setattr(sys, "path", list(sys.path))


def _checked(tmp_path: Path, capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    code = run.main(["--eval-dir", str(tmp_path), *argv])
    out = capsys.readouterr().out
    print(f"{argv} -> {code}\n{out}")
    return code, out


def test_checks_runstats_passes_on_matching_counters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The request's counts equal the summed per-call counters and the library's usage.

    Catches a prefix hit counted as a full prompt, and counts that drift from
    the counters they are summed from.
    """
    fake = _CountedLLM()
    _counted(monkeypatch, fake)
    code, out = _checked(tmp_path, capsys, "--checks", "runstats", "--model", str(_gguf(tmp_path)))
    assert code == 0
    for leg in ("blocking", "streamed", "prefix repeat"):
        assert f"leg {leg}: calls=" in out
        assert f"leg {leg}: pass" in out
    assert "deltas n_p_eval=" in out
    assert "run_stats prompt_tokens=" in out
    assert "usage=" in out
    assert "runstats: 3 legs, 3 passed" in out


def test_checks_runstats_fails_on_a_counter_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _counted(monkeypatch, _CountedLLM(completion_offset=5))
    code, out = _checked(tmp_path, capsys, "--checks", "runstats", "--model", str(_gguf(tmp_path)))
    assert code == 1
    assert "leg blocking: fail completion_tokens" in out
    assert "runstats: 3 legs, 2 passed" in out


def test_checks_unknown_name_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = _checked(tmp_path, capsys, "--checks", "nope")
    assert code == 2
    assert "nope" in out
    assert all(name in out for name in run.CHECKS)
    code, out = _checked(tmp_path, capsys, "--checks", "")
    assert code == 2


def _snapshots(
    local: tuple[int, ...], host: tuple[str, str] | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``snapshot_now`` returns a start, then an end snapshot of the invented fixtures."""
    taken: list[Any] = []
    for at_end in (False, True):
        table = None if host is None else (lambda cpu=host[at_end]: _host_table(cpu))
        taken.append(run.take_snapshot(_processes(at_end, local), _OWN_PID, table))
    monkeypatch.setattr(run, "snapshot_now", lambda: taken.pop(0))


def test_contention_mode_exits_1_on_either_side(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A busy model process on either side fails the check; an idle or unreadable one does not.

    Catches the mutation "exit 0 whenever the local side is clear", so a busy
    host process passes.
    """
    import time

    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    busy, idle = ("2.0", "7.25"), ("2.0", "2.0")

    _snapshots((101, 103), idle, monkeypatch)
    code, out = _checked(tmp_path, capsys, "--contention")
    assert code == 1
    assert "contention local=1 unknown=0 idle=tallow-idle host=0 idle=orchard-llama-host" in out

    _snapshots((103,), busy, monkeypatch)
    code, out = _checked(tmp_path, capsys, "--contention")
    assert code == 1
    assert "host=1" in out

    _snapshots((103,), idle, monkeypatch)
    assert _checked(tmp_path, capsys, "--contention")[0] == 0

    taken = [
        run.take_snapshot(_processes(at_end, (103, 104)), _OWN_PID, lambda: None)
        for at_end in (False, True)
    ]
    monkeypatch.setattr(run, "snapshot_now", lambda: taken.pop(0))
    code, out = _checked(tmp_path, capsys, "--contention")
    assert code == 0
    assert "unknown=1" in out
    assert "host=unknown" in out

    _snapshots((103,), None, monkeypatch)
    code, out = _checked(tmp_path, capsys, "--contention", "3")
    assert code == 0
    assert "host=n/a" in out
    assert slept == [run.CONTENTION_WINDOW_S] * 4 + [3.0]

    code, out = _checked(tmp_path, capsys, "--contention", "0")
    assert code == 2
    assert slept[-1] == 3.0


_LABELLED = {
    "zh": (
        "村里的面包店每天清晨五点开门并把炉子烧热。",
        "学校的菜园里种着豆子、胡萝卜和高高的向日葵。",
    ),
    "de": (
        "Die Fähre fährt jeden Morgen um sechs Uhr über den breiten Fluss.",
        "Im Schulgarten wachsen Bohnen, Karotten und hohe gelbe Sonnenblumen.",
    ),
    "en": (
        "The river ferry leaves the wooden pier at six every morning.",
        "The school garden grows beans, carrots and tall yellow sunflowers.",
    ),
    "fr": (
        "Le bac quitte le ponton en bois chaque matin à six heures.",
        "Le jardin de l'école donne des haricots, des carottes et de grands tournesols.",
    ),
    "es": (
        "El transbordador sale del muelle de madera cada mañana a las seis.",
        "El huerto de la escuela da judías, zanahorias y altos girasoles amarillos.",
    ),
    "ru": (
        "Паром каждое утро отходит от деревянной пристани в шесть часов.",
        "Школьный сад даёт бобы, морковь и высокие жёлтые подсолнухи.",
    ),
}


def test_self_test_language_computes_three_rates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [{"text": text, "label": label} for label, texts in _LABELLED.items() for text in texts]
    rows.append({"text": "a row without its label"})
    labelled = tmp_path / "labelled.jsonl"
    labelled.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    code, out = _checked(tmp_path, capsys, "--self-test", "language", str(labelled))
    assert code == 2
    assert "12 sentences" in out
    assert "6 languages" in out

    monkeypatch.setattr(run, "SELF_TEST_MIN_SENTENCES", 12)
    monkeypatch.setattr(run, "SELF_TEST_MIN_LANGUAGES", 6)
    code, out = _checked(tmp_path, capsys, "--self-test", "language", str(labelled))
    assert code in (0, 1)
    assert "skipped 1 row without text or label" in out
    assert re.search(r"sentence error \d+/12 \(\d+\.\d%\), bar 1\.5%", out)
    assert re.search(r"false fail \d+/1000 \(\d+\.\d%\), bar 3\.0%", out)
    assert re.search(r"missed drift \d+/1000 \(\d+\.\d%\), bar 1\.0%", out)

    code, out = _checked(tmp_path, capsys, "--self-test", "nope", str(labelled))
    assert code == 2


_DOCS_HEADING = "## Evaluating a model, prompt or pipeline change"


def _docs_section() -> str:
    text = (_REPO / "DEVELOPMENT.md").read_text(encoding="utf-8")
    start = text.find(_DOCS_HEADING)
    if start < 0:
        return ""
    end = text.find("\n## ", start + len(_DOCS_HEADING))
    return text[start : end if end >= 0 else len(text)]


def test_docs_name_every_runner_option() -> None:
    """Every runner option and check is in the evaluation guide; the scripts index lists it."""
    options = sorted(
        option
        for action in run.build_parser()._actions
        for option in action.option_strings
        if option.startswith("--") and option != "--help"
    )
    names = options + sorted(run.CHECKS)
    print(f"{len(options)} options, {len(run.CHECKS)} checks")
    assert options
    section = _docs_section()
    missing = [
        name for name in names if not re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", section)
    ]
    assert missing == []
    readme = (_REPO / "scripts" / "README.md").read_text(encoding="utf-8")
    assert any(line.startswith("|") and "`eval/run.py`" in line for line in readme.splitlines())
