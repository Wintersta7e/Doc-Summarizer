# SPDX-License-Identifier: MIT
"""The evaluation runner: runs cases against a local corpus and records what it measured.

Module-level imports are the standard library only. ``bootstrap()`` decides
which checkout ``docsummarizer`` resolves from and where its app data goes, so
``cases``, ``metrics`` and every ``docsummarizer`` module are imported inside
the functions that use them, after it has run.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import IO, TYPE_CHECKING, Any, Literal, Protocol, cast, overload

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from docsummarizer.model_manager import (
        ChatCompletionResponseChoice,
        ChatCompletionResponseMessage,
        ChatModel,
        CompletionUsage,
        CreateChatCompletionResponse,
        CreateChatCompletionStreamResponse,
        RunStats,
        StructuredSummary,
        Summarizer,
    )
    from eval.cases import Case
    from eval.metrics import CaseScore, Target

    class SeededModel(ChatModel, Protocol):
        """A loaded ``llama_cpp.Llama``: the chat calls plus its context reset and sampler seed."""

        def reset(self) -> None:
            """Clear the context state."""
            ...

        def set_seed(self, seed: int) -> None:
            """Seed the sampler."""
            ...


_REPO = Path(__file__).resolve().parents[2]
_PACKAGE = "docsummarizer"


def bootstrap(app_data: Path) -> None:
    """Point app data at ``app_data`` and import ``docsummarizer`` from this checkout."""
    os.environ["XDG_DATA_HOME"] = str(app_data)
    os.environ["LOCALAPPDATA"] = str(app_data)
    for entry in (str(_REPO / "scripts"), str(_REPO / "src")):
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    import docsummarizer

    location = Path(str(docsummarizer.__file__)).resolve()
    if not location.is_relative_to(_REPO / "src"):
        msg = f"docsummarizer resolves from {location.parent}, not {_REPO / 'src'}"
        raise SystemExit(msg)


# --------------------------------------------------------------------------- #
# The tree a run measured
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GitState:
    """The checkout's commit and whether it had uncommitted changes."""

    commit: str | None
    dirty: bool | None
    error: str | None  # why commit / dirty are unknown


def git_state(repo: Path) -> GitState:
    """Read HEAD and the working tree's state; a missing or failing git is recorded."""
    git = shutil.which("git")
    if git is None:
        return GitState(None, None, "git not found")
    head = subprocess.run(
        [git, "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=False
    )
    if head.returncode != 0:
        return GitState(None, None, head.stderr.strip() or f"git exited {head.returncode}")
    status = subprocess.run(
        [git, "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=False
    )
    if status.returncode != 0:
        return GitState(head.stdout.strip(), None, status.stderr.strip())
    return GitState(head.stdout.strip(), bool(status.stdout.strip()), None)


# The modules whose change requires a measured run, and the ones that never do.
GATED_ROOTS = ("model_manager.py", "provenance.py", "native.py", "hardware.py")
NOT_GATED = ("logger.py", "paths.py")


@dataclass(frozen=True)
class Gated:
    """The gated files (package-relative POSIX paths) and the roots not present."""

    files: tuple[str, ...]
    missing_roots: tuple[str, ...]


def _module_file(package_dir: Path, parts: tuple[str, ...]) -> str | None:
    if not parts:
        candidates = ["__init__.py"]
    else:
        base = "/".join(parts)
        candidates = [f"{base}.py", f"{base}/__init__.py"]
    return next((c for c in candidates if (package_dir / c).is_file()), None)


def _import_targets(node: ast.AST, here: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Package-relative module paths an import node may load, most of them not files."""
    if isinstance(node, ast.Import):
        return [
            tuple(alias.name.split(".")[1:])
            for alias in node.names
            if alias.name.split(".")[0] == _PACKAGE
        ]
    if not isinstance(node, ast.ImportFrom):
        return []
    module = tuple(node.module.split(".")) if node.module else ()
    if node.level:
        if node.level - 1 > len(here):
            return []
        base = here[: len(here) - node.level + 1] + module
    elif module and module[0] == _PACKAGE:
        base = module[1:]
    else:
        return []
    return [base, *((*base, alias.name) for alias in node.names)]


def _imported_files(package_dir: Path, rel: str) -> set[str]:
    tree = ast.parse((package_dir / rel).read_text(encoding="utf-8"), filename=rel)
    here = PurePosixPath(rel).parent.parts
    found: set[str] = set()
    for node in ast.walk(tree):
        for parts in _import_targets(node, here):
            target = _module_file(package_dir, parts)
            if target is not None:
                found.add(target)
    return found


def derive_gated(package_dir: Path) -> Gated:
    """The roots plus every package module they import, transitively, less ``NOT_GATED``.

    Imports inside function bodies count. A root that exists but cannot be read
    or parsed raises: unreadable is not absent.
    """
    present = [root for root in GATED_ROOTS if (package_dir / root).is_file()]
    missing = tuple(root for root in GATED_ROOTS if root not in present)
    seen = set(present)
    pending = list(present)
    while pending:
        for target in _imported_files(package_dir, pending.pop()):
            if target not in seen:
                seen.add(target)
                pending.append(target)
    return Gated(tuple(sorted(seen - set(NOT_GATED))), missing)


def gated_hash(package_dir: Path, files: Iterable[str]) -> str:
    """sha256 over each file's relative path and bytes, in sorted order."""
    digest = hashlib.sha256()
    for rel in sorted(files):
        digest.update(rel.encode("utf-8") + b"\0")
        digest.update((package_dir / rel).read_bytes() + b"\0")
    return digest.hexdigest()


def machine_id() -> str:
    """The host name and CPU model, which a comparison requires to match."""
    cpu = platform.processor()
    if not cpu:
        try:
            lines = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        cpu = next(
            (line.split(":", 1)[1].strip() for line in lines if line.startswith("model name")),
            "",
        )
    return f"{platform.node()}/{cpu}"


# --------------------------------------------------------------------------- #
# The corpus manifest
# --------------------------------------------------------------------------- #


class ManifestError(Exception):
    """The corpus manifest or a document it names cannot be used."""


@dataclass(frozen=True)
class ManifestDocument:
    """One corpus document as the manifest records it."""

    id: str
    path: Path
    label: str
    script: str | None
    sha256: str


_SCRIPTS = ("Simplified", "Traditional")
_SPLIT_MIN_PARTS = 2  # a split case must reach at least two parts
_REQUIRED = ("id", "file", "label", "sha256")


def _manifest_document(entry: object, corpus: Path) -> ManifestDocument:
    from eval import cases

    if not isinstance(entry, dict) or not all(isinstance(entry.get(k), str) for k in _REQUIRED):
        msg = f"manifest unreadable: an entry lacks one of {_REQUIRED}"
        raise ManifestError(msg)
    doc_id = entry["id"]
    shape = cases.DOCUMENTS.get(doc_id)
    if shape is None:
        msg = f"manifest unknown: {doc_id} (ad-hoc documents go through --doc)"
        raise ManifestError(msg)
    if entry["label"] != shape.label:
        msg = f"manifest label: {doc_id} ({entry['label']!r}, not {shape.label!r})"
        raise ManifestError(msg)
    script = entry.get("script")
    if shape.label == "zh" and (
        script not in _SCRIPTS or (shape.script is not None and script != shape.script)
    ):
        msg = f"manifest script: {doc_id} ({script!r})"
        raise ManifestError(msg)
    path = (corpus / entry["file"]).resolve()
    if not path.is_relative_to(corpus.resolve()):
        msg = f"manifest file outside the corpus: {doc_id}"
        raise ManifestError(msg)
    return ManifestDocument(doc_id, path, entry["label"], script, entry["sha256"])


def load_manifest(eval_dir: Path) -> dict[str, ManifestDocument]:
    """Read ``<eval-dir>/corpus/manifest.json`` and check it against the case matrix."""
    corpus = eval_dir / "corpus"
    manifest = corpus / "manifest.json"
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except FileNotFoundError:
        msg = f"manifest absent: {manifest}"
        raise ManifestError(msg) from None
    except (OSError, ValueError) as exc:
        msg = f"manifest unreadable: {manifest}: {exc}"
        raise ManifestError(msg) from exc
    entries = data.get("documents") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        msg = f"manifest unreadable: {manifest}: no documents list"
        raise ManifestError(msg)
    documents: dict[str, ManifestDocument] = {}
    for entry in entries:
        doc = _manifest_document(entry, corpus)
        if doc.id in documents:
            msg = f"manifest duplicate: {doc.id}"
            raise ManifestError(msg)
        documents[doc.id] = doc
    return documents


def verify_documents(docs: Iterable[ManifestDocument]) -> None:
    """Check that each document exists and matches its recorded sha256."""
    problems: list[str] = []
    for doc in docs:
        try:
            digest = hashlib.sha256(doc.path.read_bytes()).hexdigest()
        except FileNotFoundError:
            problems.append(f"absent: {doc.id}")
            continue
        except OSError as exc:
            problems.append(f"unreadable: {doc.id} ({exc})")
            continue
        if digest != doc.sha256:
            problems.append(f"sha256: {doc.id}")
    if problems:
        raise ManifestError("corpus " + "; ".join(problems))


def corpus_hash(docs: Iterable[ManifestDocument]) -> str:
    """sha256 over the sorted ``id:sha256`` lines of the documents a run uses."""
    lines = sorted(f"{doc.id}:{doc.sha256}\n" for doc in docs)
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Contention: another model process using the CPU during a case run
# --------------------------------------------------------------------------- #

# CPU seconds a matching process must use over a case run to mark it contended.
CONTENTION_CPU_SECONDS: float = 1.0
_MODEL_MARKERS = ("llama", "ggml")


@dataclass(frozen=True)
class ContentionSnapshot:
    """Matching processes at one instant: pid -> (name, user + system CPU seconds)."""

    local: dict[int, tuple[str, float]]
    local_unknown: int  # processes that denied access
    host: dict[int, tuple[str, float]] | None  # the Windows side of a WSL run
    host_state: Literal["ok", "unknown", "n/a"]


@dataclass(frozen=True)
class SideCount:
    """One side's count over a case run; ``count`` is ``None`` when it is unknown."""

    count: int | None
    unknown: int
    busy: tuple[str, ...]
    idle: tuple[str, ...]


@dataclass(frozen=True)
class Contention:
    """Both sides' counts over a case run; ``host`` is ``None`` off WSL."""

    local: SideCount
    host: SideCount | None

    @property
    def contended(self) -> bool:
        """Whether either side counted a contending process."""
        if (self.local.count or 0) > 0:
            return True
        return self.host is not None and (self.host.count or 0) > 0


def _basename(path: str) -> str:
    return PureWindowsPath(path).name.lower()  # splits on both separators


def under_wsl() -> bool:
    """Whether this process runs under WSL, with a Windows host beside it."""
    return "microsoft" in platform.uname().release.lower()


def read_host_processes() -> str | None:
    """The Windows host's process table as CSV; ``None`` when the query fails.

    Process id, name and CPU seconds of every Windows process. ``tasklist /v``
    carries the same CPU column but takes about a minute on a busy host.
    """
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        return None
    try:
        done = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-Process | Select-Object Id,ProcessName,CPU | ConvertTo-Csv -NoTypeInformation",
            ],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def parse_host_processes(text: str) -> dict[int, tuple[str, float]]:
    """Matching host processes: pid (column 0), name (1) and CPU seconds (2).

    The header and a process whose CPU cell is empty (protected, unreadable)
    are skipped, so neither counts as busy or idle.
    """
    found: dict[int, tuple[str, float]] = {}
    for row in csv.reader(io.StringIO(text)):
        try:
            pid, image, cpu = int(row[0]), row[1], float(row[2])
        except (IndexError, ValueError):
            continue
        if any(marker in image.lower() for marker in _MODEL_MARKERS):
            found[pid] = (image, cpu)
    return found


def take_snapshot(
    processes: Iterable[Any], own_pid: int, host_processes: Callable[[], str | None] | None
) -> ContentionSnapshot:
    """Record every matching process's CPU time, and the host's when ``host_processes`` is given.

    A process matches when its executable's basename contains ``llama``, or a
    basename among its mapped files contains ``llama`` or ``ggml``.
    """
    import psutil

    local: dict[int, tuple[str, float]] = {}
    unknown = 0
    for proc in processes:
        if proc.pid == own_pid:
            continue
        try:
            exe = _basename(proc.exe() or "")
            matched = "llama" in exe or any(
                marker in _basename(mapped.path)
                for mapped in proc.memory_maps()
                for marker in _MODEL_MARKERS
            )
            if matched:
                times = proc.cpu_times()
                local[proc.pid] = (exe or f"pid {proc.pid}", times.user + times.system)
        except psutil.AccessDenied:
            unknown += 1
        except psutil.NoSuchProcess:
            continue
    if host_processes is None:
        return ContentionSnapshot(local, unknown, None, "n/a")
    text = host_processes()
    if text is None:
        return ContentionSnapshot(local, unknown, None, "unknown")
    return ContentionSnapshot(local, unknown, parse_host_processes(text), "ok")


def snapshot_now() -> ContentionSnapshot:
    """A snapshot of this machine, and of the Windows host under WSL."""
    import psutil

    host = read_host_processes if under_wsl() else None
    return take_snapshot(psutil.process_iter(), os.getpid(), host)


def _side(
    start: dict[int, tuple[str, float]], end: dict[int, tuple[str, float]], unknown: int
) -> SideCount:
    busy: list[str] = []
    idle: list[str] = []
    for pid, (name, cpu) in sorted(end.items()):
        before = start[pid][1] if pid in start else 0.0
        (busy if cpu - before > CONTENTION_CPU_SECONDS else idle).append(name)
    return SideCount(len(busy), unknown, tuple(busy), tuple(idle))


def count_contention(start: ContentionSnapshot, end: ContentionSnapshot) -> Contention:
    """Count, per side, the matching processes that used the CPU between two snapshots."""
    local = _side(start.local, end.local, max(start.local_unknown, end.local_unknown))
    if "n/a" in (start.host_state, end.host_state):
        return Contention(local, None)
    if start.host is None or end.host is None:
        return Contention(local, SideCount(None, 0, (), ()))
    return Contention(local, _side(start.host, end.host, 0))


# --------------------------------------------------------------------------- #
# Capture and the case loop
# --------------------------------------------------------------------------- #

_PERF_FIELDS = ("n_p_eval", "n_eval", "t_p_eval_ms", "t_eval_ms")


@dataclass
class CallRecord:
    """One completion as the model returned it, before any clean-up."""

    seq: int
    messages: list[dict[str, Any]]
    kwargs: dict[str, Any]
    stream: bool
    content: str
    finish_reason: str | None
    usage: CompletionUsage | None
    perf: dict[str, float] | None  # the call's counter deltas; None without counters
    error: str | None


def _jsonable(value: object) -> object:
    """``value`` when it serializes to JSON, else its type name."""
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return type(value).__name__
    return value


class CapturingLlama:
    """Wraps the model and records every completion and every tokenize length.

    Every other attribute is the wrapped model's, so the pipeline sees the
    same context pointer and counters.
    """

    def __init__(self, real: SeededModel, snapshot: Callable[[], object]) -> None:
        """Wrap ``real``; ``snapshot`` reads its perf counters, or ``None`` without them."""
        self._real: SeededModel = real
        self._snapshot: Callable[[], object] = snapshot
        self.calls: list[CallRecord] = []
        self.tokenized: dict[str, int] = {}

    def __getattr__(self, name: str) -> object:
        """Any attribute not wrapped here is the wrapped model's."""
        return getattr(self._real, name)

    def _perf_since(self, before: object) -> dict[str, float] | None:
        after = self._snapshot() if before is not None else None
        if before is None or after is None:
            return None
        return {name: getattr(after, name) - getattr(before, name) for name in _PERF_FIELDS}

    @overload
    def create_chat_completion(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        stream: Literal[False] = False,
        **kwargs: object,
    ) -> CreateChatCompletionResponse: ...

    @overload
    def create_chat_completion(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        stream: Literal[True],
        **kwargs: object,
    ) -> Iterator[CreateChatCompletionStreamResponse]: ...

    def create_chat_completion(
        self, *, messages: Sequence[Mapping[str, str]], stream: bool = False, **kwargs: object
    ) -> CreateChatCompletionResponse | Iterator[CreateChatCompletionStreamResponse]:
        """Forward the completion to the model and record the request and its answer."""
        before = self._snapshot()
        record = CallRecord(
            seq=len(self.calls),
            messages=[dict(message) for message in messages],
            kwargs={key: _jsonable(value) for key, value in kwargs.items()},
            stream=stream,
            content="",
            finish_reason=None,
            usage=None,
            perf=None,
            error=None,
        )
        self.calls.append(record)
        if stream:
            chunks = self._real.create_chat_completion(messages=messages, stream=True, **kwargs)
            return self._stream(record, before, chunks)
        try:
            response = self._real.create_chat_completion(messages=messages, stream=False, **kwargs)
            choice = response["choices"][0]
            record.content = str(choice["message"].get("content") or "")
            record.finish_reason = choice.get("finish_reason")
            record.usage = response.get("usage")
        except Exception as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.perf = self._perf_since(before)
        return response

    def _stream(
        self,
        record: CallRecord,
        before: object,
        chunks: Iterable[CreateChatCompletionStreamResponse],
    ) -> Iterator[CreateChatCompletionStreamResponse]:
        try:
            for chunk in chunks:
                choice = chunk["choices"][0]
                delta = (choice.get("delta") or {}).get("content")
                if delta:
                    record.content += delta
                record.finish_reason = choice.get("finish_reason") or record.finish_reason
                yield chunk
        except Exception as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            # An aborted stream is recorded too: this runs when the consumer stops.
            record.perf = self._perf_since(before)

    def tokenize(self, text: bytes, *, add_bos: bool = True) -> list[int]:
        """Tokenize through the model and record the token count by text hash."""
        result: list[int] = self._real.tokenize(text, add_bos=add_bos)
        self.tokenized[hashlib.sha256(text).hexdigest()] = len(result)
        return result

    def reset(self) -> None:
        """Clear the model's context state."""
        self._real.reset()

    def set_seed(self, seed: int) -> None:
        """Seed the model's sampler."""
        self._real.set_seed(seed)


def capture_calls(summarizer: Summarizer) -> CapturingLlama:
    """Wrap the summarizer's model in a ``CapturingLlama``, once."""
    llm = summarizer.llm
    if isinstance(llm, CapturingLlama):
        return llm
    if llm is None:
        msg = "the summarizer is closed"
        raise RuntimeError(msg)
    # build_summarizer loads a llama_cpp.Llama, which has reset and set_seed.
    capture = CapturingLlama(cast("SeededModel", llm), summarizer.perf_snapshot)
    summarizer.llm = capture
    return capture


def build_summarizer(
    model: Path, *, n_ctx: int, n_threads: int | None, n_gpu_layers: int
) -> Summarizer:
    """Load the model: the runner's one construction site."""
    from docsummarizer.model_manager import Summarizer

    return Summarizer(model, n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=n_gpu_layers)


@dataclass(frozen=True)
class RunSpec:
    """One case run to make."""

    case: Case
    seed: int
    repetition: int = 0


@dataclass
class CaseRun:
    """One case run's outcome, its calls and its scores."""

    case: str
    seed: int
    repetition: int
    n_ctx: int
    parts: int
    result: dict[str, Any] | None
    run_stats: dict[str, Any] | None
    calls: list[CallRecord]
    tokenized: dict[str, int]
    contention: dict[str, Any]
    contended: bool
    error: str | None
    metrics: dict[str, Any] = field(default_factory=dict)


def _json_units(content: str) -> list[str] | None:
    """The lead and point texts the app's parser reads from one JSON call."""
    from docsummarizer.model_manager import parse_structured_json
    from eval import metrics

    parsed = parse_structured_json(content)
    if parsed is None:
        return None
    texts = [str(parsed.get("lead") or ""), *(text for text, _ in _json_points(parsed))]
    return [unit for text in texts for unit in metrics.split_units(text)]


def _json_points(parsed: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Every ``(text, quote)`` pair of a parsed call: Detailed points and each section."""
    from docsummarizer.model_manager import read_raw_points

    pairs = read_raw_points(parsed.get("points"))
    sections = parsed.get("sections")
    if isinstance(sections, dict):
        for value in sections.values():
            pairs.extend(read_raw_points(value))
    return pairs


def _model_quotes(calls: Sequence[CallRecord]) -> dict[str, str]:
    """Each point text's quote as the model wrote it, over every parsed JSON call."""
    from docsummarizer.model_manager import parse_structured_json

    quotes: dict[str, str] = {}
    for call in calls:
        parsed = parse_structured_json(call.content) if "response_format" in call.kwargs else None
        if parsed is not None:
            quotes.update(_json_points(parsed))
    return quotes


def _result_points(result: StructuredSummary) -> list[Any]:
    points = list(result.points)
    for section in (result.sections or {}).values():
        points.extend(section)
    return points


@dataclass(frozen=True)
class ScoringInput:
    """What a case run was asked: the case, its document text, its target and its part count."""

    case: Case
    text: str
    target: Target
    parts: int


def score_case_run(
    asked: ScoringInput, result: StructuredSummary | None, calls: Sequence[CallRecord]
) -> dict[str, Any]:
    """Score one case run: every call's raw content and the returned result.

    The one scoring path a run and a rescore share. The rendered ``text`` of a
    result is never scored.
    """
    from docsummarizer.model_manager import SUMMARY_TYPE_BRIEF
    from eval import metrics

    case, text, target, parts = asked.case, asked.text, asked.target, asked.parts
    is_json = ["response_format" in call.kwargs for call in calls]
    call_units = [
        _json_units(call.content) if json_call else metrics.split_units(call.content)
        for call, json_call in zip(calls, is_json, strict=True)
    ]
    result_units: list[str] = []
    m7: dict[str, Any] | None = None
    if result is not None:
        points = _result_points(result)
        texts = [result.lead or "", *(point.text for point in points)]
        result_units = [unit for t in texts for unit in metrics.split_units(t)]
        if case.summary_type != SUMMARY_TYPE_BRIEF:
            scored = metrics.m7(metrics.point_records(points, _model_quotes(calls)), text)
            m7 = {**asdict(scored), "hit_rate": scored.hit_rate}
    m1 = [metrics.m1(call.content).passed for call in calls]
    m4 = [metrics.m4(call.finish_reason) for call in calls]
    m2 = metrics.m2(call_units, result_units, target)
    script = f"-{target.script}" if target.script else ""
    path = "prose" if case.summary_type == SUMMARY_TYPE_BRIEF else "json"
    return {
        "cell": [target.code + script, path, "single" if parts == 1 else "multi"],
        "M1": {
            "violations": m1.count(False),
            "calls": len(calls),
            "failed_calls": [call.seq for call, ok in zip(calls, m1, strict=True) if not ok],
        },
        "M2": {
            "violations": len(m2.off_target),
            "passed": m2.passed,
            "scored": m2.scored,
            "unparsed_calls": m2.unparsed_calls,
            "off_target": [asdict(off) for off in m2.off_target],
        },
        "M4": {
            "violations": sum(m4),
            "calls": len(calls),
            "truncated_calls": [call.seq for call, cut in zip(calls, m4, strict=True) if cut],
        },
        "M7": m7,
        "M12a": None,
    }


def _first_run_of(rerun: Case, seed: int, done: Sequence[CaseRun]) -> CaseRun | None:
    first_id = replace(rerun, m12a=False).id
    return next(
        (r for r in done if r.case == first_id and r.seed == seed and r.repetition == 0), None
    )


def _m12a(first: CaseRun | None, rerun: CaseRun) -> dict[str, Any] | None:
    from eval import metrics

    if first is None or first.result is None or rerun.result is None:
        return None
    outcome = metrics.m12a(first.result["text"], rerun.result["text"])
    return {**asdict(outcome), "violations": 0 if outcome.identical else 1, "first": first.case}


def _counting_chunks(summarizer: Summarizer, observed: list[int]) -> Callable[..., Any]:
    """``chunks_within``, recording the piece count of each top-level call."""
    original = summarizer.chunks_within
    depth = 0

    def counting(
        text: str, budget_tokens: int, split: Callable[[str, int], list[tuple[str, int]]]
    ) -> list[tuple[str, int]]:
        nonlocal depth
        depth += 1
        try:
            pieces = original(text, budget_tokens, split)
        finally:
            depth -= 1
        if depth == 0:
            observed.append(len(pieces))
        return pieces

    return counting


def _run_one(
    summarizer: Summarizer, capture: CapturingLlama, spec: RunSpec, text: str
) -> tuple[StructuredSummary | None, str | None, int, Contention]:
    capture.calls = []
    capture.tokenized = {}
    capture.reset()
    capture.set_seed(spec.seed)
    observed: list[int] = []
    summarizer.chunks_within = _counting_chunks(summarizer, observed)  # type: ignore[method-assign]
    result: StructuredSummary | None = None
    error: str | None = None
    start = snapshot_now()
    try:
        result = summarizer.summarize_structured(
            text,
            spec.case.summary_type,
            should_cancel=lambda: False,
            language=spec.case.language,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"  # recorded in the results file, never logged
    finally:
        end = snapshot_now()
        del summarizer.chunks_within
    parts = getattr(result, "parts_total", None) or (observed[0] if observed else 0)
    return result, error, parts, count_contention(start, end)


def run_cases(
    summarizer: Summarizer,
    runs: Sequence[RunSpec],
    texts: Mapping[str, str],
    targets: Mapping[str, Target],
    on_run: Callable[[CaseRun], None] | None = None,
) -> list[CaseRun]:
    """Run each spec in order from a reset model and its seed, capturing every call.

    ``texts`` is keyed by document id and ``targets`` by case id. A case flagged
    ``m12a`` is compared with repetition 0 of the same case without the flag.
    ``on_run`` sees each case run as soon as it is scored.
    """
    capture = capture_calls(summarizer)
    done: list[CaseRun] = []
    for spec in runs:
        case, text = spec.case, texts[spec.case.doc]
        result, error, parts, contention = _run_one(summarizer, capture, spec, text)
        result_dict = asdict(result) if result is not None else None
        stats = result_dict.pop("run_stats", None) if result_dict is not None else None
        case_run = CaseRun(
            case=case.id,
            seed=spec.seed,
            repetition=spec.repetition,
            n_ctx=summarizer.n_ctx,
            parts=parts,
            result=result_dict,
            run_stats=stats,
            calls=capture.calls,
            tokenized=capture.tokenized,
            contention={**asdict(contention), "contended": contention.contended},
            contended=contention.contended,
            error=error,
        )
        asked = ScoringInput(case, text, targets[case.id], parts)
        case_run.metrics = score_case_run(asked, result, case_run.calls)
        if case.m12a:
            case_run.metrics["M12a"] = _m12a(_first_run_of(case, spec.seed, done), case_run)
        done.append(case_run)
        if on_run is not None:
            on_run(case_run)
    return done


# --------------------------------------------------------------------------- #
# The tier CLI and the results file
# --------------------------------------------------------------------------- #

RUNNABLE_TIERS = ("smoke", "lang")
REPEAT_SCOPE = "seed-0 auto runs"
HEADER_FIELDS: tuple[str, ...] = (
    "kind",
    "metric_version",
    "commit",
    "dirty",
    "git_error",
    "gated_hash",
    "gated_files",
    "gated_missing_roots",
    "corpus_hash",
    "documents",
    "model_sha256",
    "model_size",
    "llama_cpp_version",
    "lib_dir",
    "backend",
    "n_gpu_layers",
    "threads",
    "n_ctx",
    "split_n_ctx",
    "seeds",
    "repeat",
    "repeat_scope",
    "tier",
    "cases",
    "not_split",
    "machine_id",
    "utc_start",
    "utc_end",
    "load_average",
    "contention",
    "peak_rss_bytes",
)


class _RefusedError(Exception):
    """A selection or input the runner will not measure; main prints it and exits 2."""


def _seed_list(text: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part) for part in text.split(","))
    except ValueError:
        msg = f"not a comma-separated list of seeds: {text!r}"
        raise argparse.ArgumentTypeError(msg) from None
    return seeds


def _at_least_one(text: str) -> int:
    value = int(text)
    if value < 1:
        msg = f"must be at least 1, got {value}"
        raise argparse.ArgumentTypeError(msg)
    return value


class EvalArgs(argparse.Namespace):
    """The parsed command line, one typed attribute per option; defaults match the parser's."""

    tier: str | None = None
    rescore: Path | None = None
    compare: list[Path] | None = None
    checks: str | None = None
    self_test: list[str] | None = None
    contention: float | None = None
    eval_dir: Path | None = None
    model: Path | None = None
    threads: int | None = None
    n_gpu_layers: int = 0
    n_ctx: int | None = None
    seeds: tuple[int, ...] = (0,)
    repeat: int = 1

    def __init__(self) -> None:
        """Give each repeatable option its own empty list."""
        super().__init__()
        self.case: list[str] = []
        self.doc: list[Path] = []
        self.label: list[str] = []
        self.script: list[str] = []


def build_parser() -> argparse.ArgumentParser:
    """The runner's command line: one mode, plus the options a tier run reads."""
    parser = argparse.ArgumentParser(
        prog="run.py", description="Run evaluation cases against a local corpus."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    _ = mode.add_argument("--tier", choices=RUNNABLE_TIERS, help="run a tier's cases")
    _ = mode.add_argument(
        "--rescore", type=Path, metavar="R", help="replay results file R and score it again"
    )
    _ = mode.add_argument(
        "--compare", type=Path, nargs=2, metavar=("B", "C"), help="candidate C against baseline B"
    )
    _ = mode.add_argument("--checks", metavar="LIST", help="comma-separated checks to run")
    _ = mode.add_argument(
        "--self-test", nargs="+", metavar=("KIND", "ARG"), help="language <labelled set>"
    )
    _ = mode.add_argument(
        "--contention",
        type=float,
        nargs="?",
        const=CONTENTION_WINDOW_S,
        metavar="SECONDS",
        help="count busy model processes over a window (default 10 s)",
    )
    _ = parser.add_argument("--eval-dir", type=Path, help="holds corpus/, results/ and appdata/")
    _ = parser.add_argument("--model", type=Path, help="the GGUF model file")
    _ = parser.add_argument("--threads", type=int, help="CPU threads (default: half the cores)")
    _ = parser.add_argument("--n-gpu-layers", type=int, default=0, help="layers to offload")
    _ = parser.add_argument("--n-ctx", type=int, help="context window (default: the app's)")
    _ = parser.add_argument("--seeds", type=_seed_list, default=(0,), help="comma-separated")
    _ = parser.add_argument(
        "--repeat", type=_at_least_one, default=1, help="repeat each seed-0 auto run N times"
    )
    _ = parser.add_argument("--case", action="append", default=[], help="a case id or document id")
    _ = parser.add_argument("--doc", action="append", type=Path, default=[], help="ad-hoc document")
    _ = parser.add_argument("--label", action="append", default=[], help="each --doc's language")
    _ = parser.add_argument("--script", action="append", default=[], help="each zh --doc's script")
    return parser


def expand_runs(cases: Sequence[Case], seeds: Sequence[int], repeat: int) -> list[RunSpec]:
    """The case runs to make, in order; only seed-0 ``auto`` runs repeat.

    Per case in tier order: at seed 0 an ``auto`` case runs ``repeat`` times
    back to back and any other case once; at each later seed only an ``auto``
    case runs, once. The ``m12a`` re-runs go last, at seed 0.
    """
    runs: list[RunSpec] = []
    reruns: list[RunSpec] = []
    for case in cases:
        if case.m12a:
            reruns.append(RunSpec(case, 0))
            continue
        auto = case.language == "auto"
        for seed in seeds:
            if seed == 0:
                runs.extend(RunSpec(case, 0, rep) for rep in range(repeat if auto else 1))
            elif auto:
                runs.append(RunSpec(case, seed))
    return runs + reruns


def results_path(eval_dir: Path, tier: str, short_sha: str, now: datetime) -> Path:
    """``<eval-dir>/results/<UTC>-<tier>-<sha>.jsonl``."""
    return eval_dir / "results" / f"{now:%Y%m%dT%H%M%SZ}-{tier}-{short_sha}.jsonl"


def write_line(fh: IO[str], record: Mapping[str, Any]) -> None:
    """One JSON record per line."""
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class Results:
    """A results file: its header, call lines and case-run lines."""

    header: dict[str, Any]
    calls: list[dict[str, Any]]
    case_runs: list[dict[str, Any]]


def read_results(path: Path) -> Results:
    """Read a results file; the header must be its first line."""
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not lines or lines[0].get("kind") != "header":
        msg = f"{path}: the first line is not a header"
        raise ValueError(msg)
    calls = [line for line in lines if line.get("kind") == "call"]
    case_runs = [line for line in lines if line.get("kind") == "case_run"]
    return Results(lines[0], calls, case_runs)


def _share(num: int, den: int) -> str:
    return f"{num}/{den}"


def _table_row(name: str, runs: Sequence[Mapping[str, Any]]) -> str:
    metrics = [run["metrics"] for run in runs]
    calls = sum(m["M1"]["calls"] for m in metrics)
    m1 = _share(calls - sum(m["M1"]["violations"] for m in metrics), calls)
    m2 = _share(sum(1 for m in metrics if m["M2"]["passed"]), len(metrics))
    m4 = _share(calls - sum(m["M4"]["violations"] for m in metrics), calls)
    m7s = [m["M7"] for m in metrics if m["M7"] is not None]
    m7 = _share(sum(m["tight"] for m in m7s), sum(m["points"] for m in m7s))
    m12as = [m["M12a"] for m in metrics if m.get("M12a") is not None]
    m12a = _share(sum(1 for m in m12as if m["identical"]), len(m12as))
    return f"{name:<28} {len(runs):>5} {m1:>9} {m2:>9} {m4:>9} {m7:>9} {m12a:>7}"


def print_table(case_runs: Sequence[Mapping[str, Any]]) -> None:
    """One row per cell and an aggregate row, read from repetition 0; then the counts."""
    first = [run for run in case_runs if run.get("repetition", 0) == 0]
    cells: dict[str, list[Mapping[str, Any]]] = {}
    for run in first:
        cells.setdefault("/".join(run["metrics"]["cell"]), []).append(run)
    print(f"{'cell':<28} {'runs':>5} {'M1':>9} {'M2':>9} {'M4':>9} {'M7':>9} {'M12a':>7}")
    for name in sorted(cells):
        print(_table_row(name, cells[name]))
    print(_table_row("aggregate", first))
    for run in first:
        metrics = run["metrics"]
        for off in metrics["M2"]["off_target"]:
            where = "result" if off["call"] is None else f"call {off['call']}"
            print(f"M2 off-target: {run['case']} {where} [{off['got']}]: {off['unit']}")
        m12a = metrics.get("M12a")
        if m12a is not None and not m12a["identical"]:
            print(f"M12a first difference: {run['case']} at char {m12a['first_diff']}")
        if run.get("error"):
            print(f"error: {run['case']}: {run['error']}")
    calls = sum(run["calls"] for run in case_runs)
    scored = sum(run["metrics"]["M2"]["scored"] for run in case_runs)
    unique = len({run["case"] for run in case_runs})
    print(f"cases={unique} case_runs={len(case_runs)} calls={calls} scored_units={scored}")


def _select_cases(args: EvalArgs, tier_name: str) -> list[Case]:
    from eval import cases

    tier = cases.TIERS[tier_name]
    selected = list(tier) if not args.doc else []
    if args.case:
        chosen: set[str] = set()
        for selector in args.case:
            matched = {case.id for case in tier if selector in (case.id, case.doc)}
            if not matched:
                msg = f"--case {selector}: no case in tier {tier_name} matches"
                raise _RefusedError(msg)
            chosen |= matched
        selected = [case for case in tier if case.id in chosen]
    for doc in _adhoc_documents(args):
        types = cases.TIER_TYPES[tier_name]
        selected += [cases.Case(doc.id, t, "auto") for t in types]
        if tier_name == "lang":
            selected += [cases.Case(doc.id, t, "auto", split=True) for t in types]
    return selected


def _adhoc_documents(args: EvalArgs) -> list[ManifestDocument]:
    if len(args.label) != len(args.doc) or len(args.script) not in (0, len(args.doc)):
        msg = "each --doc needs a --label (and, if any is given, a --script)"
        raise _RefusedError(msg)
    scripts = args.script or [None] * len(args.doc)
    documents = []
    for path, label, script in zip(args.doc, args.label, scripts, strict=True):
        try:
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            msg = f"--doc {path}: {exc}"
            raise _RefusedError(msg) from exc
        documents.append(ManifestDocument(sha[:12], path.resolve(), label, script, sha))
    return documents


def _documents_for(
    args: EvalArgs, eval_dir: Path, runs: Sequence[RunSpec]
) -> list[ManifestDocument]:
    adhoc = {doc.id: doc for doc in _adhoc_documents(args)}
    needed = list(dict.fromkeys(spec.case.doc for spec in runs))
    corpus = [doc_id for doc_id in needed if doc_id not in adhoc]
    manifest = load_manifest(eval_dir) if corpus else {}
    lacking = [doc_id for doc_id in corpus if doc_id not in manifest]
    if lacking:
        msg = f"manifest lacks: {', '.join(lacking)}"
        raise _RefusedError(msg)
    documents = [manifest[doc_id] for doc_id in corpus]
    verify_documents(documents)
    return [*documents, *(adhoc[doc_id] for doc_id in needed if doc_id in adhoc)]


def _targets(runs: Sequence[RunSpec], documents: Mapping[str, ManifestDocument]) -> dict[str, Any]:
    from eval import metrics

    targets: dict[str, Any] = {}
    for spec in runs:
        doc = documents[spec.case.doc]
        try:
            targets[spec.case.id] = metrics.target_for(spec.case.language, doc.label, doc.script)
        except ValueError as exc:
            msg = f"{spec.case.id}: {exc}"
            raise _RefusedError(msg) from exc
    return targets


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _llama_cpp_install() -> tuple[str | None, str | None]:
    """The installed llama-cpp-python version and its library dir, without importing it."""
    try:
        version: str | None = importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        version = None
    lib_dir = os.environ.get("LLAMA_CPP_LIB_PATH")
    if not lib_dir:
        spec = importlib.util.find_spec("llama_cpp")
        lib_dir = str(Path(spec.origin).parent) if spec is not None and spec.origin else None
    return version, lib_dir


def _peak_rss_bytes() -> int | None:
    if sys.platform == "win32":
        import psutil

        return int(psutil.Process().memory_info().peak_wset)
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _default_n_ctx() -> int:
    from docsummarizer.model_manager import Summarizer

    return int(inspect.signature(Summarizer.__init__).parameters["n_ctx"].default)


def _call_lines(index: int, case_run: CaseRun) -> list[dict[str, Any]]:
    where = {"run": index, "case": case_run.case, "seed": case_run.seed}
    where["repetition"] = case_run.repetition
    return [{"kind": "call", **where, **asdict(call)} for call in case_run.calls]


def _case_run_line(index: int, case_run: CaseRun) -> dict[str, Any]:
    line = {"kind": "case_run", "run": index, **asdict(case_run)}
    line["calls"] = len(case_run.calls)
    return line


@dataclass(frozen=True)
class ModelBuild:
    """How a tier run loads the model: the file, its default context and the hardware split."""

    model: Path
    n_ctx: int
    n_threads: int | None
    n_gpu_layers: int


def _run_passes(
    build: ModelBuild,
    runs: Sequence[RunSpec],
    texts: Mapping[str, str],
    targets: Mapping[str, Any],
    sink: IO[str],
) -> tuple[list[dict[str, Any]], int | None]:
    """Run every pass, a split pass on its own model build; stream each case run's lines."""
    from eval import cases

    passes: list[tuple[int, list[RunSpec]]] = []
    for spec in runs:
        n_ctx = cases.SPLIT_N_CTX if spec.case.split else build.n_ctx
        if passes and passes[-1][0] == n_ctx:
            passes[-1][1].append(spec)
        else:
            passes.append((n_ctx, [spec]))
    lines: list[dict[str, Any]] = []
    threads: int | None = None

    def stream(case_run: CaseRun) -> None:
        line = _case_run_line(len(lines), case_run)
        for record in [*_call_lines(len(lines), case_run), line]:
            write_line(sink, record)
        sink.flush()
        lines.append(line)

    for n_ctx, specs in passes:
        summarizer = build_summarizer(
            build.model, n_ctx=n_ctx, n_threads=build.n_threads, n_gpu_layers=build.n_gpu_layers
        )
        threads = getattr(summarizer, "n_threads", build.n_threads) if threads is None else threads
        try:
            run_cases(summarizer, specs, texts, targets, on_run=stream)
        finally:
            summarizer.close()
    return lines, threads


def _run_tier(args: EvalArgs) -> int:
    from eval import cases, metrics

    if args.tier is None:
        msg = "no mode chosen"
        raise _RefusedError(msg)
    if tuple(args.seeds) != (0,):
        msg = "--seeds: seed 0 only until the seed API lands"
        raise _RefusedError(msg)
    if args.eval_dir is None or args.model is None:
        msg = "a tier run needs --eval-dir and --model"
        raise _RefusedError(msg)
    if not args.model.is_file():
        msg = f"model not found: {args.model}"
        raise _RefusedError(msg)
    args.n_ctx = args.n_ctx or _default_n_ctx()
    runs = expand_runs(_select_cases(args, args.tier), args.seeds, args.repeat)
    documents = {doc.id: doc for doc in _documents_for(args, args.eval_dir, runs)}
    targets = _targets(runs, documents)
    texts = {doc_id: doc.path.read_text(encoding="utf-8") for doc_id, doc in documents.items()}
    model_sha = _file_sha256(args.model)
    git = git_state(_REPO)
    package = _REPO / "src" / _PACKAGE
    gated = derive_gated(package)
    start = datetime.now(UTC)
    final = results_path(args.eval_dir, args.tier, (git.commit or "unknown")[:12], start)
    final.parent.mkdir(parents=True, exist_ok=True)
    partial = final.with_name(final.name + ".partial")
    with partial.open("w", encoding="utf-8") as sink:
        lines, threads = _run_passes(
            ModelBuild(args.model, args.n_ctx, args.threads, args.n_gpu_layers),
            runs,
            texts,
            targets,
            sink,
        )
    not_split = [
        line["case"]
        for line in lines
        if "@split" in line["case"] and line["parts"] < _SPLIT_MIN_PARTS
    ]

    for case_id in not_split:
        print(f"not split: {case_id}")
    version, lib_dir = _llama_cpp_install()
    gpu = f"gpu:{Path(lib_dir).name}" if lib_dir else "gpu"
    header = {
        "kind": "header",
        "metric_version": metrics.METRIC_VERSION,
        "commit": git.commit,
        "dirty": git.dirty,
        "git_error": git.error,
        "gated_hash": gated_hash(package, gated.files),
        "gated_files": list(gated.files),
        "gated_missing_roots": list(gated.missing_roots),
        "corpus_hash": corpus_hash(documents.values()),
        "documents": {
            doc.id: {
                "sha256": doc.sha256,
                **({"path": str(doc.path)} if doc.id not in cases.DOCUMENTS else {}),
            }
            for doc in documents.values()
        },
        "model_sha256": model_sha,
        "model_size": args.model.stat().st_size,
        "llama_cpp_version": version,
        "lib_dir": lib_dir,
        "backend": "cpu" if args.n_gpu_layers == 0 else gpu,
        "n_gpu_layers": args.n_gpu_layers,
        "threads": threads,
        "n_ctx": args.n_ctx,
        "split_n_ctx": cases.SPLIT_N_CTX if any(s.case.split for s in runs) else None,
        "seeds": list(args.seeds),
        "repeat": args.repeat,
        "repeat_scope": REPEAT_SCOPE,
        "tier": args.tier,
        "cases": list(dict.fromkeys(spec.case.id for spec in runs)),
        "not_split": not_split,
        "machine_id": machine_id(),
        "utc_start": start.isoformat(),
        "utc_end": datetime.now(UTC).isoformat(),
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "contention": [
            {"run": line["run"], "case": line["case"], **line["contention"]} for line in lines
        ],
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    with final.open("x", encoding="utf-8") as out:
        write_line(out, header)
        out.write(partial.read_text(encoding="utf-8"))
    partial.unlink()
    print(f"results: {final}")
    print_table(lines)
    calls = sum(line["calls"] for line in lines)
    return 2 if not lines or calls == 0 else 0


# --------------------------------------------------------------------------- #
# Rescore: the stored completions replayed through this tree, with no model
# --------------------------------------------------------------------------- #

# Coverage-only locates a quote by the coverage pre-filter alone; production
# also applies the strict citation check. The drop between them is its cost.
RESCORE_ARMS = ("coverage_only", "production")


class ReplayMismatchError(Exception):
    """This tree asked the model something the stored run did not."""

    def __init__(self, seq: int, field: str) -> None:
        """Name the call (``seq``) and the request field that differed."""
        super().__init__(f"seq {seq} field {field}")
        self.seq = seq
        self.field = field


def _normalized(value: object) -> object:
    return json.loads(json.dumps(value))


class ReplayLlama:
    """Answers each completion with the stored one, once the request matches it.

    It has no ``ctx``, so the pipeline reads no counters and installs no abort
    callback. The first mismatch is kept: the pipeline may swallow the raise and
    fall back, and the replay must still report the first difference.
    """

    def __init__(self, calls: Sequence[Mapping[str, Any]], tokenized: Mapping[str, int]) -> None:
        """Replay ``calls`` in order; ``tokenized`` maps a text hash to its token count."""
        self._calls: list[Mapping[str, Any]] = list(calls)
        self._tokenized: dict[str, int] = dict(tokenized)
        self.consumed: int = 0
        self.mismatch: ReplayMismatchError | None = None

    def _refuse(self, field: str) -> ReplayMismatchError:
        if self.mismatch is None:
            self.mismatch = ReplayMismatchError(self.consumed, field)
        return self.mismatch

    @overload
    def create_chat_completion(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        stream: Literal[False] = False,
        **kwargs: object,
    ) -> CreateChatCompletionResponse: ...

    @overload
    def create_chat_completion(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        stream: Literal[True],
        **kwargs: object,
    ) -> Iterator[CreateChatCompletionStreamResponse]: ...

    def create_chat_completion(
        self, *, messages: Sequence[Mapping[str, str]], stream: bool = False, **kwargs: object
    ) -> CreateChatCompletionResponse | Iterator[CreateChatCompletionStreamResponse]:
        """Return the stored answer when the request matches the stored one, else refuse.

        A replay runs without a cancel check, so the pipeline never streams to it.
        """
        if stream:
            msg = "stream"
            raise self._refuse(msg)
        if self.mismatch is not None:
            raise self.mismatch
        if self.consumed >= len(self._calls):
            msg = "call count"
            raise self._refuse(msg)
        stored = self._calls[self.consumed]
        if _normalized([dict(message) for message in messages]) != stored["messages"]:
            msg = "messages"
            raise self._refuse(msg)
        asked = {key: _normalized(_jsonable(value)) for key, value in kwargs.items()}
        for key in sorted(set(asked) | set(stored["kwargs"])):
            if asked.get(key) != stored["kwargs"].get(key):
                raise self._refuse(key)
        self.consumed += 1
        if stored["error"]:
            raise RuntimeError(stored["error"])  # the stored call failed: fail the same way
        message: ChatCompletionResponseMessage = {
            "role": "assistant",
            "content": stored["content"],
        }
        choice: ChatCompletionResponseChoice = {
            "message": message,
            "finish_reason": stored["finish_reason"],
        }
        return {"choices": [choice], "usage": stored["usage"]}

    def tokenize(self, text: bytes, *, add_bos: bool = True) -> list[int]:
        """Placeholder tokens, as many as the stored run counted for this text."""
        del add_bos  # the stored count already reflects it
        # Message equality still guards an unknown text: any chunking difference
        # changes the next request.
        return [0] * self._tokenized.get(hashlib.sha256(text).hexdigest(), len(text) // 3)


def replay_summarizer(llm: ReplayLlama, n_ctx: int) -> Summarizer:
    """A summarizer around ``llm`` with no model loaded: the replay's construction site."""
    from docsummarizer.model_manager import Summarizer

    summarizer = object.__new__(Summarizer)
    summarizer.llm = llm
    summarizer.n_ctx = n_ctx
    return summarizer


@contextmanager
def accept_every_match() -> Iterator[None]:
    """Accept every candidate sentence the coverage pre-filter keeps, then restore."""
    from docsummarizer import provenance

    original = provenance.is_verified_match

    def accept(quote: str, span_text: str) -> bool:
        del quote, span_text  # every candidate is accepted
        return True

    provenance.is_verified_match = accept

    try:
        yield
    finally:
        provenance.is_verified_match = original


def _cited_points(result: Mapping[str, Any]) -> list[tuple[int, Mapping[str, Any]]]:
    """``(index, point)`` for each cited point: the Detailed points, then each section."""
    sections = result.get("sections") or {}
    points = [*(result.get("points") or []), *(p for s in sections.values() for p in s)]
    return [(i, point) for i, point in enumerate(points) if point.get("citation") is not None]


def audit_citations(
    result: Mapping[str, Any], quotes: Mapping[str, str], document: str
) -> list[str]:
    """Each cited point whose span fails to hold the model's quote or the document's text.

    A span fails when it does not verifiably hold the quote, or is not the
    document's own text at its offsets.
    """
    from docsummarizer import provenance

    failures: list[str] = []
    for index, point in _cited_points(result):
        span = point["citation"]
        if not provenance.is_verified_match(quotes.get(point["text"], ""), span["quote"]):
            failures.append(f"point {index}: the span does not hold the quote")
        elif document[span["start"] : span["end"]] != span["quote"]:
            failures.append(f"point {index}: the span is not the document's text")
    return failures


def _case_of(case_id: str) -> Case:
    """The case a results line names: ``<doc>/<type>/<language>[@split][#m12a]``."""
    from eval import cases

    m12a = case_id.endswith("#m12a")
    body = case_id.removesuffix("#m12a")
    split = body.endswith("@split")
    doc, summary_type, language = body.removesuffix("@split").split("/", 2)
    return cases.Case(doc, summary_type, language, split=split, m12a=m12a)


def _call_record(line: Mapping[str, Any]) -> CallRecord:
    return CallRecord(**{f.name: line[f.name] for f in fields(CallRecord)})


def _replay(
    case: Case, text: str, n_ctx: int, line: Mapping[str, Any], calls: Sequence[Mapping[str, Any]]
) -> tuple[StructuredSummary | None, str | None]:
    """One arm's replay of a case run: its result or the error it raised.

    Raises ``ReplayMismatchError`` when this tree asks anything the stored run did
    not, or fewer calls than it made.
    """
    llm = ReplayLlama(calls, line["tokenized"])
    summarizer = replay_summarizer(llm, n_ctx)
    result: StructuredSummary | None = None
    error: str | None = None
    try:
        result = summarizer.summarize_structured(text, case.summary_type, language=case.language)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    if llm.mismatch is not None:
        raise llm.mismatch
    if llm.consumed != len(calls):
        raise ReplayMismatchError(llm.consumed, "call count")
    return result, error


def _rescore_documents(eval_dir: Path, header: Mapping[str, Any]) -> dict[str, ManifestDocument]:
    manifest = load_manifest(eval_dir)
    documents: dict[str, ManifestDocument] = {}
    for doc_id, recorded in header["documents"].items():
        doc = manifest.get(doc_id)
        if doc is None:
            msg = f"{doc_id}: not in the manifest (ad-hoc runs are not rescored)"
            raise _RefusedError(msg)
        if doc.sha256 != recorded["sha256"]:
            msg = f"{doc_id}: the manifest's sha256 differs from the run's"
            raise _RefusedError(msg)
        documents[doc_id] = doc
    verify_documents(documents.values())
    return documents


def _arm_sums(m7s: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    keys = ("points", "cited", "tight", "whole_document", "quotes", "verbatim")
    return {key: sum(m7[key] for m7 in m7s) for key in keys}


def print_arms(case_runs: Sequence[Mapping[str, Any]]) -> None:
    """M7 per cell and aggregate in both arms, and the cited share the strict check costs."""
    scored = [
        run
        for run in case_runs
        if run.get("repetition", 0) == 0 and run["metrics"]["M7"] is not None
    ]
    if not scored:
        return
    cells: dict[str, list[Mapping[str, Any]]] = {}
    for run in scored:
        cells.setdefault("/".join(run["metrics"]["cell"]), []).append(run)
    for scope, runs in [*sorted(cells.items()), ("aggregate", scored)]:
        production = _arm_sums([run["metrics"]["M7"] for run in runs])
        covered = [run["m7_coverage_only"] for run in runs]
        coverage = None if None in covered else _arm_sums(covered)
        for arm, sums in zip(RESCORE_ARMS, (coverage, production), strict=True):
            if sums is None:
                print(f"M7 {scope} {arm} n/a")
                continue
            n = sums["points"]
            print(
                f"M7 {scope} {arm} cited={sums['cited']}/{n} tight={sums['tight']}/{n} "
                f"whole_document={sums['whole_document']}/{n} "
                f"verbatim={sums['verbatim']}/{sums['quotes']}"
            )
        drop = "n/a" if coverage is None else f"{coverage['cited'] - production['cited']}"
        print(f"M7 {scope} drop cited={drop}/{production['points']}")


def _rescore_one(
    line: Mapping[str, Any],
    stored: Sequence[Mapping[str, Any]],
    doc: ManifestDocument,
    text: str,
    done: Sequence[CaseRun],
) -> tuple[CaseRun, dict[str, Any] | None]:
    """Replay one case run in both arms: the production case run and the coverage-only M7.

    A production-arm mismatch raises ``ReplayMismatchError``; a coverage-only one is
    printed and leaves that arm's M7 ``None``.
    """
    from eval import metrics

    case = _case_of(line["case"])
    try:
        target = metrics.target_for(case.language, doc.label, doc.script)
    except ValueError as exc:
        msg = f"{case.id}: {exc}"
        raise _RefusedError(msg) from exc
    records = [_call_record(call) for call in stored]
    asked = ScoringInput(case, text, target, line["parts"])
    result, error = _replay(case, text, line["n_ctx"], line, stored)
    coverage_m7: dict[str, Any] | None = None
    try:
        with accept_every_match():
            covered, _ = _replay(case, text, line["n_ctx"], line, stored)
    except ReplayMismatchError as exc:
        print(f"coverage-only arm: replay mismatch: {exc} ({case.id})")
    else:
        coverage_m7 = score_case_run(asked, covered, records)["M7"]
    result_dict = asdict(result) if result is not None else None
    if result_dict is not None:
        result_dict.pop("run_stats", None)
    case_run = CaseRun(
        case=case.id,
        seed=line["seed"],
        repetition=line["repetition"],
        n_ctx=line["n_ctx"],
        parts=line["parts"],
        result=result_dict,
        run_stats=line["run_stats"],
        calls=records,
        tokenized=line["tokenized"],
        contention=line["contention"],
        contended=line["contended"],
        error=error,
    )
    case_run.metrics = score_case_run(asked, result, records)
    if case.m12a:
        case_run.metrics["M12a"] = _m12a(_first_run_of(case, case_run.seed, done), case_run)
    return case_run, coverage_m7


def _rescore_header(source: Path, header: Mapping[str, Any]) -> dict[str, Any]:
    """The generating run's header, plus the tree that scored it again."""
    from eval import metrics

    git = git_state(_REPO)
    package = _REPO / "src" / _PACKAGE
    scored_by = {
        "commit": git.commit,
        "dirty": git.dirty,
        "gated_hash": gated_hash(package, derive_gated(package).files),
    }
    return {
        **header,
        "metric_version": metrics.METRIC_VERSION,
        "scored_by": scored_by,
        "rescored_from": str(source),
        "arms": list(RESCORE_ARMS),
    }


def rescore(eval_dir: Path, source: Path) -> int:
    """Replay ``source``'s completions through this tree in both arms and score them again.

    The production arm's scores become each line's ``metrics``; the
    coverage-only arm's M7 is kept beside them as ``m7_coverage_only``.
    """
    try:
        results = read_results(source)
    except (OSError, ValueError) as exc:
        msg = f"--rescore {source}: {exc}"
        raise _RefusedError(msg) from exc
    if not results.case_runs:
        msg = f"--rescore {source}: zero case runs"
        raise _RefusedError(msg)
    documents = _rescore_documents(eval_dir, results.header)
    texts = {doc_id: doc.path.read_text(encoding="utf-8") for doc_id, doc in documents.items()}
    calls_of: dict[int, list[dict[str, Any]]] = {}
    for call in results.calls:
        calls_of.setdefault(call["run"], []).append(call)

    done: list[CaseRun] = []
    lines: list[dict[str, Any]] = []
    citations = 0
    failures: list[str] = []
    for line in results.case_runs:
        doc_id = _case_of(line["case"]).doc
        stored, text = calls_of.get(line["run"], []), texts[doc_id]
        try:
            case_run, coverage_m7 = _rescore_one(line, stored, documents[doc_id], text, done)
        except ReplayMismatchError as exc:
            print(f"replay mismatch: {exc} ({line['case']})")
            return 2
        done.append(case_run)
        if case_run.result is not None:
            citations += len(_cited_points(case_run.result))
            audit = audit_citations(case_run.result, _model_quotes(case_run.calls), text)
            failures += [f"{case_run.case} {failure}" for failure in audit]
        lines.append({**_case_run_line(line["run"], case_run), "m7_coverage_only": coverage_m7})

    header = _rescore_header(source, results.header)
    commit = header["scored_by"]["commit"] or "unknown"
    plain = results_path(eval_dir, str(header["tier"]), commit[:12], datetime.now(UTC))
    final = plain.with_name(f"{plain.stem}-rescore.jsonl")
    final.parent.mkdir(parents=True, exist_ok=True)
    with final.open("x", encoding="utf-8") as out:
        write_line(out, header)
        for line in lines:
            for call in calls_of.get(line["run"], []):
                write_line(out, call)
            write_line(out, line)
    print(f"results: {final}")
    print_table(lines)
    print_arms(lines)
    print(f"replayed case_runs={len(lines)} calls={len(results.calls)}")
    for failure in failures:
        print(f"audit: {failure}")
    print(f"audit: {citations} citations, {len(failures)} failed")
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# Compare: a candidate against a baseline, only when both measured the same thing
# --------------------------------------------------------------------------- #

COMPARE_FIELDS = ("machine_id", "backend", "metric_version", "corpus_hash", "seeds", "cases")


def _case_scores(case_runs: Sequence[Mapping[str, Any]]) -> list[CaseScore]:
    """A verdict's view of each repetition-0 case run; repetitions exist for timing only."""
    from eval import metrics

    scores: list[CaseScore] = []
    for line in case_runs:
        if line.get("repetition", 0) != 0:
            continue
        scored = line["metrics"]
        hard = {
            name: int(scored[name]["violations"])
            for name in metrics.HARD_METRICS
            if scored.get(name) is not None
        }
        m7 = scored.get("M7")
        proportion = {"M7": (int(m7["tight"]), int(m7["points"]))} if m7 is not None else {}
        cell = scored["cell"]
        scores.append(
            metrics.CaseScore(
                line["case"], line["seed"], (cell[0], cell[1], cell[2]), hard, proportion
            )
        )
    return scores


def _share_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def compare(eval_dir: Path | None, b: Path, c: Path) -> int:
    """Compare candidate ``c`` with baseline ``b``: 2 when not comparable, 1 on a regression.

    A bare file name that does not exist as given is read from ``<eval-dir>/results/``.
    """
    from eval import metrics

    arms: list[Results] = []
    for given in (b, c):
        path = given  # the file actually read
        if not given.exists() and eval_dir is not None and given.name == str(given):
            path = eval_dir / "results" / given
        try:
            arms.append(read_results(path))
        except (OSError, ValueError) as exc:
            msg = f"--compare {path}: {exc}"
            raise _RefusedError(msg) from exc
    base, candidate = arms
    lacking = [
        f"{arm} {name}"
        for arm, results in (("B", base), ("C", candidate))
        for name in COMPARE_FIELDS
        if name not in results.header
    ]
    if lacking:
        msg = f"header lacks: {', '.join(lacking)}"
        raise _RefusedError(msg)
    differ = [name for name in COMPARE_FIELDS if base.header[name] != candidate.header[name]]
    if differ:
        msg = f"the arms differ in {', '.join(differ)}"
        raise _RefusedError(msg)
    b_scores, c_scores = _case_scores(base.case_runs), _case_scores(candidate.case_runs)
    if not b_scores or not c_scores:
        msg = "zero case runs"
        raise _RefusedError(msg)

    outcome = metrics.verdict(b_scores, c_scores)
    print(f"verdict: B {len(b_scores)} case runs, C {len(c_scores)} case runs (repetition 0)")
    print(f"{'scope':<34} {'metric':<9} {'B':>6} {'n':>5} {'C':>6} {'n':>5} {'delta':>6}  status")
    for row in outcome.rows:
        print(
            f"{row.scope:<34} {row.metric:<9} {_share_text(row.p_b):>6} {row.n_b:>5} "
            f"{_share_text(row.p_c):>6} {row.n_c:>5} {_share_text(row.delta):>6}  {row.status}"
        )
    for flip in outcome.regressions:
        print(f"regression: {flip.case} seed {flip.seed} {flip.metric}")
    for flip in outcome.improvements:
        print(f"improvement: {flip.case} {flip.metric} (every seed)")
    print("A hard flip blocks a merge unless a recorded reader marks it spurious.")
    return 1 if outcome.regressed else 0


# --------------------------------------------------------------------------- #
# Checks, the language self-test and the standalone contention count
# --------------------------------------------------------------------------- #

# The fixed input of every check: project-written, about invented places.
CHECK_TEXT = (
    "Every morning at six the small ferry leaves the wooden pier on the north bank of the "
    "river. Its captain has worked the crossing for twenty years, and she checks the ropes, "
    "the engine and the life jackets before anyone is allowed to board. The first passengers "
    "are usually the bakers from the village, who carry baskets of bread to the market on "
    "the far side. Behind them come farmers with crates of eggs and early vegetables, and "
    "a few office workers who prefer the quiet water to the crowded bridge further "
    "downstream.\n\n"
    "At half past seven the ferry fills with children. The village has no secondary school "
    "of its own, so more than forty students cross the river twice a day. The school pays "
    "for their tickets, and the captain keeps a list of every child who should be on board. "
    "When the river floods in spring, the crossing is cancelled, and the students study at "
    "home with lessons sent by their teachers. Last year the river closed the crossing for "
    "eleven days, the longest stop anyone could remember.\n\n"
    "The school itself sits on a low hill above the market square. Its garden was planted "
    "five years ago by a group of parents who wanted the children to learn where food comes "
    "from. Each class looks after one bed. The youngest children grow beans and radishes, "
    "which come up quickly, while the older ones manage potatoes, onions and a long row of "
    "tall yellow sunflowers along the fence. In autumn the harvest is sold at a stall in the "
    "market, and the money buys seeds and tools for the next season.\n\n"
    "The bakery that supplies the school kitchen opens at five. Its owner bakes rye loaves, "
    "white rolls and, on Fridays, a sweet bread with raisins that the students queue for at "
    "lunch. She buys her flour from a mill two villages away and sends her delivery van "
    "across on the ferry, because the bridge adds half an hour to the journey. When the "
    "crossing is closed she bakes less, since the market on the far side cannot be reached "
    "in time.\n\n"
    "The village council has discussed building a second, larger ferry. Supporters say it "
    "would carry the delivery vans and a school bus at once. Others worry about the cost and "
    "about the old pier, which would need to be rebuilt. A decision is expected next year, "
    "after the council has counted how many people use the crossing in each season."
)


def _perf_sums(calls: Sequence[CallRecord]) -> dict[str, float]:
    return {name: sum((call.perf or {}).get(name, 0) for call in calls) for name in _PERF_FIELDS}


_SECONDS_TOLERANCE = 1e-6  # run_stats seconds against the counters' milliseconds


def _counter_failures(stats: RunStats, sums: Mapping[str, float]) -> list[str]:
    """The run_stats fields that disagree with the summed counter deltas."""
    failed = [
        key
        for key, got, want in (
            ("prompt_evaluated", stats.prompt_evaluated, sums["n_p_eval"]),
            ("generated_tokens", stats.generated_tokens, sums["n_eval"]),
        )
        if got != want
    ]
    for key, got, want_ms in (
        ("prompt_seconds", stats.prompt_seconds, sums["t_p_eval_ms"]),
        ("generation_seconds", stats.generation_seconds, sums["t_eval_ms"]),
    ):
        if got is None or abs(got - want_ms / 1000) > _SECONDS_TOLERANCE:
            failed.append(key)
    return failed


def _usage_failures(last: CallRecord, n_tokens: object) -> list[str]:
    """The last call's usage against the context's token count, within one token."""
    usage = last.usage or {}
    n_eval = (last.perf or {}).get("n_eval", 0)
    if not isinstance(n_tokens, int) or "prompt_tokens" not in usage:
        return ["usage"]
    failed: list[str] = []
    prompt_tokens = usage["prompt_tokens"]
    if abs((n_tokens - n_eval) - prompt_tokens) > 1:
        failed.append(
            f"prompt_tokens (usage {prompt_tokens}, n_tokens - n_eval {n_tokens - n_eval:g})"
        )
    if abs(n_eval - usage.get("completion_tokens", -10)) > 1:
        failed.append(
            f"completion_tokens (usage {usage.get('completion_tokens')}, n_eval {n_eval:g})"
        )
    return failed


def _runstats_leg(
    name: str, summarizer: Summarizer, capture: CapturingLlama, *, reset: bool, stream: bool
) -> list[str] | None:
    """Run one leg and print its figures; the failed fields, or ``None`` when no call was made."""
    capture.calls = []
    if reset:
        capture.reset()
    cancel = (lambda: False) if stream else None
    result = summarizer.summarize_structured(CHECK_TEXT, "brief", should_cancel=cancel)
    if not capture.calls:
        return None
    stats, sums, last = result.run_stats, _perf_sums(capture.calls), capture.calls[-1]
    n_tokens = getattr(capture, "n_tokens", None)
    print(
        f"leg {name}: calls={len(capture.calls)} deltas "
        + " ".join(f"{key}={sums[key]:g}" for key in _PERF_FIELDS)
        + " | run_stats "
        + (
            "none"
            if stats is None
            else " ".join(
                f"{key}={getattr(stats, key)}"
                for key in (
                    "prompt_tokens",
                    "prompt_evaluated",
                    "generated_tokens",
                    "prompt_seconds",
                    "generation_seconds",
                )
            )
        )
        + f" | usage={last.usage} n_tokens={n_tokens}"
    )
    if stats is None:
        return ["run_stats"]
    failed = _counter_failures(stats, sums)
    if name == "blocking":
        failed.extend(_usage_failures(last, n_tokens))
    evaluated, prompt_tokens = stats.prompt_evaluated, stats.prompt_tokens
    if name == "prefix repeat" and not (
        evaluated is not None and prompt_tokens is not None and evaluated < prompt_tokens
    ):
        failed.append(f"prefix (prompt_evaluated {evaluated}, prompt_tokens {prompt_tokens})")

    return failed


def check_runstats(args: EvalArgs) -> int:
    """The request counters against llama.cpp's own: blocking, streamed, and a prefix repeat."""
    if args.model is None or not args.model.is_file():
        print("runstats: needs --model <gguf>")
        return 2
    summarizer = build_summarizer(
        args.model,
        n_ctx=args.n_ctx or _default_n_ctx(),
        n_threads=args.threads,
        n_gpu_layers=args.n_gpu_layers,
    )
    try:
        capture = capture_calls(summarizer)
        if summarizer.perf_snapshot() is None:
            print("Counters unavailable: llama_perf_context missing")
            return 2
        passed = 0
        legs = (("blocking", True, False), ("streamed", True, True), ("prefix repeat", False, True))
        for name, reset, stream in legs:
            failed = _runstats_leg(name, summarizer, capture, reset=reset, stream=stream)
            if failed is None:
                print(f"runstats: leg {name} made no call")
                return 2
            for reason in failed:
                print(f"leg {name}: fail {reason}")
            if not failed:
                print(f"leg {name}: pass")
                passed += 1
    finally:
        summarizer.close()
    print(f"runstats: {len(legs)} legs, {passed} passed")
    return 0 if passed == len(legs) else 1


CHECKS: dict[str, Callable[[EvalArgs], int]] = {"runstats": check_runstats}


def run_checks(listed: str, args: EvalArgs) -> int:
    """Run each check ``listed`` names; 2 for an unknown or empty list, else the worst exit code."""
    names = [name.strip() for name in listed.split(",") if name.strip()]
    unknown = [name for name in names if name not in CHECKS]
    if not names or unknown:
        print(f"refused: --checks {listed!r}: known checks are {', '.join(CHECKS)}")
        return 2
    return max(CHECKS[name](args) for name in names)


SELF_TEST_MIN_SENTENCES = 1000
SELF_TEST_MIN_LANGUAGES = 10
_SELF_TEST_OUTPUTS = 1000
_DRIFT_LANGUAGES = ("zh", "de", "en", "es", "fr")
# Bars: sentence error, run-level false fail, missed drift.
_SELF_TEST_BARS = (0.015, 0.03, 0.01)


def _rate_line(name: str, num: int, den: int, bar: float) -> bool:
    rate = num / den if den else 1.0
    ok = rate <= bar
    print(f"{name} {num}/{den} ({rate:.1%}), bar {bar:.1%}: {'ok' if ok else 'MISSED'}")
    return ok


def self_test_language(path: Path) -> int:
    """The M2 classifier's three error rates over a labelled set of sentences."""
    from eval import metrics

    by_label: dict[str, list[str]] = {}
    skipped = 0
    try:
        raw = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"refused: --self-test language {path}: {exc}")
        return 2
    for text_line in raw:
        try:
            row = json.loads(text_line) if text_line.strip() else None
        except ValueError:
            row = None
        text = row.get("text") if isinstance(row, dict) else None
        label = row.get("label") if isinstance(row, dict) else None
        if not isinstance(text, str) or not isinstance(label, str) or not text or not label:
            skipped += 1 if text_line.strip() else 0
            continue
        by_label.setdefault(label, []).append(text)
    sentences = sum(len(texts) for texts in by_label.values())
    print(f"self-test language: {sentences} sentences, {len(by_label)} languages")
    if skipped:
        print(f"skipped {skipped} row{'s' if skipped != 1 else ''} without text or label")
    if sentences < SELF_TEST_MIN_SENTENCES or len(by_label) < SELF_TEST_MIN_LANGUAGES:
        print(
            f"refused: needs at least {SELF_TEST_MIN_SENTENCES} sentences in "
            f"{SELF_TEST_MIN_LANGUAGES} languages"
        )
        return 2

    scored = [
        (t, label) for label, texts in by_label.items() for t in texts if metrics.is_scored(t)
    ]
    wrong = sum(1 for text, label in scored if metrics.classify_unit(text) != label)
    rng = random.Random(0)  # a reproducible sample, not a secret
    labels = sorted(by_label)
    false_fail = missed = 0
    for _ in range(_SELF_TEST_OUTPUTS):
        label = rng.choice(labels)
        target = metrics.Target(label, None)
        clean = [rng.choice(by_label[label]) for _ in range(4)]
        false_fail += not metrics.m2([clean], [], target).passed
        others = [code for code in _DRIFT_LANGUAGES if code != label and code in by_label]
        drift = [*clean[:3], rng.choice(by_label[rng.choice(others)])] if others else clean
        missed += metrics.m2([drift], [], target).passed
    results = (
        _rate_line("sentence error", wrong, len(scored), _SELF_TEST_BARS[0]),
        _rate_line("false fail", false_fail, _SELF_TEST_OUTPUTS, _SELF_TEST_BARS[1]),
        _rate_line("missed drift", missed, _SELF_TEST_OUTPUTS, _SELF_TEST_BARS[2]),
    )
    return 0 if all(results) else 1


def run_self_test(kind_and_arg: Sequence[str]) -> int:
    """``--self-test KIND [ARG]``; the only kind so far is ``language <set>``."""
    match kind_and_arg:
        case ["language", labelled_set]:
            return self_test_language(Path(labelled_set))
        case _:
            print("refused: --self-test takes: language <labelled set>")
            return 2


# The default window of a standalone contention count, in seconds.
CONTENTION_WINDOW_S: float = 10.0


def _idle_names(side: SideCount) -> str:
    return ",".join(side.idle) or "-"


def contention_report(seconds: float) -> int:
    """Count model processes using the CPU over ``seconds``, on both sides of a WSL run."""
    if seconds <= 0:
        print(f"refused: --contention {seconds:g}: the window must be above 0 seconds")
        return 2
    start = snapshot_now()
    time.sleep(seconds)
    outcome = count_contention(start, snapshot_now())
    local, host = outcome.local, outcome.host
    if host is None:
        host_text = "n/a"
    elif host.count is None:
        host_text = "unknown"
    else:
        host_text = f"{host.count} idle={_idle_names(host)}"
    print(
        f"contention local={local.count} unknown={local.unknown} idle={_idle_names(local)} "
        f"host={host_text}"
    )
    return 1 if outcome.contended else 0


def _run_mode(args: EvalArgs) -> int:
    """The one mode the command line chose; a tier run when no other is."""
    if args.rescore is not None and args.eval_dir is not None:
        return rescore(args.eval_dir, args.rescore)
    if args.compare is not None:
        baseline, candidate = args.compare
        return compare(args.eval_dir, baseline, candidate)
    if args.checks is not None:
        return run_checks(args.checks, args)
    if args.self_test is not None:
        return run_self_test(args.self_test)
    if args.contention is not None:
        return contention_report(args.contention)
    return _run_tier(args)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse, bootstrap app data, then run the chosen mode."""
    args = build_parser().parse_args(argv, namespace=EvalArgs())
    if args.rescore is not None and args.eval_dir is None:
        print("refused: --rescore needs --eval-dir")
        return 2
    if args.eval_dir is not None:
        app_data = args.eval_dir / "appdata"
    else:
        app_data = Path(tempfile.mkdtemp(prefix="docsummarizer-eval-"))
        print(f"app data: {app_data}")
    bootstrap(app_data)
    try:
        return _run_mode(args)
    except (_RefusedError, ManifestError) as exc:
        print(f"refused: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
