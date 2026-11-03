"""The evaluation runner: runs cases against a local corpus and records what it measured.

Module-level imports are the standard library only. ``bootstrap()`` decides
which checkout ``docsummarizer`` resolves from and where its app data goes, so
``cases``, ``metrics`` and every ``docsummarizer`` module are imported inside
the functions that use them, after it has run.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from docsummarizer.model_manager import StructuredSummary, Summarizer
    from eval.cases import Case
    from eval.metrics import Target

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
        raise SystemExit(f"docsummarizer resolves from {location.parent}, not {_REPO / 'src'}")


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
    head = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [git, "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=False
    )
    if head.returncode != 0:
        return GitState(None, None, head.stderr.strip() or f"git exited {head.returncode}")
    status = subprocess.run(  # noqa: S603 - fixed argv, no shell
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
_REQUIRED = ("id", "file", "label", "sha256")


def _manifest_document(entry: Any, corpus: Path) -> ManifestDocument:
    from eval import cases

    if not isinstance(entry, dict) or not all(isinstance(entry.get(k), str) for k in _REQUIRED):
        raise ManifestError(f"manifest unreadable: an entry lacks one of {_REQUIRED}")
    doc_id = entry["id"]
    shape = cases.DOCUMENTS.get(doc_id)
    if shape is None:
        raise ManifestError(f"manifest unknown: {doc_id} (ad-hoc documents go through --doc)")
    if entry["label"] != shape.label:
        raise ManifestError(f"manifest label: {doc_id} ({entry['label']!r}, not {shape.label!r})")
    script = entry.get("script")
    if shape.label == "zh" and (
        script not in _SCRIPTS or (shape.script is not None and script != shape.script)
    ):
        raise ManifestError(f"manifest script: {doc_id} ({script!r})")
    path = (corpus / entry["file"]).resolve()
    if not path.is_relative_to(corpus.resolve()):
        raise ManifestError(f"manifest file outside the corpus: {doc_id}")
    return ManifestDocument(doc_id, path, entry["label"], script, entry["sha256"])


def load_manifest(eval_dir: Path) -> dict[str, ManifestDocument]:
    """Read ``<eval-dir>/corpus/manifest.json`` and check it against the case matrix."""
    corpus = eval_dir / "corpus"
    manifest = corpus / "manifest.json"
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ManifestError(f"manifest absent: {manifest}") from None
    except (OSError, ValueError) as exc:
        raise ManifestError(f"manifest unreadable: {manifest}: {exc}") from exc
    entries = data.get("documents") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise ManifestError(f"manifest unreadable: {manifest}: no documents list")
    documents: dict[str, ManifestDocument] = {}
    for entry in entries:
        doc = _manifest_document(entry, corpus)
        if doc.id in documents:
            raise ManifestError(f"manifest duplicate: {doc.id}")
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
        if (self.local.count or 0) > 0:
            return True
        return self.host is not None and (self.host.count or 0) > 0


def _basename(path: str) -> str:
    return PureWindowsPath(path).name.lower()  # splits on both separators


def under_wsl() -> bool:
    """Whether this process runs under WSL, with a Windows host beside it."""
    return "microsoft" in platform.uname().release.lower()


def read_tasklist() -> str | None:
    """The Windows host's process table as CSV; ``None`` without interop."""
    try:
        done = subprocess.run(
            ["tasklist.exe", "/v", "/fo", "csv", "/nh"],  # noqa: S607 - resolved on the interop PATH
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def _clock_seconds(text: str) -> float:
    hours, minutes, seconds = (int(part) for part in text.split(":"))
    return float(hours * 3600 + minutes * 60 + seconds)


def parse_tasklist(text: str) -> dict[int, tuple[str, float]]:
    """Matching host processes: image name (column 0), pid (1) and CPU time (7)."""
    found: dict[int, tuple[str, float]] = {}
    for row in csv.reader(io.StringIO(text)):
        try:
            image, pid, cpu = row[0], int(row[1]), _clock_seconds(row[7])
        except (IndexError, ValueError):
            continue
        if any(marker in image.lower() for marker in _MODEL_MARKERS):
            found[pid] = (image, cpu)
    return found


def take_snapshot(
    processes: Iterable[Any], own_pid: int, tasklist: Callable[[], str | None] | None
) -> ContentionSnapshot:
    """Record every matching process's CPU time, and the host's when ``tasklist`` is given.

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
    if tasklist is None:
        return ContentionSnapshot(local, unknown, None, "n/a")
    text = tasklist()
    if text is None:
        return ContentionSnapshot(local, unknown, None, "unknown")
    return ContentionSnapshot(local, unknown, parse_tasklist(text), "ok")


def snapshot_now() -> ContentionSnapshot:
    """A snapshot of this machine, and of the Windows host under WSL."""
    import psutil

    return take_snapshot(psutil.process_iter(), os.getpid(), read_tasklist if under_wsl() else None)


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
    usage: dict[str, Any] | None
    perf: dict[str, float] | None  # the call's counter deltas; None without counters
    error: str | None


def _jsonable(value: Any) -> Any:
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

    def __init__(self, real: Any, snapshot: Callable[[], Any]) -> None:
        self._real = real
        self._snapshot = snapshot
        self.calls: list[CallRecord] = []
        self.tokenized: dict[str, int] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def _perf_since(self, before: Any) -> dict[str, float] | None:
        after = self._snapshot() if before is not None else None
        if before is None or after is None:
            return None
        return {name: getattr(after, name) - getattr(before, name) for name in _PERF_FIELDS}

    def create_chat_completion(self, **kwargs: Any) -> Any:
        before = self._snapshot()
        messages = kwargs.pop("messages")
        stream = bool(kwargs.pop("stream", False))
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
            response = self._real.create_chat_completion(messages=messages, **kwargs)
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

    def _stream(self, record: CallRecord, before: Any, chunks: Iterable[Any]) -> Iterator[Any]:
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

    def tokenize(self, text: bytes, add_bos: bool = True) -> Any:
        result = self._real.tokenize(text, add_bos=add_bos)
        self.tokenized[hashlib.sha256(text).hexdigest()] = len(result)
        return result

    def reset(self) -> None:
        self._real.reset()

    def set_seed(self, seed: int) -> None:
        self._real.set_seed(seed)


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
    from docsummarizer.model_manager import _parse_structured_json
    from eval import metrics

    parsed = _parse_structured_json(content)
    if parsed is None:
        return None
    texts = [str(parsed.get("lead") or ""), *(text for text, _ in _json_points(parsed))]
    return [unit for text in texts for unit in metrics.split_units(text)]


def _json_points(parsed: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Every ``(text, quote)`` pair of a parsed call: Detailed points and each section."""
    from docsummarizer.model_manager import _read_raw_points

    pairs = _read_raw_points(parsed.get("points"))
    sections = parsed.get("sections")
    if isinstance(sections, dict):
        for value in sections.values():
            pairs.extend(_read_raw_points(value))
    return pairs


def _result_points(result: StructuredSummary) -> list[Any]:
    points = list(result.points)
    for section in (result.sections or {}).values():
        points.extend(section)
    return points


def score_case_run(
    case: Case,
    result: StructuredSummary | None,
    calls: Sequence[CallRecord],
    text: str,
    target: Target,
    *,
    parts: int,
) -> dict[str, Any]:
    """Score one case run: every call's raw content and the returned result.

    The one scoring path a run and a rescore share. The rendered ``text`` of a
    result is never scored.
    """
    from docsummarizer.model_manager import SUMMARY_TYPE_BRIEF, _parse_structured_json
    from eval import metrics

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
            quotes: dict[str, str] = {}
            for call, json_call in zip(calls, is_json, strict=True):
                parsed = _parse_structured_json(call.content) if json_call else None
                if parsed is not None:
                    quotes.update(_json_points(parsed))
            scored = metrics.m7(metrics.point_records(points, quotes), text)
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
    """``_chunks_within``, recording the piece count of each top-level call."""
    original = summarizer._chunks_within
    depth = 0

    def counting(text: str, budget_tokens: int, split: Any) -> list[tuple[str, int]]:
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
    summarizer._chunks_within = _counting_chunks(summarizer, observed)  # type: ignore[method-assign]
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
        del summarizer._chunks_within
    parts = getattr(result, "parts_total", None) or (observed[0] if observed else 0)
    return result, error, parts, count_contention(start, end)


def run_cases(
    summarizer: Summarizer,
    runs: Sequence[RunSpec],
    texts: Mapping[str, str],
    targets: Mapping[str, Target],
) -> list[CaseRun]:
    """Run each spec in order from a reset model and its seed, capturing every call.

    ``texts`` is keyed by document id and ``targets`` by case id. A case flagged
    ``m12a`` is compared with repetition 0 of the same case without the flag.
    """
    capture = summarizer.llm
    if not isinstance(capture, CapturingLlama):
        capture = CapturingLlama(summarizer.llm, summarizer._perf_snapshot)
        summarizer.llm = capture
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
        case_run.metrics = score_case_run(
            case, result, case_run.calls, text, targets[case.id], parts=parts
        )
        if case.m12a:
            case_run.metrics["M12a"] = _m12a(_first_run_of(case, spec.seed, done), case_run)
        done.append(case_run)
    return done
