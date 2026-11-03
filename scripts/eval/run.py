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
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

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
