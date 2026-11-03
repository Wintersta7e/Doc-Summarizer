"""
I/O helpers shared between the GUI and CLI.

Both entry points need to write summaries to text or docx files with the
same header format. Keeping that here means the format only has to be
maintained in one place. Every write renders into a temp file beside its
target and goes through `publish`, so no file is ever half-written or
replaced unless the caller asked for that.
"""

import os
import secrets
import sys
from collections.abc import Callable
from pathlib import Path

# Why a folder run left an input's output alone (shared by the CLI and GUI).
SKIPPED_EXISTS = "summary exists"
SKIPPED_APPEARED = "appeared during run"


def publish(tmp: Path, target: Path, *, replace: bool) -> bool:
    """Move the finished temp file `tmp` to `target`; `tmp` never survives.

    `replace=True` swaps it in atomically, replacing any existing target.
    `replace=False` creates `target` only if it is absent and returns
    False, leaving the existing file untouched, when it already exists.
    Any other OSError (e.g. a filesystem without hard links) propagates:
    there is deliberately no fallback that writes the target directly.
    """
    try:
        if replace:
            tmp.replace(target)
            return True
        try:
            if sys.platform == "win32":
                # Windows rename refuses an existing destination.
                tmp.rename(target)
            else:
                # POSIX rename replaces silently; a hard link refuses.
                os.link(tmp, target)
        except FileExistsError:
            return False
        return True
    finally:
        tmp.unlink(missing_ok=True)


def _write_and_publish(target: Path, write: Callable[[Path], None], *, replace: bool) -> bool:
    """Render via `write` into a fresh temp beside `target`, then publish it."""
    tmp = target.with_name(f"{target.name}.{secrets.token_hex(8)}.partial")
    # Exclusive create with 0o666 so the umask decides the mode, as a plain
    # open() would (mkstemp would make every summary owner-only).
    os.close(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
    try:
        write(tmp)
        return publish(tmp, target, replace=replace)
    finally:
        tmp.unlink(missing_ok=True)


def same_file(a: str | os.PathLike[str], b: str | os.PathLike[str]) -> bool:
    """True when `a` and `b` name the same file, however each is spelled."""
    if os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b)):
        return True
    return Path(a).exists() and Path(b).exists() and Path(a).samefile(b)


def summary_filename(source: Path, summary_type: str, fmt: str) -> str:
    """The output name for `source`: `<source name>.<type>.<fmt>`."""
    return f"{source.name}.{summary_type}.{fmt}"


def write_summary_txt(
    output_path: str | Path,
    *,
    source_name: str,
    summary: str,
    summary_type: str | None = None,
    separator_width: int = 50,
    replace: bool,
) -> bool:
    """Write `summary` to a UTF-8 text file with a standard header.

    The header is:

        Summary of: <source_name>
        Type: <summary_type>          # omitted when summary_type is None
        ==================================================

        <summary body>

    Returns False when `replace` is False and the file already exists.
    """

    def write(tmp: Path) -> None:
        with tmp.open("w", encoding="utf-8") as f:
            f.write(f"Summary of: {source_name}\n")
            if summary_type:
                f.write(f"Type: {summary_type}\n")
            f.write("=" * separator_width + "\n\n")
            f.write(summary)

    return _write_and_publish(Path(output_path), write, replace=replace)


def write_summary_docx(
    output_path: str | Path,
    *,
    source_name: str,
    summary: str,
    replace: bool,
) -> bool:
    """Write `summary` to a `.docx` file with `source_name` as the heading.

    Returns False when `replace` is False and the file already exists.
    """
    # Imported lazily; python-docx is only needed when the user opts into
    # docx output, and the CLI shouldn't pay for the import.
    from docx import Document

    def write(tmp: Path) -> None:
        doc = Document()
        doc.add_heading(f"Summary: {source_name}", 0)
        doc.add_paragraph(summary)
        doc.save(str(tmp))

    return _write_and_publish(Path(output_path), write, replace=replace)
