#!/usr/bin/env python3
"""
DocSummarizer Command Line Interface
For users who prefer terminal over GUI.
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Literal, NoReturn

from .document_parser import extract_text, find_documents, get_document_info
from .io_helpers import (
    SKIPPED_APPEARED,
    SKIPPED_EXISTS,
    same_file,
    summary_filename,
    write_summary_txt,
)
from .model_manager import (
    DEFAULT_MODEL,
    LANGUAGE_AUTO,
    OUTPUT_LANGUAGES,
    SUMMARY_TYPE_DETAILED,
    SUMMARY_TYPES,
    Summarizer,
    download_model,
    get_model_path,
    is_model_downloaded,
    normalize_language,
)
from .settings import Settings, load_settings


def print_progress(percent: float, message: str) -> None:
    """Print progress to console."""
    bar_length = 40
    filled = int(bar_length * percent / 100)
    bar = "=" * filled + "-" * (bar_length - filled)
    print(f"\r[{bar}] {percent:.1f}% - {message}", end="", flush=True)
    if percent >= 100:
        print()


def ensure_model() -> bool:
    """Ensure the model is downloaded."""
    if is_model_downloaded():
        return True

    print(f"Model not found. Downloading {DEFAULT_MODEL.name} ({DEFAULT_MODEL.size_gb} GB)...")
    print("This is a one-time download.\n")

    _path, error = download_model(progress_callback=print_progress)

    if error:
        print(f"\nError: {error}")
        return False

    print("Model downloaded successfully!\n")
    return True


Outcome = Literal["written", "skipped", "failed"]

# One planned input: the source, its output (None = console), and whether its
# output already exists and is left alone.
PlannedInput = tuple[Path, Path | None, bool]


def summarize_file(
    filepath: str,
    summarizer: Summarizer,
    summary_type: str = SUMMARY_TYPE_DETAILED,
    target: Path | None = None,
    language: str = LANGUAGE_AUTO,
    *,
    replace: bool = True,
) -> Outcome:
    """Summarize a single file into `target`, or to the console without one.

    With `replace=False` an output that appeared since the run checked it is
    kept and the input counts as skipped.
    """
    info = get_document_info(filepath)
    print(f"Processing: {info['name']} ({info['size_mb']} MB)")

    text, error = extract_text(filepath)
    if error:
        print(f"  Error: {error}")
        return "failed"

    print(f"  Extracted {len(text)} characters")

    print("  Generating summary...")
    try:
        summary = summarizer.summarize(text, summary_type=summary_type, language=language)
    except Exception as e:
        print(f"  Error: {e!s}")
        return "failed"

    if target is None:
        print("\n" + "=" * 60)
        print(f"SUMMARY ({summary_type})")
        print("=" * 60 + "\n")
        print(summary)
        print("\n" + "=" * 60)
        return "written"

    try:
        written = write_summary_txt(
            target,
            source_name=info["name"],
            summary=summary,
            summary_type=summary_type,
            separator_width=60,
            replace=replace,
        )
    except OSError as e:  # e.g. a locked target; the rest of the run goes on
        print(f"  Error: could not write {target}: {e}")
        return "failed"
    if not written:
        print(f"{info['name']}: skipped ({SKIPPED_APPEARED})")
        return "skipped"
    print(f"  Saved to: {target}")
    return "written"


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (split out so it can be unit-tested)."""
    parser = argparse.ArgumentParser(
        description="DocSummarizer - Offline Document Summarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf                     Summarize a single file
  %(prog)s document.pdf -t structured       Use structured summary format
  %(prog)s ./documents/ -o ./summaries/     Batch process a folder
  %(prog)s report.docx -o summary.txt       Save to specific file
  %(prog)s document.pdf --gpu               Offload to the GPU for this run
  %(prog)s document.pdf -l English          Force the summary's language
        """,
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="Input file or directory to process",
    )
    parser.add_argument(
        "-t",
        "--type",
        choices=list(SUMMARY_TYPES),
        default=SUMMARY_TYPE_DETAILED,
        help=f"Summary type (default: {SUMMARY_TYPE_DETAILED})",
    )
    parser.add_argument(
        "-o", "--output", help="Output file or directory (default: print to console)"
    )
    # Free-form rather than `choices=`: the model handles far more languages
    # than the list the GUI offers, so any name is passed straight through.
    parser.add_argument(
        "-l",
        "--language",
        default=None,
        metavar="LANG",
        help=f"Language to write the summary in, e.g. {', '.join(OUTPUT_LANGUAGES[1:5])}. "
        f"'{LANGUAGE_AUTO}' matches the document. Overrides the saved setting.",
    )
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Offload the model to the GPU (--no-gpu forces CPU). "
        "Overrides the saved setting for this run.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        metavar="N",
        help="CPU threads for inference (overrides the saved setting).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace summaries that already exist in the output folder "
        "(folder runs; by default they are skipped)",
    )
    parser.add_argument(
        "--download-only", action="store_true", help="Only download the model, do not process files"
    )
    return parser


def _resolve_runtime(args: argparse.Namespace, settings: Settings) -> tuple[int | None, int]:
    """Resolve ``(n_threads, n_gpu_layers)`` — CLI flags override saved settings."""
    use_gpu = settings.use_gpu if args.gpu is None else args.gpu
    n_threads = settings.n_threads if args.threads is None else args.threads
    return n_threads, (-1 if use_gpu else 0)


def _resolve_language(args: argparse.Namespace, settings: Settings) -> str:
    """Resolve the output language — the ``--language`` flag beats the setting."""
    return normalize_language(settings.output_language if args.language is None else args.language)


def _refuse(message: str) -> NoReturn:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(2)


def _names_a_directory(value: str) -> bool:
    """`-o` names a directory: an existing one, or a path ending in a separator."""
    return (
        Path(value).is_dir()
        or value.endswith(os.sep)
        or (os.altsep is not None and value.endswith(os.altsep))
    )


def _make_output_dir(out: Path) -> None:
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _refuse(f"could not create the output folder: {e.strerror or e}")


def _resolve_targets(
    input_path: Path, files: list[Path], output: str | None, summary_type: str
) -> list[Path | None]:
    """One output per input (None = console); exits 2 on an unsafe `-o`."""
    if output is None:
        return [None for _ in files]
    out = Path(output)
    if input_path.is_dir():
        if out.exists() and not out.is_dir():
            _refuse("a folder input needs an output folder")
        if same_file(out, input_path):
            _refuse(
                f"Choose a different output folder (default: {input_path.resolve() / 'summaries'})"
            )
        _make_output_dir(out)
        return [out / summary_filename(f, summary_type, "txt") for f in files]
    to_dir = _names_a_directory(output)
    target = out / summary_filename(input_path, summary_type, "txt") if to_dir else out
    if same_file(target, input_path):
        _refuse("the output would replace the input file; choose another output")
    if to_dir:
        _make_output_dir(out)
    return [target]


def _plan_run(
    args: argparse.Namespace, input_path: Path, files: list[Path]
) -> tuple[list[PlannedInput], bool]:
    """Resolve every output before the model loads; returns the plan and `replace`.

    A folder run leaves an existing output alone unless `--overwrite`; a named
    single input always writes its output.
    """
    targets = _resolve_targets(input_path, files, args.output, args.type)
    folder_run = input_path.is_dir() and args.output is not None
    keep_existing = folder_run and not args.overwrite
    plan = [
        (f, t, keep_existing and t is not None and os.path.lexists(t))
        for f, t in zip(files, targets, strict=True)
    ]
    return plan, not keep_existing


def _run_plan(
    plan: list[PlannedInput],
    summarizer: Summarizer,
    summary_type: str,
    language: str,
    *,
    replace: bool,
) -> Counter[Outcome]:
    counts: Counter[Outcome] = Counter()
    for filepath, target, exists in plan:
        if exists:
            print(f"{filepath.name}: skipped: {SKIPPED_EXISTS}")
            counts["skipped"] += 1
        else:
            counts[
                summarize_file(
                    str(filepath), summarizer, summary_type, target, language, replace=replace
                )
            ] += 1
        print()
    return counts


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not ensure_model():
        sys.exit(1)

    if args.download_only:
        print("Model is ready.")
        sys.exit(0)

    if args.input is None:
        parser.error("the following arguments are required: input")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Path does not exist: {args.input}")
        sys.exit(1)

    files = find_documents(input_path)
    if not files:
        print(f"Error: No supported documents found in: {args.input}")
        sys.exit(1)

    plan, replace = _plan_run(args, input_path, files)

    settings = load_settings()
    n_threads, n_gpu_layers = _resolve_runtime(args, settings)
    language = _resolve_language(args, settings)

    print(f"Loading model ({'GPU' if n_gpu_layers else 'CPU'})...")
    try:
        summarizer = Summarizer(get_model_path(), n_threads=n_threads, n_gpu_layers=n_gpu_layers)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print(f"Model loaded. Processing {len(files)} file(s)...")
    print(
        "Summary language: "
        + ("matching each document" if language == LANGUAGE_AUTO else language)
        + "\n"
    )

    try:
        counts = _run_plan(plan, summarizer, args.type, language, replace=replace)
    finally:
        summarizer.close()

    print(
        f"Done. {counts['written']} written, {counts['skipped']} skipped, "
        f"{counts['failed']} failed."
    )


if __name__ == "__main__":
    main()
