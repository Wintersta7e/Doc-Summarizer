#!/usr/bin/env python3
"""
DocSummarizer Command Line Interface
For users who prefer terminal over GUI.
"""

import argparse
import sys
from pathlib import Path

from .document_parser import extract_text, find_documents, get_document_info
from .io_helpers import write_summary_txt
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


def summarize_file(
    filepath: str,
    summarizer: Summarizer,
    summary_type: str = SUMMARY_TYPE_DETAILED,
    output_path: str | None = None,
    language: str = LANGUAGE_AUTO,
) -> bool:
    """Summarize a single file."""
    info = get_document_info(filepath)
    print(f"Processing: {info['name']} ({info['size_mb']} MB)")

    text, error = extract_text(filepath)
    if error:
        print(f"  Error: {error}")
        return False

    print(f"  Extracted {len(text)} characters")

    print("  Generating summary...")
    try:
        summary = summarizer.summarize(text, summary_type=summary_type, language=language)
    except Exception as e:
        print(f"  Error: {e!s}")
        return False

    if output_path:
        out_file = Path(output_path)
        if out_file.is_dir():
            out_file = out_file / f"{Path(filepath).stem}_summary.txt"

        write_summary_txt(
            out_file,
            source_name=info["name"],
            summary=summary,
            summary_type=summary_type,
            separator_width=60,
        )

        print(f"  Saved to: {out_file}")
    else:
        print("\n" + "=" * 60)
        print(f"SUMMARY ({summary_type})")
        print("=" * 60 + "\n")
        print(summary)
        print("\n" + "=" * 60)

    return True


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

    success_count = 0
    try:
        for filepath in files:
            if summarize_file(str(filepath), summarizer, args.type, args.output, language):
                success_count += 1
            print()
    finally:
        summarizer.close()

    print(f"Done. Successfully processed {success_count}/{len(files)} file(s).")


if __name__ == "__main__":
    main()
