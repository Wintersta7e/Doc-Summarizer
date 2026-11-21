# SPDX-License-Identifier: MIT
"""Tests for the CLI argument parser and runtime resolution.

These cover the new ``--gpu`` / ``--threads`` flags and the rule that explicit
flags override saved settings. They don't invoke ``main()`` (which loads the
model and touches the filesystem); the testable logic is factored into
``_build_parser`` and ``_resolve_runtime``.
"""

from __future__ import annotations

import argparse

from docsummarizer.cli import CliArgs, _build_parser, _resolve_language, _resolve_runtime
from docsummarizer.settings import Settings


def test_cli_args_defaults_match_the_parser() -> None:
    """The typed namespace's class defaults are the parser's own, flag for flag."""
    parsed = vars(_build_parser().parse_args([], namespace=argparse.Namespace()))
    declared = {name: getattr(CliArgs, name) for name in CliArgs.__annotations__}
    assert declared == parsed


def test_parser_defaults() -> None:
    args = _build_parser().parse_args(["doc.pdf"], namespace=CliArgs())
    assert args.input == "doc.pdf"
    assert args.type == "detailed"
    assert args.gpu is None
    assert args.threads is None


def test_parser_gpu_flags() -> None:
    assert _build_parser().parse_args(["d", "--gpu"], namespace=CliArgs()).gpu is True
    assert _build_parser().parse_args(["d", "--no-gpu"], namespace=CliArgs()).gpu is False


def test_parser_threads_and_type() -> None:
    args = _build_parser().parse_args(["d", "--threads", "6", "-t", "brief"], namespace=CliArgs())
    assert args.threads == 6
    assert args.type == "brief"


def test_resolve_runtime_uses_settings_without_flags() -> None:
    args = _build_parser().parse_args(["d"], namespace=CliArgs())
    assert _resolve_runtime(args, Settings(n_threads=4, use_gpu=False)) == (4, 0)


def test_resolve_runtime_flags_override_settings() -> None:
    args = _build_parser().parse_args(["d", "--gpu", "--threads", "8"], namespace=CliArgs())
    assert _resolve_runtime(args, Settings(n_threads=4, use_gpu=False)) == (8, -1)


def test_resolve_runtime_no_gpu_forces_cpu() -> None:
    args = _build_parser().parse_args(["d", "--no-gpu"], namespace=CliArgs())
    assert _resolve_runtime(args, Settings(use_gpu=True)) == (None, 0)


def test_parser_language_defaults_to_none() -> None:
    """No flag means 'defer to the saved setting', not 'auto'."""
    assert _build_parser().parse_args(["doc.pdf"], namespace=CliArgs()).language is None


def test_parser_language_flag() -> None:
    assert (
        _build_parser().parse_args(["d", "-l", "Chinese"], namespace=CliArgs()).language
        == "Chinese"
    )
    assert (
        _build_parser().parse_args(["d", "--language", "German"], namespace=CliArgs()).language
        == "German"
    )


def test_resolve_language_uses_settings_without_flag() -> None:
    args = _build_parser().parse_args(["d"], namespace=CliArgs())
    assert _resolve_language(args, Settings(output_language="Chinese")) == "Chinese"


def test_resolve_language_flag_overrides_settings() -> None:
    args = _build_parser().parse_args(["d", "-l", "French"], namespace=CliArgs())
    assert _resolve_language(args, Settings(output_language="Chinese")) == "French"


def test_resolve_language_normalizes() -> None:
    args = _build_parser().parse_args(["d", "-l", "  auto "], namespace=CliArgs())
    assert _resolve_language(args, Settings(output_language="Chinese")) == "auto"
