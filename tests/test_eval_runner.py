"""Tests for the evaluation runner under ``scripts/eval``.

Everything here runs without a model: the case matrix, its budgets and the
package's configuration are checked as data.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
from eval import cases

from docsummarizer import model_manager
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
