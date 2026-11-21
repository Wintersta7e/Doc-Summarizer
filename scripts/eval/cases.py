# SPDX-License-Identifier: MIT
"""The evaluation corpus by shape, and the cases each tier runs.

Documents are named here only by an anonymous id and their shape. The texts,
their sources, licences, labels and measured token counts live in
``<eval-dir>/corpus/manifest.json``, never in the repo.
"""

from __future__ import annotations

from dataclasses import dataclass

from docsummarizer.model_manager import (
    LANGUAGE_AUTO,
    SUMMARY_TYPE_BRIEF,
    SUMMARY_TYPE_DETAILED,
    SUMMARY_TYPES,
)


@dataclass(frozen=True)
class Document:
    """One corpus document's shape: language label, target size and parts."""

    id: str
    label: str
    script: str | None  # set only where the id fixes it; otherwise the manifest's
    tokens: int  # target size in the model's tokens
    parts: int  # parts at the incumbent's context
    kind: str


@dataclass(frozen=True)
class Case:
    """One summary request the runner issues: a document, a type and a language."""

    doc: str
    summary_type: str
    language: str
    split: bool = False  # run at SPLIT_N_CTX, so the document takes >= 2 parts
    m12a: bool = False  # a re-run that must reproduce the first run exactly

    @property
    def id(self) -> str:
        """The case's report id: ``doc/type/language``, plus its run-mode suffixes."""
        suffix = ("@split" if self.split else "") + ("#m12a" if self.m12a else "")
        return f"{self.doc}/{self.summary_type}/{self.language}{suffix}"


_CORPUS = (
    Document("zh-lit-s", "zh", None, 2300, 1, "narrative"),
    Document("zh-lit-m", "zh", None, 9000, 2, "narrative"),
    Document("zh-lit-l", "zh", None, 45000, 8, "narrative"),
    Document("zh-off-s", "zh", None, 2500, 1, "official"),
    Document("zh-mix-s", "zh", None, 2000, 1, "transcript"),
    Document("de-lit-s", "de", None, 2500, 1, "narrative"),
    Document("de-off-m", "de", None, 9000, 2, "official"),
    Document("de-lit-l", "de", None, 36000, 6, "narrative"),
    Document("en-lit-s", "en", None, 2500, 1, "narrative"),
    Document("en-tech-m", "en", None, 9000, 2, "technical"),
    Document("en-tech-l", "en", None, 40000, 6, "technical"),
)
# The language set: one part at the incumbent's context, two at SPLIT_N_CTX.
_LANGUAGE_SET = (
    Document("de-cred-s", "de", None, 2800, 1, "credits"),
    Document("zht-lit-s", "zh", "Traditional", 2800, 1, "narrative"),
    Document("fr-lit-s", "fr", None, 2800, 1, "narrative"),
    Document("ru-lit-s", "ru", None, 2800, 1, "narrative"),
    Document("ja-lit-s", "ja", None, 2800, 1, "narrative"),
    Document("es-lit-s", "es", None, 2800, 1, "narrative"),
)

DOCUMENTS: dict[str, Document] = {doc.id: doc for doc in (*_CORPUS, *_LANGUAGE_SET)}
CORPUS_IDS: tuple[str, ...] = tuple(doc.id for doc in _CORPUS)
LANGUAGE_SET_IDS: tuple[str, ...] = tuple(doc.id for doc in _LANGUAGE_SET)

# Explicit-language runs, each over a single-part document (Detailed, seed 0).
EXPLICIT_CASES: tuple[tuple[str, str], ...] = (
    ("de-lit-s", "Spanish"),
    ("zh-lit-s", "German"),
    ("en-lit-s", "Chinese"),
    ("en-lit-s", "Traditional Chinese"),
    ("zht-lit-s", "German"),
)

# The lang tier's second pass. It stays fixed so split runs compare across the phase.
SPLIT_N_CTX: int = 4096

TIER_TYPES: dict[str, tuple[str, ...]] = {
    "smoke": (SUMMARY_TYPE_BRIEF, SUMMARY_TYPE_DETAILED),
    "lang": SUMMARY_TYPES,
    "full": SUMMARY_TYPES,
}

_SMOKE_AUTO_IDS = ("zh-lit-s", "zh-mix-s", "de-lit-s", "en-lit-s", "zh-lit-m")
_SMOKE_M12A = (
    ("zh-lit-s", SUMMARY_TYPE_BRIEF),
    ("zh-lit-s", SUMMARY_TYPE_DETAILED),
    ("zh-lit-m", SUMMARY_TYPE_DETAILED),
)


def _auto(doc_ids: tuple[str, ...], types: tuple[str, ...], *, split: bool = False) -> list[Case]:

    return [Case(doc, t, LANGUAGE_AUTO, split=split) for doc in doc_ids for t in types]


def _explicit(doc_ids: tuple[str, ...]) -> list[Case]:
    return [
        Case(doc, SUMMARY_TYPE_DETAILED, language)
        for doc, language in EXPLICIT_CASES
        if doc in doc_ids
    ]


def _smoke() -> tuple[Case, ...]:
    m12a = [Case(doc, t, LANGUAGE_AUTO, m12a=True) for doc, t in _SMOKE_M12A]
    # The re-runs go last, after every other case has run.
    return (*_auto(_SMOKE_AUTO_IDS, TIER_TYPES["smoke"]), *_explicit(CORPUS_IDS), *m12a)


def _lang() -> tuple[Case, ...]:
    types = TIER_TYPES["lang"]
    return (
        *_auto(LANGUAGE_SET_IDS, types),
        *_explicit(LANGUAGE_SET_IDS),
        *_auto(LANGUAGE_SET_IDS, types, split=True),
    )


def _full(smoke: tuple[Case, ...], lang: tuple[Case, ...]) -> tuple[Case, ...]:
    cases = [*smoke, *lang]
    seen = {case.id for case in cases}
    cases.extend(case for case in _auto(CORPUS_IDS, TIER_TYPES["full"]) if case.id not in seen)
    return tuple(cases)


_SMOKE = _smoke()
_LANG = _lang()

TIERS: dict[str, tuple[Case, ...]] = {
    "smoke": _SMOKE,
    "lang": _LANG,
    "full": _full(_SMOKE, _LANG),
}
