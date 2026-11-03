"""Scoring for evaluation runs: pure functions over stored text, no model and no I/O.

M1 output presence, M2 language compliance (per call and on the result), M4
truncation and M12a reproducibility. Every metric keeps what it counted, so a
report can print numerator, denominator and each failing unit.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

METRIC_VERSION: int = 1

# Each output language a run can name, mapped to the classifier's code. The
# names are the GUI picker's, plus Traditional Chinese.
LANGUAGE_CODES: dict[str, str] = {
    "English": "en",
    "Chinese": "zh",
    "Traditional Chinese": "zh",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Portuguese": "pt",
    "Italian": "it",
    "Dutch": "nl",
    "Polish": "pl",
    "Russian": "ru",
    "Ukrainian": "uk",
    "Turkish": "tr",
    "Arabic": "ar",
    "Hindi": "hi",
    "Japanese": "ja",
    "Korean": "ko",
    "Vietnamese": "vi",
}

# The Chinese script an explicit language name requires.
TARGET_SCRIPT: dict[str, str] = {"Chinese": "Simplified", "Traditional Chinese": "Traditional"}

# (Simplified, Traditional) pairs of common characters that differ between scripts.
SCRIPT_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (pair[0], pair[1])
    for pair in [
        "这這",
        "们們",
        "说說",
        "时時",
        "为為",
        "会會",
        "个個",
        "来來",
        "对對",
        "学學",
        "国國",
        "过過",
        "还還",
        "没沒",
        "从從",
        "现現",
        "实實",
        "经經",
        "动動",
        "进進",
        "种種",
        "样樣",
        "长長",
        "开開",
        "关關",
        "问問",
        "题題",
        "见見",
        "间間",
        "让讓",
        "认認",
        "识識",
        "话話",
        "语語",
        "书書",
        "东東",
        "车車",
        "门門",
        "马馬",
        "鸟鳥",
        "鱼魚",
        "风風",
        "飞飛",
        "爱愛",
        "头頭",
        "买買",
        "卖賣",
        "写寫",
        "读讀",
        "页頁",
        "电電",
        "报報",
        "将將",
        "应應",
        "该該",
        "当當",
        "产產",
        "业業",
        "华華",
        "处處",
        "级級",
        "结結",
        "论論",
        "统統",
        "计計",
        "设設",
    ]
)
_SIMPLIFIED = frozenset(s for s, _ in SCRIPT_PAIRS)
_TRADITIONAL = frozenset(t for _, t in SCRIPT_PAIRS)

_AUTO = "auto"  # the runner's name for "match the document"

# Sentence ends: the CJK full stop, exclamation and question marks end a unit at
# once; their ASCII forms only before whitespace.
_UNIT_END = re.compile("(?<=[。！？])|(?<=[.!?])\\s+")  # noqa: RUF001 - CJK punctuation
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_MIN_LETTERS = 25
_MIN_CJK = 10


@dataclass(frozen=True)
class Target:
    """What a run's output must be written in."""

    code: str
    script: str | None  # for zh: "Simplified" or "Traditional"; None = not gated


def target_for(language: str, label: str, script: str | None) -> Target:
    """The target for a run: the document's label for ``auto``, else the named language."""
    if language == _AUTO:
        if label not in LANGUAGE_CODES.values():
            raise ValueError(f"no classifier code for the label {label!r}")
        return Target(label, script if label == "zh" else None)
    code = LANGUAGE_CODES.get(language)
    if code is None:
        raise ValueError(f"no classifier code for the language {language!r}")
    return Target(code, TARGET_SCRIPT.get(language))


def _is_han(c: str) -> bool:
    o = ord(c)
    return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF or o >= 0x20000


def _is_kana(c: str) -> bool:
    o = ord(c)
    return 0x3040 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF or 0xFF66 <= o <= 0xFF9F


def _is_hangul(c: str) -> bool:
    o = ord(c)
    return 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F


def _is_arabic(c: str) -> bool:
    o = ord(c)
    return (
        0x0600 <= o <= 0x06FF
        or 0x0750 <= o <= 0x077F
        or 0x08A0 <= o <= 0x08FF
        or 0xFB50 <= o <= 0xFDFF
        or 0xFE70 <= o <= 0xFEFF
    )


def _is_devanagari(c: str) -> bool:
    return 0x0900 <= ord(c) <= 0x097F


def _is_cjk(c: str) -> bool:
    return _is_han(c) or _is_kana(c) or _is_hangul(c)


def split_units(text: str) -> list[str]:
    """Split output into scoring units: sentences and lines, bullets stripped."""
    units: list[str] = []
    for line in text.splitlines():
        body = _BULLET.sub("", line)
        units.extend(unit.strip() for unit in _UNIT_END.split(body) if unit.strip())
    return units


def is_scored(unit: str) -> bool:
    """A unit is scored with >= 25 letters or >= 10 Han / kana / Hangul characters."""
    cjk = sum(1 for c in unit if _is_cjk(c))
    letters = sum(1 for c in unit if c.isalpha() and not _is_cjk(c))
    return letters >= _MIN_LETTERS or cjk >= _MIN_CJK


@functools.cache
def _identifier() -> Any:
    """The language identifier, restricted to the output languages' codes."""
    from py3langid.langid import MODEL_FILE, LanguageIdentifier

    identifier = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
    identifier.set_languages(sorted(set(LANGUAGE_CODES.values())))
    return identifier


def classify_unit(unit: str) -> str:
    """The unit's language code: script shares first, then the statistical identifier.

    Shares are taken over alphabetic characters.
    """
    alpha = [c for c in unit if c.isalpha()]
    if alpha:
        total = len(alpha)
        kana = sum(1 for c in alpha if _is_kana(c)) / total
        hangul = sum(1 for c in alpha if _is_hangul(c)) / total
        han = sum(1 for c in alpha if _is_han(c)) / total
        if kana >= 0.05:
            return "ja"
        if hangul >= 0.30:
            return "ko"
        if han + kana >= 0.30:
            return "zh"
        if sum(1 for c in alpha if _is_arabic(c)) / total >= 0.50:
            return "ar"
        if sum(1 for c in alpha if _is_devanagari(c)) / total >= 0.50:
            return "hi"
    code, _ = _identifier().classify(unit)
    return str(code)


def unit_script(unit: str) -> str | None:
    """The majority script of the unit's listed characters; ``None`` if none or tied."""
    simplified = sum(1 for c in unit if c in _SIMPLIFIED)
    traditional = sum(1 for c in unit if c in _TRADITIONAL)
    if simplified > traditional:
        return "Simplified"
    if traditional > simplified:
        return "Traditional"
    return None


@dataclass(frozen=True)
class M1:
    """Output presence on one raw completion."""

    empty: bool
    think_tags: int
    passed: bool


def m1(content: str) -> M1:
    """Fails on empty content or any ``<think>`` / ``</think>`` tag."""
    empty = not content.strip()
    tags = content.count("<think>") + content.count("</think>")
    return M1(empty, tags, not empty and tags == 0)


def m4(finish_reason: str | None) -> bool:
    """Whether a completion was truncated by its token limit."""
    return finish_reason == "length"


@dataclass(frozen=True)
class OffTarget:
    """One scored unit in the wrong language or script."""

    call: int | None  # index of the call; None for the returned result
    unit: str
    got: str


@dataclass(frozen=True)
class M2:
    """Language compliance over every call of a run and its returned result."""

    scored: int
    off_target: list[OffTarget]
    unparsed_calls: int
    passed: bool


def _off_target(unit: str, target: Target) -> str | None:
    code = classify_unit(unit)
    if code != target.code:
        return code
    if code == "zh" and target.script is not None:
        script = unit_script(unit)
        if script is not None and script != target.script:
            return f"zh/{script}"
    return None


def m2(calls: Sequence[Sequence[str] | None], result_units: Sequence[str], target: Target) -> M2:
    """Score every scored unit of every call and of the result against ``target``.

    A ``None`` call is a JSON call that did not parse: it has no units and is
    counted. A run complies iff no scored unit is off-target.
    """
    scored = 0
    unparsed = 0
    off: list[OffTarget] = []
    sources: list[tuple[int | None, Sequence[str] | None]] = [*enumerate(calls)]
    sources.append((None, result_units))
    for index, units in sources:
        if units is None:
            unparsed += 1
            continue
        for unit in units:
            if not is_scored(unit):
                continue
            scored += 1
            got = _off_target(unit, target)
            if got is not None:
                off.append(OffTarget(index, unit, got))
    return M2(scored, off, unparsed, not off)


@dataclass(frozen=True)
class M12a:
    """Whether a seed-0 re-run reproduced the first run's returned text."""

    identical: bool
    first_diff: int | None


def m12a(first: str, rerun: str) -> M12a:
    """Byte-identical check; on a difference, the index of the first differing char."""
    if first == rerun:
        return M12a(True, None)
    for index, (a, b) in enumerate(zip(first, rerun, strict=False)):
        if a != b:
            return M12a(False, index)
    return M12a(False, min(len(first), len(rerun)))
