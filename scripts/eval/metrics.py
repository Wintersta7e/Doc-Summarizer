# SPDX-License-Identifier: MIT
"""Scoring for evaluation runs: pure functions over stored text, no model and no I/O.

M1 output presence, M2 language compliance (per call and on the result), M4
truncation, M7 citation hit rate and M12a reproducibility, plus the verdict
between two runs. Every metric keeps what it counted, so a report can print
numerator, denominator and each failing unit.
"""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

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


class _SpanLike(Protocol):
    @property
    def start(self) -> int: ...

    @property
    def end(self) -> int: ...


class _PointLike(Protocol):
    @property
    def text(self) -> str: ...

    @property
    def citation(self) -> _SpanLike | None: ...


@dataclass(frozen=True)
class PointRecord:
    """One returned point: its citation's offsets as grounding left them, and its quote."""

    citation: tuple[int, int] | None
    quote: str


def point_records(points: Sequence[_PointLike], quotes: Mapping[str, str]) -> list[PointRecord]:
    """Records for the returned points; ``quotes`` maps a point's text to the model's quote."""
    return [
        PointRecord(
            None if point.citation is None else (point.citation.start, point.citation.end),
            quotes.get(point.text, ""),
        )
        for point in points
    ]


@dataclass(frozen=True)
class M7:
    """Citation hit rate over a run's points, plus what is reported beside it."""

    points: int
    cited: int  # a citation as the production grounding returned it
    tight: int  # cited, with a span no longer than the span cap
    whole_document: int  # cited spans covering the whole document
    quotes: int  # non-empty quotes
    verbatim: int  # non-empty quotes that are exact substrings of the document

    @property
    def hit_rate(self) -> float:
        return self.tight / self.points if self.points else 0.0


def m7(points: Sequence[PointRecord], document: str) -> M7:
    """Score the citations the production grounding returned; never re-locate a quote."""
    from docsummarizer import provenance

    cited = [point.citation for point in points if point.citation is not None]
    whole = document.strip()
    quotes = [point.quote for point in points if point.quote]
    return M7(
        points=len(points),
        cited=len(cited),
        tight=sum(1 for start, end in cited if end - start <= provenance.MAX_SPAN_CHARS),
        whole_document=sum(1 for start, end in cited if document[start:end].strip() == whole),
        quotes=len(quotes),
        verbatim=sum(1 for quote in quotes if quote in document),
    )


# The noise band's floors: on the corpus aggregate and per cell.
AGG_FLOOR = 0.05
CELL_FLOOR = 0.10
# Below this many units in either arm, a cell's proportions are not compared.
MIN_CELL_N = 20

HARD_METRICS = ("M1", "M2", "M4", "M12a")
PROPORTION_METRICS = ("M7",)


def noise_delta(p_b: float, n_b: int, p_c: float, n_c: int, floor: float) -> float:
    """The band within which two proportions differ only by sampling noise."""
    return max(floor, 2 * math.sqrt(p_b * (1 - p_b) / n_b + p_c * (1 - p_c) / n_c))


@dataclass(frozen=True)
class CaseScore:
    """One case run's scores, as a verdict reads them."""

    case: str
    seed: int
    cell: tuple[str, str, str]  # (target, path, length)
    hard: dict[str, int]  # metric -> violations; 0 passes
    proportion: dict[str, tuple[int, int]]  # metric -> (numerator, denominator)


@dataclass(frozen=True)
class Flip:
    """A hard metric that changed between the arms.

    ``seed`` is ``None`` for an improvement, which must hold at every seed.
    """

    case: str
    seed: int | None
    metric: str


@dataclass(frozen=True)
class VerdictRow:
    """One line of a verdict: a metric over a cell, the aggregate or one case run.

    ``status`` is ``ok``, ``regression``, ``improvement`` or ``insufficient n``,
    or ``n/a`` for a hard metric no paired run in the cell measured; a case
    run present in one arm only reads ``only in B`` or ``only in C``.
    """

    scope: str
    metric: str
    p_b: float | None
    n_b: int
    p_c: float | None
    n_c: int
    delta: float | None
    status: str


@dataclass(frozen=True)
class Verdict:
    """Candidate C against baseline B: every hard flip and every proportion row."""

    regressions: list[Flip]
    improvements: list[Flip]
    rows: list[VerdictRow]

    @property
    def regressed(self) -> bool:
        return bool(self.regressions) or any(row.status == "regression" for row in self.rows)


def _scope(cell: tuple[str, str, str]) -> str:
    return "/".join(cell)


def _hard_flips(
    pairs: Sequence[tuple[CaseScore, CaseScore]],
) -> tuple[list[Flip], list[Flip]]:
    regressions: list[Flip] = []
    by_case: dict[tuple[str, str], list[tuple[bool, bool]]] = {}
    for b, c in pairs:
        for metric in HARD_METRICS:
            if metric not in b.hard or metric not in c.hard:
                continue
            b_pass, c_pass = b.hard[metric] == 0, c.hard[metric] == 0
            if b_pass and not c_pass:
                regressions.append(Flip(b.case, b.seed, metric))
            by_case.setdefault((b.case, metric), []).append((b_pass, c_pass))
    improvements = [
        Flip(case, None, metric)
        for (case, metric), seeds in by_case.items()
        if not all(b_pass for b_pass, _ in seeds) and all(c_pass for _, c_pass in seeds)
    ]
    return regressions, improvements


def _proportion_row(
    scope: str, metric: str, b: tuple[int, int], c: tuple[int, int], floor: float
) -> VerdictRow:
    (num_b, n_b), (num_c, n_c) = b, c
    p_b = num_b / n_b if n_b else None
    p_c = num_c / n_c if n_c else None
    if p_b is None or p_c is None or n_b < MIN_CELL_N or n_c < MIN_CELL_N:
        return VerdictRow(scope, metric, p_b, n_b, p_c, n_c, None, "insufficient n")
    delta = noise_delta(p_b, n_b, p_c, n_c, floor)
    status = "ok"
    if p_c - p_b < -delta:
        status = "regression"
    elif p_c - p_b > delta:
        status = "improvement"
    return VerdictRow(scope, metric, p_b, n_b, p_c, n_c, delta, status)


def _summed(scores: Sequence[CaseScore], metric: str) -> tuple[int, int]:
    values = [score.proportion[metric] for score in scores if metric in score.proportion]
    return sum(num for num, _ in values), sum(den for _, den in values)


def verdict(b: Sequence[CaseScore], c: Sequence[CaseScore]) -> Verdict:
    """Compare two arms case run by case run, with hard flips and noise-banded proportions.

    Runs pair by (case, seed). A hard metric regresses on any pass-to-fail flip
    and improves only when a case failing in B passes in C at every seed.
    Proportions are summed per cell and over the aggregate of the paired runs.
    """
    b_runs = {(score.case, score.seed): score for score in b}
    c_runs = {(score.case, score.seed): score for score in c}
    pairs = [(b_runs[key], c_runs[key]) for key in b_runs if key in c_runs]
    rows = [
        VerdictRow(f"{case} seed {seed}", "case run", None, 0, None, 0, None, status)
        for runs, other, status in ((b_runs, c_runs, "only in B"), (c_runs, b_runs, "only in C"))
        for case, seed in runs
        if (case, seed) not in other
    ]
    regressions, improvements = _hard_flips(pairs)

    cells: dict[tuple[str, str, str], list[tuple[CaseScore, CaseScore]]] = {}
    for pair in pairs:
        cells.setdefault(pair[0].cell, []).append(pair)
    for metric in HARD_METRICS:
        for cell, cell_pairs in cells.items():
            measured = [(b, c) for b, c in cell_pairs if metric in b.hard and metric in c.hard]
            n = len(measured)
            if n == 0:
                rows.append(VerdictRow(_scope(cell), metric, None, 0, None, 0, None, "n/a"))
                continue
            in_cell = {score.case for score, _ in measured}
            status = "ok"
            if any(flip.metric == metric and flip.case in in_cell for flip in regressions):
                status = "regression"
            elif any(flip.metric == metric and flip.case in in_cell for flip in improvements):
                status = "improvement"
            passed_b = sum(1 for score, _ in measured if score.hard[metric] == 0)
            passed_c = sum(1 for _, score in measured if score.hard[metric] == 0)
            rows.append(
                VerdictRow(_scope(cell), metric, passed_b / n, n, passed_c / n, n, None, status)
            )
    for metric in PROPORTION_METRICS:
        for cell, cell_pairs in cells.items():
            sums_b = _summed([score for score, _ in cell_pairs], metric)
            sums_c = _summed([score for _, score in cell_pairs], metric)
            rows.append(_proportion_row(_scope(cell), metric, sums_b, sums_c, CELL_FLOOR))
        sums_b = _summed([score for score, _ in pairs], metric)
        sums_c = _summed([score for _, score in pairs], metric)
        rows.append(_proportion_row("aggregate", metric, sums_b, sums_c, AGG_FLOOR))
    return Verdict(regressions, improvements, rows)
