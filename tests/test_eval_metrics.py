"""Tests for ``scripts/eval/metrics.py``: units, the language classifier and scoring.

Fixtures are short project-written texts about a village bakery and a river
ferry, one per language the classifier must tell apart.
"""

# The CJK fixtures use full-width punctuation on purpose.
# ruff: noqa: RUF001

from __future__ import annotations

import random
import re
from pathlib import Path

from eval import cases, metrics

from docsummarizer.model_manager import LANGUAGE_AUTO, OUTPUT_LANGUAGES

_REPO = Path(__file__).resolve().parents[1]

ZH = (
    "村里的面包店每天清晨五点开门。店主会先把炉子烧热，然后开始揉面。"
    "孩子们放学后常常来买刚出炉的面包。到了周末，店门口总是排起长队。"
    "大家都说这里的面包是全村最香的。"
)
ZH_TRADITIONAL = (
    "河邊的渡船每天早上六點開出。船夫會先檢查繩子，然後讓乘客上船。"
    "學生們常常坐這艘船去對岸的學校。到了冬天，河面上的風很冷。"
    "大家都說這條航線是全村最可靠的。"
)
DE = (
    "Die Bäckerei im Dorf öffnet jeden Morgen um fünf Uhr. "
    "Der Bäcker heizt zuerst den Ofen und knetet dann den Teig. "
    "Nach der Schule kommen die Kinder oft vorbei und kaufen frisches Brot. "
    "Am Wochenende steht vor der Tür immer eine lange Schlange. "
    "Alle sagen, dass es hier das beste Brot im ganzen Tal gibt."
)
EN = (
    "The village bakery opens every morning at five o'clock. "
    "The baker first heats the oven and then kneads the dough. "
    "After school the children often stop by to buy fresh bread. "
    "On weekends there is always a long queue outside the door. "
    "Everyone says the bread here is the best in the whole valley."
)
ES = (
    "La panadería del pueblo abre todas las mañanas a las cinco. "
    "El panadero primero calienta el horno y luego amasa la masa. "
    "Después de la escuela, los niños suelen pasar a comprar pan recién hecho. "
    "Los fines de semana siempre hay una larga fila frente a la puerta. "
    "Todos dicen que aquí se hace el mejor pan de todo el valle."
)
FR = (
    "La boulangerie du village ouvre chaque matin à cinq heures. "
    "Le boulanger chauffe d'abord le four, puis il pétrit la pâte. "
    "Après l'école, les enfants passent souvent acheter du pain frais. "
    "Le week-end, il y a toujours une longue file devant la porte. "
    "Tout le monde dit que c'est ici que l'on trouve le meilleur pain de la vallée."
)
JA = (
    "村のパン屋は毎朝五時に店を開けます。店主はまずオーブンを温めてから、生地をこねます。"
    "学校の帰りに、子どもたちはよく焼きたてのパンを買いに来ます。"
    "週末になると、店の前にはいつも長い列ができます。"
    "みんなここのパンが一番おいしいと言います。"
)
FIXTURES = {"zh": ZH, "de": DE, "en": EN, "es": ES, "fr": FR, "ja": JA}


def test_split_units_and_scored_rule() -> None:
    assert metrics.split_units("甲乙。丙丁！戊己？") == ["甲乙。", "丙丁！", "戊己？"]
    assert metrics.split_units("One. Two! Three? Four") == ["One.", "Two!", "Three?", "Four"]
    assert metrics.split_units("It costs 3.50 today") == ["It costs 3.50 today"]
    assert metrics.split_units("first line\nsecond line") == ["first line", "second line"]
    bullets = "- dash item\n* star item\n• dot item\n1. numbered item"
    assert metrics.split_units(bullets) == ["dash item", "star item", "dot item", "numbered item"]

    assert not metrics.is_scored("a" * 24)
    assert metrics.is_scored("a" * 25)
    assert not metrics.is_scored(" ".join("abcdefgh") * 3)  # 24 letters, spaced
    assert not metrics.is_scored("中" * 9)
    assert metrics.is_scored("中" * 10)


def test_classify_unit_script_rules() -> None:
    assert metrics.classify_unit("日" * 19 + "の") == "ja"  # kana at 5 %
    assert metrics.classify_unit("日" * 39 + "の") == "zh"  # kana at 2.5 %, Han + kana >= 30 %
    assert metrics.classify_unit("한" * 30 + "x" * 70) == "ko"
    assert metrics.classify_unit("中" * 30 + "x" * 70) == "zh"
    assert metrics.classify_unit("ب" * 50 + "x" * 50) == "ar"
    assert metrics.classify_unit("क" * 50 + "x" * 50) == "hi"
    for code, text in FIXTURES.items():
        got = metrics.classify_unit(text)
        print(f"fixture {code}: {got}")
        assert got == code


def test_classify_unit_is_restricted_to_output_languages() -> None:
    rng = random.Random(0)  # noqa: S311 - seeded test data
    codes = set(metrics.LANGUAGE_CODES.values())
    letters = "abcdefghijklmnopqrstuvwxyz    "
    texts = ["".join(rng.choice(letters) for _ in range(60)) for _ in range(50)]
    got = [metrics.classify_unit(text) for text in texts]
    print(f"random strings classified: {len(got)}")
    assert len(got) == 50
    assert set(got) <= codes


def test_unit_script_majority_and_pairs() -> None:
    pairs = metrics.SCRIPT_PAIRS
    print(f"script pairs: {len(pairs)}")
    assert len(pairs) >= 50
    simplified = {s for s, _ in pairs}
    traditional = {t for _, t in pairs}
    assert not simplified & traditional
    assert all(s != t for s, t in pairs)

    assert metrics.unit_script(ZH_TRADITIONAL) == "Traditional"
    assert metrics.unit_script(ZH) == "Simplified"
    assert metrics.unit_script("天地人山水火土木金石") is None


def test_m1_flags_empty_and_think_tags() -> None:
    assert not metrics.m1("").passed
    assert not metrics.m1("<think>x</think>ok").passed
    assert not metrics.m1("ok </think>").passed
    assert metrics.m1("A plain summary.").passed


def test_m4_counts_length_finishes() -> None:
    finishes = ["stop", "length", None, "length"]
    assert sum(metrics.m4(finish) for finish in finishes) == 2
    assert metrics.m4("length") is True
    assert metrics.m4("stop") is False


def test_m12a_prints_first_differing_char() -> None:
    assert metrics.m12a("same text", "same text") == metrics.M12a(True, None)
    result = metrics.m12a("abcd", "abXd")
    print(f"m12a: {result}")
    assert result == metrics.M12a(False, 2)
    assert metrics.m12a("abc", "abcd") == metrics.M12a(False, 3)


def test_m2_fails_on_an_off_target_map_call() -> None:
    """Language is scored on every completion, not only the returned result.

    Drift can start in a map call and vanish from the final text. Catches
    scoring only the returned result.
    """
    english_map_units = metrics.split_units(EN)
    german_final_units = metrics.split_units(DE)
    target = metrics.Target("de", None)

    result = metrics.m2([english_map_units, german_final_units], german_final_units, target)
    print(f"m2 per call: scored={result.scored} off_target={result.off_target}")
    assert result.passed is False
    assert result.off_target
    assert {off.call for off in result.off_target} == {0}

    final_only = metrics.m2([], german_final_units, target)
    print(f"m2 on the result alone: passed={final_only.passed}")
    assert final_only.passed is True

    unparsed = metrics.m2([None, german_final_units], [], target)
    assert unparsed.unparsed_calls == 1
    assert unparsed.passed is True


def test_m2_script_is_gated() -> None:
    traditional = metrics.Target("zh", "Traditional")
    assert metrics.m2([metrics.split_units(ZH)], [], traditional).passed is False
    assert metrics.m2([metrics.split_units(ZH_TRADITIONAL)], [], traditional).passed is True

    assert metrics.target_for("Chinese", "zh", "Traditional") == metrics.Target("zh", "Simplified")
    assert metrics.target_for(LANGUAGE_AUTO, "zh", "Traditional") == traditional

    tied = "这這中文内容讲得很清楚"  # one Simplified, one Traditional listed char
    print(f"tied unit, unscored for script: {tied}")
    assert metrics.is_scored(tied)
    assert metrics.unit_script(tied) is None
    assert metrics.m2([[tied]], [], traditional).passed is True


def test_every_output_language_maps_to_an_m2_code() -> None:
    """Every language a run can target resolves to a classifier code."""
    names = [name for name in OUTPUT_LANGUAGES if name != LANGUAGE_AUTO]
    names += [language for _, language in cases.EXPLICIT_CASES]
    codes = set(metrics.LANGUAGE_CODES.values())
    missing = [name for name in names if name not in metrics.LANGUAGE_CODES]
    for name in set(names) - set(missing):
        assert metrics.target_for(name, "en", None).code in codes
    labels = [(doc.label, doc.script) for doc in cases.DOCUMENTS.values()]
    missing += [label for label, _ in labels if label not in codes]
    for label, script in labels:
        if label in codes:
            assert metrics.target_for(LANGUAGE_AUTO, label, script).code == label
    print(f"names and labels checked: {len(names) + len(labels)}; missing: {missing}")
    assert names
    assert labels
    assert missing == []


def test_eval_never_imports_the_detector() -> None:
    """The evaluation never grades language with the product's own detector."""
    pattern = re.compile(r"detect_language|docsummarizer\.language")
    files = sorted((_REPO / "scripts" / "eval").glob("*.py"))
    hits = [path.name for path in files if pattern.search(path.read_text(encoding="utf-8"))]
    print(f"files scanned: {len(files)}; hits: {hits}")
    assert files
    assert hits == []
