"""Тесты регулярных выражений и строгих метрик эксперимента E0."""

import json
from pathlib import Path

import pytest

from experiments.e0 import (
    RULE_LABELS,
    calculate_metrics,
    count_entities,
    evaluate,
    extract_entities,
    main,
    summarize,
)


def test_e0_solution():
    """Контроль правил, строгих границ, типов и ручного примера TP/FP/FN."""
    examples = (
        ("Почта: test@example.org.", "EMAIL", "test@example.org"),
        ("Телефон: +7 (000) 123-45-67.", "PHONE", "+7 (000) 123-45-67"),
        ("Телефон: +7 000 1234567.", "PHONE", "+7 000 1234567"),
        ("Телефон: 8-000-1234567.", "PHONE", "8-000-1234567"),
        ("Полис: 0000 123456.", "ID", "0000 123456"),
        ("Почта: 80001234567@example.org.", "EMAIL", "80001234567@example.org"),
    )
    for text, label, fragment in examples:
        start = text.index(fragment)
        assert extract_entities(text) == [
            {"start": start, "end": start + len(fragment), "label": label}
        ]
    assert not extract_entities("Температура 37,2 °C, доза 5 мг, кашель 3 дня.")
    assert not extract_entities("Числа: 180001234567, 0000 1234567. Дата: 12.03.2026.")
    assert not extract_entities("Телефон с ошибкой: +7 (000 123-45-67.")
    text = "a b c d e"
    expected = [
        {"start": 0, "end": 1, "label": "PHONE"},
        {"start": 2, "end": 3, "label": "EMAIL"},
        {"start": 4, "end": 5, "label": "ID"},
    ]
    predicted = [
        {"start": 0, "end": 1, "label": "PHONE"},
        {"start": 2, "end": 4, "label": "EMAIL"},
        {"start": 4, "end": 5, "label": "PHONE"},
    ]
    counts = count_entities(expected, predicted, text)
    assert counts["PHONE"] == {"tp": 1, "fp": 1, "fn": 0}
    assert counts["EMAIL"] == {"tp": 0, "fp": 1, "fn": 1}
    assert counts["ID"] == {"tp": 0, "fp": 0, "fn": 1}
    summary = summarize(counts, RULE_LABELS)
    assert summary["micro"]["tp"] == 1
    assert summary["micro"]["fp"] == summary["micro"]["fn"] == 2
    assert summary["micro"]["f1"] == pytest.approx(1 / 3)
    assert summary["macro"]["f1"] == pytest.approx(2 / 9)
    assert calculate_metrics({"tp": 0, "fp": 0, "fn": 0})["f1"] == 0
    repeated = {"text": "test@example.org", "entities": [
        {"start": 0, "end": 16, "label": "EMAIL"}
    ]}
    assert evaluate([repeated, repeated])["per_class"]["EMAIL"]["tp"] == 2
    assert evaluate([])["micro"]["f1"] == 0
    result = evaluate([repeated])
    assert set(result["per_class"]) == set(RULE_LABELS)
    assert set(result) == {"records", "per_class", "micro", "macro"}
    unrelated = {"text": "Учебнов", "entities": [
        {"start": 0, "end": 7, "label": "NAME"}
    ]}
    assert evaluate([unrelated])["micro"]["fn"] == 0


def test_e0_results(capsys):
    main()
    actual = json.loads(capsys.readouterr().out)
    path = Path(__file__).resolve().parents[1] / "reports" / "e0-validation.json"
    assert actual == json.loads(path.read_text(encoding="utf-8"))
