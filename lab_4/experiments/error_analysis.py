"""Галерея ошибок из сохранённых предсказаний моделей."""

import argparse
import csv
import json
from pathlib import Path
from zipfile import ZipFile

from experiments.e0 import RULE_LABELS, extract_entities
from src.entity_metrics import entity_keys
from src.generate_data import DATA_PATH
from src.masking import mask_entities

ROOT = DATA_PATH.parents[1]
RUNS = (
    ("SciBERT", "validation", ROOT / "reports" / "e0-scibert-validation-predictions.jsonl"),
    ("SciBERT", "test", ROOT / "reports" / "e0-scibert-test-predictions.jsonl"),
    ("SciBERT", "diagnostic", ROOT / "reports" / "e0-scibert-diagnostic-predictions.jsonl"),
    ("RuBERT", "validation", ROOT / "experiments" / "e1_rubert_results.zip"),
    ("RuBERT", "test", ROOT / "experiments" / "e1_rubert_results.zip"),
)
FIELDS = (
    "source", "model", "split", "record_id", "category", "masked_context",
    "expected", "predicted", "analysis",
)


def describe(spans):
    return ", ".join(f"{label} [{start}, {end})" for start, end, label in sorted(spans)) or "—"


def as_entities(spans):
    return [{"start": start, "end": end, "label": label} for start, end, label in sorted(spans)]


def classify_errors(expected, predicted):
    """Сначала сопоставляет точные границы, затем максимальные пересечения."""
    missing = expected - predicted
    extras = predicted - expected
    candidates = []
    for gold in missing:
        for guess in extras:
            overlap = min(gold[1], guess[1]) - max(gold[0], guess[0])
            if overlap > 0:
                priority = (gold[:2] == guess[:2], gold[2] == guess[2], overlap)
                candidates.append((priority, gold, guess))

    for _, gold, guess in sorted(candidates, reverse=True):
        if gold not in missing or guess not in extras:
            continue
        missing.remove(gold)
        extras.remove(guess)
        if gold[:2] == guess[:2]:
            yield "неверный тип", gold, guess, "Фрагмент скрывается маркером другого класса."
        elif gold[2] != guess[2]:
            yield "ошибка типа и границ", gold, guess, "Маркер другого класса может скрыть лишнее или оставить часть сущности."
        else:
            yield "ошибка границ", gold, guess, "Маска захватывает лишние символы или оставляет часть сущности."

    for gold in sorted(missing):
        yield "пропуск", gold, None, "Отсутствует отдельное предсказание для эталонной сущности."
    for guess in sorted(extras):
        yield "ложное срабатывание", None, guess, "Лишнее предсказание может скрыть текст, который следовало сохранить."


def run_errors(model, split, predictions):
    records_path = (
        ROOT / "data" / "diagnostic" / "scibert.jsonl"
        if split == "diagnostic"
        else DATA_PATH / f"{split}.jsonl"
    )
    with records_path.open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source]
    by_id = {row["id"]: row["entities"] for row in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {row["id"] for row in records}:
        raise ValueError(f"Идентификаторы предсказаний не совпадают с корпусом: {model}, {split}")

    for record in records:
        text = record["text"]
        expected = entity_keys(record["entities"], text)
        predicted = entity_keys(by_id[record["id"]], text)
        errors = list(classify_errors(expected, predicted))
        rules = entity_keys(extract_entities(text), text)
        covered = {span for span in predicted if span[2] in RULE_LABELS}
        if rules != covered:
            yield (
                "эксперимент", model, split, record["id"], "конфликт правила и модели",
                mask_entities(text, as_entities(expected | predicted | rules)),
                describe(expected), f"Правила: {describe(rules)}; модель: {describe(covered)}",
                "Решения расходятся по одному или нескольким диапазонам EMAIL, PHONE или ID.",
            )
        if not errors:
            continue
        context = mask_entities(text, as_entities(expected | predicted))
        for category, gold, guess, note in errors:
            if gold is None and text[guess[0]:guess[1]].casefold() == "боль":
                category = "ложное срабатывание на клиническом термине"
                note = "Предсказание маскирует слово «боль», обозначающее симптом."
            yield (
                "эксперимент", model, split, record["id"], category, context,
                describe([gold] if gold else []), describe([guess] if guess else []), note,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", "-o", type=Path, default=ROOT / "reports" / "error-analysis.csv",
        help="Путь к CSV (по умолчанию reports/error-analysis.csv)",
    )
    parser.add_argument(
        "--predictions", nargs=3, action="append", default=[],
        metavar=("MODEL", "SPLIT", "JSONL"),
        help="Дополнительные сохранённые предсказания на выборке validation или test",
    )
    args = parser.parse_args()
    runs = list(RUNS)
    for model, split, path in args.predictions:
        if split not in {"validation", "test"}:
            parser.error("В --predictions выборка должна быть validation или test")
        runs.append((model, split, Path(path)))

    rows = []
    for model, split, path in runs:
        if path.suffix == ".zip":
            with ZipFile(path) as archive, archive.open(f"{split}-predictions.jsonl") as source:
                predictions = [json.loads(line) for line in source]
        else:
            with path.open(encoding="utf-8") as source:
                predictions = [json.loads(line) for line in source]
        rows.extend(run_errors(model, split, predictions))

    with args.output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(FIELDS)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
