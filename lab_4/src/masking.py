"""Замена предсказанных сущностей маркерами по символьным позициям."""

import argparse
import csv
import json
import sys
from pathlib import Path

from .entity_metrics import entity_keys


def mask_entities(text, entities):
    """Маскирует объединение перекрытий и сохраняет остальной текст."""
    if not isinstance(text, str):
        raise ValueError("Для маскирования требуется строка")
    spans = sorted(entity_keys(entities, text, allow_duplicates=True))
    merged = []
    for start, end, label in spans:
        if merged and start < merged[-1][1]:
            previous_start, previous_end, previous_label = merged[-1]
            if label != previous_label:
                label = "REDACTED"
            merged[-1] = (previous_start, max(end, previous_end), label)
        else:
            merged.append((start, end, label))

    parts = []
    cursor = len(text)
    for start, end, label in reversed(merged):
        parts.append(text[end:cursor])
        parts.append(f"[{label}]")
        cursor = start
    parts.append(text[:cursor])
    return "".join(reversed(parts))


def _read_by_id(lines, source):
    indexed = {}
    for number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except ValueError:
            raise ValueError(f"{source}, строка {number} — некорректный JSON") from None
        record_id = record.get("id") if isinstance(record, dict) else None
        if not isinstance(record_id, str) or not record_id or record_id in indexed:
            raise ValueError(f"{source}, строка {number} — неверный или повторный ID")
        indexed[record_id] = record
    return indexed


def main(argv=None):
    parser = argparse.ArgumentParser(description="Маскирование записей по предсказаниям модели")
    parser.add_argument("records", type=Path, help="JSONL с полями id и text")
    parser.add_argument(
        "predictions", type=Path, nargs="?",
        help="JSONL с полями id и entities (по умолчанию stdin)",
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        help="Путь к выходному CSV (по умолчанию <имя>-masked.csv рядом с входным JSONL)",
    )
    args = parser.parse_args(argv)
    output = args.output or args.records.with_name(f"{args.records.stem}-masked.csv")
    try:
        with args.records.open(encoding="utf-8") as source:
            records = _read_by_id(source, "Корпус")
        if args.predictions is None:
            predictions = _read_by_id(sys.stdin, "Предсказания")
        else:
            with args.predictions.open(encoding="utf-8") as source:
                predictions = _read_by_id(source, "Предсказания")
        if records.keys() != predictions.keys():
            raise ValueError("Идентификаторы записей и предсказаний не совпадают")

        masked = []
        for record_id, record in records.items():
            original_text = record.get("text")
            masked_text = mask_entities(original_text, predictions[record_id].get("entities"))
            masked.append({
                "id": record_id,
                "original_text": original_text,
                "masked_text": masked_text,
            })

        with output.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(
                destination, fieldnames=("id", "original_text", "masked_text"),
            )
            writer.writeheader()
            writer.writerows(masked)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Ошибка маскирования — {error}\n")


if __name__ == "__main__":
    main()
