"""E0: регулярные выражения и строгая оценка сущностей на validation."""

import json
import re

from src.generate_data import DATA_PATH
from src.entity_metrics import (
    calculate_metrics,
    count_entities,
    entity_keys,
    evaluate_predictions,
    summarize,
)

RULE_LABELS = ("EMAIL", "PHONE", "ID")
PATTERNS = {
    "EMAIL": re.compile(
        r"(?<![\w.+-])[A-Za-z0-9](?:[A-Za-z0-9._%+-]*[A-Za-z0-9])?@"
        r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+"
        r"[A-Za-z]{2,}(?![\w-])"
    ),
    "PHONE": re.compile(
        r"(?<![\w+])(?:\+7|8)[ \t-]*(?:\([0-9]{3}\)|[0-9]{3})"
        r"[ \t-]*[0-9]{3}[ \t-]*[0-9]{2}[ \t-]*[0-9]{2}(?!\w)"
    ),
    "ID": re.compile(r"(?<!\w)[0-9]{4}[ \t]+[0-9]{6}(?!\w)"),
}


def extract_entities(text):
    """Поиск непересекающихся сущностей с приоритетом EMAIL, PHONE, ID."""
    if not isinstance(text, str):
        raise ValueError("Для поиска сущностей требуется строка")
    entities = []
    for label, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < entity["end"] and end > entity["start"] for entity in entities):
                continue
            entities.append({"start": start, "end": end, "label": label})
    return sorted(entities, key=lambda entity: entity["start"])






def evaluate(records):
    predictions = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("text"), str):
            raise ValueError("Запись должна содержать строковое поле text")
        predictions.append(extract_entities(record["text"]))
    return evaluate_predictions(records, predictions, RULE_LABELS)



def main():
    path = DATA_PATH / "validation.jsonl"
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            records.append(json.loads(line))
        except ValueError as error:
            raise ValueError("Некорректный JSON в {}:{}".format(path, number)) from error
    if not records:
        raise ValueError("Проверочная выборка пуста: " + str(path))
    result = {"experiment": "E0-regex", "split": "validation", **evaluate(records)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
