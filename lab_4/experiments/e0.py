"""E0: регулярные выражения и строгая оценка сущностей на validation."""

import json
import re

from src.generate_data import DATA_PATH, LABELS

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


def entity_keys(entities, text):
    """Проверка диапазонов и преобразование сущностей для строгого сравнения."""
    if not isinstance(entities, list):
        raise ValueError("Сущности должны быть списком")
    keys = set()
    for entity in entities:
        if not isinstance(entity, dict):
            raise ValueError("Сущность должна быть объектом")
        start, end, label = (entity.get(key) for key in ("start", "end", "label"))
        if type(start) is not int or type(end) is not int:
            raise ValueError("Позиции сущности должны быть целыми числами")
        if not 0 <= start < end <= len(text) or label not in LABELS:
            raise ValueError("Недопустимый диапазон или класс сущности")
        key = (start, end, label)
        if key in keys:
            raise ValueError("Повтор сущности с теми же границами и типом")
        keys.add(key)
    return keys


def count_entities(expected, predicted, text):
    """TP требует точного совпадения типа, начала и конца внутри одной записи."""
    expected_keys = entity_keys(expected, text)
    predicted_keys = entity_keys(predicted, text)
    matched = expected_keys & predicted_keys
    false_positive = predicted_keys - expected_keys
    false_negative = expected_keys - predicted_keys
    return {
        label: {
            "tp": sum(key[2] == label for key in matched),
            "fp": sum(key[2] == label for key in false_positive),
            "fn": sum(key[2] == label for key in false_negative),
        }
        for label in RULE_LABELS
    }


def calculate_metrics(counts):
    """Расчёт метрик; при нулевом знаменателе результат принят равным нулю."""
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    return dict(
        counts,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    )


def summarize(counts, labels):
    micro_counts = {
        key: sum(counts[label][key] for label in labels)
        for key in ("tp", "fp", "fn")
    }
    macro = {
        key: sum(calculate_metrics(counts[label])[key] for label in labels) / len(labels)
        for key in ("precision", "recall", "f1")
    }
    return {"micro": calculate_metrics(micro_counts), "macro": macro}


def evaluate(records):
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in RULE_LABELS}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("text"), str):
            raise ValueError("Запись должна содержать строковое поле text")
        text = record["text"]
        current = count_entities(record.get("entities"), extract_entities(text), text)
        for label in RULE_LABELS:
            for key in ("tp", "fp", "fn"):
                counts[label][key] += current[label][key]
    return {
        "records": len(records),
        "per_class": {label: calculate_metrics(counts[label]) for label in RULE_LABELS},
        **summarize(counts, RULE_LABELS),
    }



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
