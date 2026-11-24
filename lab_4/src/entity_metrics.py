"""Общие метрики строгого совпадения сущностей для правил и моделей."""

from .generate_data import LABELS


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


def count_entities(expected, predicted, text, labels=LABELS):
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
        for label in labels
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


def evaluate_predictions(records, predictions, labels=LABELS):
    """Строгая оценка списков предсказаний в порядке исходных записей."""
    if len(records) != len(predictions):
        raise ValueError("Число предсказаний не совпадает с числом записей")
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}
    for record, predicted in zip(records, predictions):
        if not isinstance(record, dict) or not isinstance(record.get("text"), str):
            raise ValueError("Запись должна содержать строковое поле text")
        current = count_entities(record.get("entities"), predicted, record["text"], labels)
        for label in labels:
            for key in ("tp", "fp", "fn"):
                counts[label][key] += current[label][key]
    return {
        "records": len(records),
        "per_class": {label: calculate_metrics(counts[label]) for label in labels},
        **summarize(counts, labels),
    }
