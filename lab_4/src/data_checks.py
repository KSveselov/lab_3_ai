"""Рабочая валидация синтетического корпуса и его разметки."""

import json
from collections import Counter

if __package__:
    from .generate_data import DATA_PATH, LABELS, TOKEN_PATTERN
else:
    from generate_data import DATA_PATH, LABELS, TOKEN_PATTERN


def validate_record(record):
    text = record["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Пустой текст")
    previous_end = 0
    for entity in record["entities"]:
        start, end = entity["start"], entity["end"]
        if type(start) is not int or type(end) is not int:
            raise ValueError("Позиции должны быть целыми")
        if not 0 <= start < end <= len(text):
            raise ValueError("Диапазон вне текста")
        if start < previous_end:
            raise ValueError("Пересекающиеся или неупорядоченные сущности")
        if entity["label"] not in LABELS:
            raise ValueError("Неизвестный класс")
        if text[start:end] != entity["text"]:
            raise ValueError("Неверная подстрока сущности")
        if entity["text"] != entity["text"].strip():
            raise ValueError("Пробелы на границах")
        previous_end = end

    tokens, tags = record["tokens"], record["bio_tags"]
    matches = list(TOKEN_PATTERN.finditer(text))
    if not len(tokens) == len(tags) == len(matches):
        raise ValueError("Число токенов и меток различается")
    for token, match in zip(tokens, matches):
        if token != {
            "text": match.group(), "start": match.start(), "end": match.end()
        }:
            raise ValueError("Неверные позиции токена")

    recovered = []
    for token, tag in zip(tokens, tags):
        if tag == "O":
            continue
        if tag[:2] not in ("B-", "I-") or tag[2:] not in LABELS:
            raise ValueError("Неверная BIO-метка")
        if tag.startswith("B-"):
            recovered.append({
                "start": token["start"], "end": token["end"], "label": tag[2:]
            })
        else:
            if not recovered:
                raise ValueError("I-метка без начала сущности")
            entity = recovered[-1]
            if entity["label"] != tag[2:]:
                raise ValueError("Смена типа внутри сущности")
            gap = text[entity["end"]:token["start"]]
            if gap and not gap.isspace():
                raise ValueError("Разрыв внутри BIO-сущности")
            entity["end"] = token["end"]
    expected = [
        {key: entity[key] for key in ("start", "end", "label")}
        for entity in record["entities"]
    ]
    if recovered != expected:
        raise ValueError("BIO-разметка не восстанавливает сущности")


def validate_corpus(corpus):
    if set(corpus) != {"train", "validation", "test"}:
        raise ValueError("Неверные части корпуса")
    seen_ids, seen_texts, seen_templates = set(), set(), set()
    totals = Counter()
    positives = negatives = 0
    for split, records in corpus.items():
        if not records:
            raise ValueError("Пустая выборка")
        template_ids = set()
        counts = Counter()
        negative_count = 0
        for record in records:
            validate_record(record)
            if record["split"] != split:
                raise ValueError("Неверная принадлежность выборке")
            if record["id"] in seen_ids:
                raise ValueError("Повтор идентификатора")
            if record["text"] in seen_texts:
                raise ValueError("Дубликат текста")
            seen_ids.add(record["id"])
            seen_texts.add(record["text"])
            template_ids.add(record["template_id"])
            counts.update(entity["label"] for entity in record["entities"])
            if record["entities"]:
                positives += 1
            else:
                negatives += 1
                negative_count += 1
        if template_ids & seen_templates:
            raise ValueError("Утечка шаблонов между выборками")
        seen_templates.update(template_ids)
        if any(counts[label] == 0 for label in LABELS):
            raise ValueError("В выборке отсутствует класс")
        if negative_count == 0:
            raise ValueError("Нет отрицательных примеров")
        totals.update(counts)
    if positives < 120:
        raise ValueError("Недостаточно записей с сущностями")
    if negatives < 30:
        raise ValueError("Недостаточно отрицательных записей")
    if any(totals[label] < 25 for label in LABELS):
        raise ValueError("Недостаточно сущностей класса")
    return totals


def main():
    corpus = {
        split: [
            json.loads(line)
            for line in (DATA_PATH / (split + ".jsonl")).read_text(encoding="utf-8").splitlines()
        ]
        for split in ("train", "validation", "test")
    }
    totals = validate_corpus(corpus)
    for split, records in corpus.items():
        negatives = sum(not record["entities"] for record in records)
        print("{}: записей {}, отрицательных {}".format(split, len(records), negatives))
    print("Сущности: " + ", ".join("{}={}".format(label, totals[label]) for label in LABELS))
    print("Валидация корпуса пройдена.")


if __name__ == "__main__":
    main()
