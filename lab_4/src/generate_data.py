"""Генерация воспроизводимого русского синтетического корпуса без внешних данных."""

import json
import random
import re
from datetime import date, timedelta
from pathlib import Path
from string import Formatter

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "synthetic"
LABELS = (
    "NAME", "DATE", "PHONE", "EMAIL", "ID", "LOCATION", "ORGANIZATION", "AGE"
)
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

MANIFEST_PATH = DATA_PATH / "manifest.json"


def load_manifest(path=MANIFEST_PATH):
    """Чтение параметров генерации и проверка допустимости полей шаблонов."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("Не удалось прочитать манифест {}: {}".format(path, error)) from error
    if not isinstance(manifest, dict):
        raise ValueError("Манифест должен содержать объект JSON")
    if type(manifest.get("seed")) is not int:
        raise ValueError("seed должен быть целым числом")
    count = manifest.get("records_per_template")
    if type(count) is not int or count <= 0:
        raise ValueError("records_per_template должен быть положительным целым числом")
    templates = manifest.get("templates")
    if not isinstance(templates, dict) or not templates:
        raise ValueError("templates должен содержать непустой объект шаблонов")
    negative_fields = {"days", "temperature", "dose", "pulse", "value", "breathing", "percent"}
    family_counts = {"clinical": 0, "contact": 0, "negative": 0}
    for template_id, template in templates.items():
        if not re.fullmatch(r"(clinical|contact|negative)_[A-Za-z0-9]+", template_id):
            raise ValueError("Неизвестная группа шаблона: " + template_id)
        family = template_id.split("_", 1)[0]
        family_counts[family] += 1
        if not isinstance(template, str) or not template.strip():
            raise ValueError("Пустой или нестроковый шаблон: " + template_id)
        allowed_fields = negative_fields if family == "negative" else set(LABELS)
        try:
            for _, field, spec, conversion in Formatter().parse(template):
                if field is None:
                    continue
                if field not in allowed_fields or spec or conversion:
                    raise ValueError("Недопустимое поле: " + field)
        except ValueError as error:
            raise ValueError("Шаблон {}: {}".format(template_id, error)) from error
    if any(count < 3 for count in family_counts.values()):
        raise ValueError("Для каждой группы требуется минимум три шаблона")
    return manifest


def annotate_text(text, entities):
    """Выделение пунктуации в отдельные токены и построение BIO по символьным диапазонам."""
    tokens = []
    bio_tags = []
    for match in TOKEN_PATTERN.finditer(text):
        start, end = match.span()
        tokens.append({"text": match.group(), "start": start, "end": end})
        tag = "O"
        for entity in entities:
            if start < entity["end"] and end > entity["start"]:
                if not entity["start"] <= start < end <= entity["end"]:
                    raise ValueError("Граница сущности проходит внутри токена")
                prefix = "B-" if start == entity["start"] else "I-"
                tag = prefix + entity["label"]
                break
        bio_tags.append(tag)
    return tokens, bio_tags


def render_record(template, values):
    """Вычисление диапазонов при подстановке значений, включая повторные вхождения."""
    fragments = []
    entities = []
    position = 0
    for literal, field, _, _ in Formatter().parse(template):
        fragments.append(literal)
        position += len(literal)
        if field is None:
            continue
        value = values[field]
        fragments.append(value)
        if field in LABELS:
            entities.append({
                "start": position,
                "end": position + len(value),
                "label": field,
                "text": value,
            })
        position += len(value)
    text = "".join(fragments)
    tokens, bio_tags = annotate_text(text, entities)
    return {"text": text, "entities": entities, "tokens": tokens, "bio_tags": bio_tags}


def sample_values(rng, template_id, index):
    surname = rng.choice(("Учебнов", "Примеров", "Тестов-Учебнов"))
    given_name = rng.choice(("Алексей", "Иван", "Павел"))
    name = rng.choice((
        surname + " " + given_name,
        surname + " " + given_name[0] + ". П.",
        surname + "\n" + given_name + " Петрович",
    ))
    if template_id == "clinical_03":
        surname_forms = {
            "Учебнов": "Учебнову",
            "Примеров": "Примерову",
            "Тестов-Учебнов": "Тестову-Учебнову",
        }
        given_forms = {"Алексей": "Алексею", "Иван": "Ивану", "Павел": "Павлу"}
        name = surname_forms[surname] + " " + given_forms[given_name]
    day = date(2024, 1, 1) + timedelta(days=rng.randrange(1096))
    months = (
        "января", "февраля", "марта", "апреля", "мая", "июня",
        "июля", "августа", "сентября", "октября", "ноября", "декабря",
    )
    date_text = rng.choice((
        day.strftime("%d.%m.%Y"),
        day.strftime("%d-%m-%Y"),
        "{} {} {} года".format(day.day, months[day.month - 1], day.year),
    ))
    # Все телефонные номера используют заведомо недействительный код 000.
    phone_tail = "{:07d}".format(rng.randrange(10 ** 7))
    phone = rng.choice((
        "+7 (000) {}-{}-{}".format(phone_tail[:3], phone_tail[3:5], phone_tail[5:]),
        "+7 000 " + phone_tail,
        "8-000-" + phone_tail,
    ))
    return {
        "NAME": name,
        "DATE": date_text,
        "PHONE": phone,
        "EMAIL": "sample.{}.{}@example.org".format(template_id, index),
        "ID": "0000 {:06d}".format(rng.randrange(10 ** 6)),
        "LOCATION": rng.choice((
            "г. Петрозаводск", "г. Учебноград", "ул. Учебная, д. 0",
            "г. Примерск, ул. Тестовая, д. 0",
        )),
        "ORGANIZATION": rng.choice((
            "ГБУЗ «Учебная больница»", "ООО «Тестовая клиника»",
            "ГБУЗ «Примерский медицинский центр»",
        )),
        "AGE": rng.choice(("87 лет", "45 лет", "5 месяцев", "21 год", "32 года")),
    }


def negative_values(rng, index):
    return {
        "days": str(index + 1),
        "temperature": "36,{}".format(rng.randrange(10)),
        "dose": str(rng.choice((5, 10, 20))),
        "pulse": str(60 + index),
        "value": "{},{}".format(index + 1, rng.randrange(10)),
        "breathing": str(12 + index),
        "percent": str(index + 1),
    }


def generate_corpus(manifest=None):
    if manifest is None:
        manifest = load_manifest()
    rng = random.Random(manifest["seed"])
    templates = manifest["templates"]
    groups = {split: [] for split in ("train", "validation", "test")}
    for family in ("clinical", "contact", "negative"):
        template_ids = sorted(
            template_id for template_id in templates
            if template_id.startswith(family + "_")
        )
        rng.shuffle(template_ids)
        groups["train"].extend(template_ids[:-2])
        groups["validation"].append(template_ids[-2])
        groups["test"].append(template_ids[-1])

    corpus = {}
    for split, template_ids in groups.items():
        records = []
        for template_id in template_ids:
            for index in range(manifest["records_per_template"]):
                if template_id.startswith("negative_"):
                    values = negative_values(rng, index)
                else:
                    values = sample_values(rng, template_id, index)
                record = render_record(templates[template_id], values)
                record.update({
                    "id": "{}_{:03d}".format(template_id, index),
                    "template_id": template_id,
                    "split": split,
                })
                records.append(record)
        rng.shuffle(records)
        corpus[split] = records
    return corpus


def main():
    if __package__:
        from .data_checks import validate_corpus
    else:
        from data_checks import validate_corpus

    corpus = generate_corpus()
    validate_corpus(corpus)
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    for split, records in corpus.items():
        content = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
        (DATA_PATH / (split + ".jsonl")).write_text(content, encoding="utf-8")
    print("Корпус создан: " + ", ".join(
        "{}={}".format(split, len(records)) for split, records in corpus.items()
    ))


if __name__ == "__main__":
    main()
