"""Тесты генерации, разметки, валидации и воспроизводимости корпуса."""

import json
from copy import deepcopy

import pytest

from src import generate_data
from src.data_checks import validate_corpus, validate_record
from src.generate_data import DATA_PATH, generate_corpus, load_manifest, render_record


def test_annotation_examples():
    record = render_record("Пациент: {NAME}, {AGE}.", {"NAME": "Учебнов А. П.", "AGE": "45 лет"})
    assert record["entities"] == [
        {"start": 9, "end": 22, "label": "NAME", "text": "Учебнов А. П."},
        {"start": 24, "end": 30, "label": "AGE", "text": "45 лет"},
    ]
    assert record["bio_tags"] == [
        "O", "O", "B-NAME", "I-NAME", "I-NAME", "I-NAME", "I-NAME",
        "O", "B-AGE", "I-AGE", "O",
    ]
    validate_record(record)
    repeated = render_record("{NAME} и {NAME}.", {"NAME": "Учебнов"})
    assert [(e["start"], e["end"]) for e in repeated["entities"]] == [(0, 7), (10, 17)]
    validate_record(repeated)
    multiline = render_record("{NAME}", {"NAME": "Тестов-Учебнов\nАлексей"})
    assert multiline["bio_tags"] == ["B-NAME", "I-NAME", "I-NAME", "I-NAME"]
    validate_record(multiline)
    validate_record(render_record("Температура 37,2 °C, доза 5 мг.", {}))


@pytest.mark.parametrize("case", ["bounds", "overlap", "substring", "bio", "token"])
def test_invalid_annotations(case):
    record = render_record("{NAME}, {AGE}.", {"NAME": "Учебнов", "AGE": "45 лет"})
    if case == "bounds":
        record["entities"][0]["end"] = len(record["text"]) + 1
    elif case == "overlap":
        record["entities"][1]["start"] = 0
    elif case == "substring":
        record["entities"][0]["text"] = "Другое имя"
    elif case == "bio":
        record["bio_tags"][0] = "I-NAME"
    else:
        record["tokens"][0]["start"] = 1
    with pytest.raises(ValueError):
        validate_record(record)


def test_manifest_settings():
    manifest = load_manifest()
    baseline = generate_corpus(manifest)
    changed = deepcopy(manifest)
    changed["seed"] += 1
    assert generate_corpus(changed) != baseline, "Изменение seed не влияет на корпус"
    changed = deepcopy(manifest)
    changed["records_per_template"] += 1
    enlarged = generate_corpus(changed)
    assert sum(map(len, enlarged.values())) == (
        len(changed["templates"]) * changed["records_per_template"]
    ), "Число записей не соответствует манифесту"
    changed = deepcopy(manifest)
    changed["templates"]["clinical_01"] = "Учебная запись. " + changed["templates"]["clinical_01"]
    updated = generate_corpus(changed)
    records = [r for rows in updated.values() for r in rows if r["template_id"] == "clinical_01"]
    assert all(r["text"].startswith("Учебная запись. ") for r in records)
    validate_corpus(updated)


@pytest.mark.parametrize("content", ["{", "[]"])
def test_invalid_manifest_json(tmp_path, content):
    path = tmp_path / "manifest.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(path)


@pytest.mark.parametrize("field,value", [
    ("seed", True), ("records_per_template", 0), ("templates", {}),
])
def test_invalid_manifest_parameters(tmp_path, field, value):
    manifest = load_manifest()
    manifest[field] = value
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(path)


@pytest.mark.parametrize("template", ["{UNKNOWN}", "{NAME!r}", "{NAME:>10}", "{NAME", ""])
def test_invalid_template(tmp_path, template):
    manifest = load_manifest()
    manifest["templates"]["clinical_01"] = template
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(path)


def test_insufficient_templates(tmp_path):
    manifest = load_manifest()
    del manifest["templates"]["negative_01"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="минимум три"):
        load_manifest(path)


def test_saved_corpus_is_reproducible():
    manifest = load_manifest()
    generated = generate_corpus(manifest)
    assert generated == generate_corpus(manifest)
    corpus = {
        split: [
            json.loads(line)
            for line in (DATA_PATH / (split + ".jsonl")).read_text(encoding="utf-8").splitlines()
        ]
        for split in ("train", "validation", "test")
    }
    assert corpus == generated
    validate_corpus(corpus)


def test_template_leakage_is_rejected():
    corpus = generate_corpus()
    corpus["test"][0]["template_id"] = corpus["train"][0]["template_id"]
    with pytest.raises(ValueError, match="Утечка"):
        validate_corpus(corpus)


def test_generation_writes_only_corpus(tmp_path, monkeypatch):
    manifest_before = generate_data.MANIFEST_PATH.read_bytes()
    monkeypatch.setattr(generate_data, "DATA_PATH", tmp_path)
    generate_data.main()
    assert {path.name for path in tmp_path.iterdir()} == {
        "train.jsonl", "validation.jsonl", "test.jsonl"
    }
    assert generate_data.MANIFEST_PATH.read_bytes() == manifest_before
