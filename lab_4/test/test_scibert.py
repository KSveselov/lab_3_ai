"""Тесты согласования подтокенов и ячеек обучения из ноутбука Colab."""

import json
from pathlib import Path

import pytest

from src.generate_data import render_record
from src.token_classification import (
    ID2LABEL,
    LABEL2ID,
    align_labels,
    decode_entities,
    encode_record,
)

NOTEBOOK = Path(__file__).resolve().parents[1] / "experiments" / "e0_scibert.ipynb"


def test_alignment_special_and_continuation_tokens():
    result = align_labels(["B-NAME", "O"], [None, 0, 0, 1, 1, None, None])
    assert result == [-100, LABEL2ID["B-NAME"], LABEL2ID["I-NAME"], 0, 0, -100, -100]


@pytest.mark.parametrize("word_id", [-1, 3, True])
def test_alignment_invalid_word_id(word_id):
    with pytest.raises(ValueError):
        align_labels(["O"], [word_id])


def test_decode_repairs_orphan_inside_tag():
    result = decode_entities(
        [LABEL2ID["I-NAME"], LABEL2ID["I-NAME"], 0, LABEL2ID["I-AGE"]],
        [(0, 3), (3, 7), (7, 8), (9, 11)],
    )
    assert result == [
        {"start": 0, "end": 7, "label": "NAME"},
        {"start": 9, "end": 11, "label": "AGE"},
    ]


@pytest.fixture
def notebook_scope():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    scope = {"__name__": "notebook_test"}
    import sys
    from types import ModuleType
    from unittest.mock import patch
    google = ModuleType("google")
    colab = ModuleType("google.colab")
    colab.files = ModuleType("google.colab.files")
    google.colab = colab
    for cell in notebook["cells"]:
        tags = cell.get("metadata", {}).get("tags", [])
        if "imports" in tags or "functions" in tags:
            with patch.dict(sys.modules, {"google": google, "google.colab": colab}):
                exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), scope)
    return scope


@pytest.fixture
def tokenizer(tmp_path, notebook_scope):
    from transformers import BertTokenizerFast

    vocabulary = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "пациент", "учеб",
        "##нов", "алексей", ",", "45", "лет", ".", "осмотр",
    ]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocabulary), encoding="utf-8")
    return BertTokenizerFast(vocab_file=str(path), do_lower_case=True)


def test_encoding_restores_original_entity_spans(tokenizer, notebook_scope):
    record = render_record("Пациент {NAME}, {AGE}.", {"NAME": "Учебнов Алексей", "AGE": "45 лет"})
    features, spans = notebook_scope["encode_record"](record, tokenizer, 64)
    assert features["labels"][0] == features["labels"][-1] == -100
    assert len(features["input_ids"]) == len(features["labels"]) == len(spans)
    expected = [{key: entity[key] for key in ("start", "end", "label")} for entity in record["entities"]]
    assert decode_entities(features["labels"], spans) == expected
    for start, end in [span for span in spans if span is not None]:
        assert record["text"][start:end]
    with pytest.raises(ValueError, match="max_length"):
        notebook_scope["encode_record"](record, tokenizer, 2)


def test_padding_labels_are_ignored(tokenizer, notebook_scope):
    records = [
        render_record("Пациент {NAME}, {AGE}.", {"NAME": "Учебнов Алексей", "AGE": "45 лет"}),
        render_record("Осмотр.", {}),
    ]
    features = [encode_record(record, tokenizer, 64)[0] for record in records]
    loader = notebook_scope["make_loader"](features, tokenizer, {"batch_size": 2, "seed": 42})
    batch = next(iter(loader))
    for index, feature in enumerate(features):
        assert (batch["labels"][index, len(feature["labels"]):] == -100).all()


def test_notebook_training_and_checkpoint(tokenizer, notebook_scope, tmp_path):
    torch = notebook_scope["torch"]
    from transformers import BertConfig, BertForTokenClassification

    notebook_scope["set_seed"](42)
    torch.set_num_threads(1)
    model = BertForTokenClassification(BertConfig(
        vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=64,
        num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
        attn_implementation="eager",
    ))
    records = [
        render_record("Пациент {NAME}, {AGE}.", {"NAME": "Учебнов Алексей", "AGE": "45 лет"}),
        render_record("Осмотр.", {}),
    ]
    frozen_before = {key: value.clone() for key, value in model.base_model.state_dict().items()}
    head_before = model.classifier.weight.detach().clone()
    config = {
        "max_length": 64, "batch_size": 2, "seed": 42, "epochs": 2,
        "learning_rate": 0.001, "weight_decay": 0.01,
    }
    metrics, predictions, history, epoch = notebook_scope["train_head"](
        model, tokenizer, records, records, config, torch.device("cpu"),
    )
    assert len(predictions) == len(records)
    assert len(history) == config["epochs"]
    best_score = max(row["validation_micro_f1"] for row in history)
    assert epoch == next(row["epoch"] for row in history if row["validation_micro_f1"] == best_score)
    assert 0 <= metrics["micro"]["f1"] <= 1
    assert not torch.equal(model.classifier.weight, head_before)
    assert all(torch.equal(value, frozen_before[key]) for key, value in model.base_model.state_dict().items())
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint)
    restored = BertForTokenClassification.from_pretrained(checkpoint, local_files_only=True)
    encoded = [encode_record(record, tokenizer, 64) for record in records]
    loader = notebook_scope["make_loader"]([item[0] for item in encoded], tokenizer, config)
    restored_metrics, restored_predictions = notebook_scope["evaluate_model"](
        restored, loader, records, [item[1] for item in encoded], torch.device("cpu"),
    )
    assert restored_predictions == predictions
    assert restored_metrics == metrics



def test_notebook_metrics_match_shared_metrics(notebook_scope):
    from src.entity_metrics import evaluate_predictions

    records = [render_record("{NAME}, {AGE}.", {"NAME": "Учебнов", "AGE": "45 лет"})]
    predictions = [[{"start": 0, "end": 7, "label": "NAME"}]]
    assert notebook_scope["evaluate_predictions"](records, predictions) == evaluate_predictions(records, predictions)



@pytest.mark.parametrize("as_zip", [False, True])
def test_notebook_uploads_data(monkeypatch, notebook_scope, as_zip):
    import io
    from zipfile import ZipFile

    data_dir = NOTEBOOK.parents[1] / "data" / "synthetic"
    uploaded = {
        name: (data_dir / name).read_bytes()
        for name in ("train.jsonl", "validation.jsonl")
    }
    if as_zip:
        buffer = io.BytesIO()
        with ZipFile(buffer, "w") as archive:
            for name, content in uploaded.items():
                archive.writestr(name, content)
        uploaded = {"e0_scibert_data.zip": buffer.getvalue()}
    monkeypatch.setattr(notebook_scope["files"], "upload", lambda: uploaded, raising=False)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if "data" in cell.get("metadata", {}).get("tags", []):
            exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), notebook_scope)
    assert notebook_scope["train_records"]
    assert notebook_scope["validation_records"]
    assert set(notebook_scope["uploaded"]) == {"train.jsonl", "validation.jsonl"}
