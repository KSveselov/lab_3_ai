"""Согласование BIO с подтокенами и восстановление символьных диапазонов."""

from .data_checks import validate_record
from .generate_data import LABELS

BIO_LABELS = ("O",) + tuple(prefix + label for label in LABELS for prefix in ("B-", "I-"))
LABEL2ID = {label: index for index, label in enumerate(BIO_LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}


def align_labels(word_labels, word_ids):
    """Последующие подтокены сущности получают I-метку, служебные — -100."""
    result = []
    previous = None
    for word_id in word_ids:
        if word_id is None:
            result.append(-100)
        else:
            if type(word_id) is not int or not 0 <= word_id < len(word_labels):
                raise ValueError("Подтокен ссылается на отсутствующее слово")
            label = word_labels[word_id]
            if word_id == previous and label.startswith("B-"):
                label = "I-" + label[2:]
            if label not in LABEL2ID:
                raise ValueError("Неизвестная BIO-метка: " + label)
            result.append(LABEL2ID[label])
        previous = word_id
    return result


def encode_record(record, tokenizer, max_length):
    """Подготовка входов и абсолютных позиций без скрытого обрезания текста."""
    validate_record(record)
    if not tokenizer.is_fast:
        raise ValueError("Требуется быстрый токенизатор с word_ids()")
    words = [token["text"] for token in record["tokens"]]
    encoded = tokenizer(
        words, is_split_into_words=True, return_offsets_mapping=True,
        truncation=False, padding=False,
    )
    if len(encoded["input_ids"]) > max_length:
        raise ValueError("Запись {} превышает max_length={}".format(
            record.get("id", "без идентификатора"), max_length
        ))
    word_ids = encoded.word_ids()
    if {word_id for word_id in word_ids if word_id is not None} != set(range(len(words))):
        raise ValueError("Токенизатор пропустил исходные слова")
    offsets = encoded.pop("offset_mapping")
    spans = []
    for word_id, (start, end) in zip(word_ids, offsets):
        if word_id is None:
            spans.append(None)
            continue
        if not 0 <= start < end <= len(words[word_id]):
            raise ValueError("Некорректные позиции подтокена")
        word_start = record["tokens"][word_id]["start"]
        spans.append((word_start + start, word_start + end))
    features = dict(encoded)
    features["labels"] = align_labels(record["bio_tags"], word_ids)
    return features, spans


def decode_entities(predictions, spans):
    """Восстановление сущностей; одиночная I-метка считается началом сущности."""
    if len(predictions) != len(spans):
        raise ValueError("Число предсказаний и диапазонов различается")
    entities = []
    current = None
    previous_end = 0
    for prediction, span in zip(predictions, spans):
        if span is None:
            current = None
            continue
        if prediction not in ID2LABEL:
            raise ValueError("Неизвестный номер BIO-метки")
        start, end = span
        if start < previous_end or start >= end:
            raise ValueError("Диапазоны подтокенов пересекаются или пусты")
        previous_end = end
        tag = ID2LABEL[prediction]
        if tag == "O":
            current = None
            continue
        label = tag[2:]
        if tag.startswith("I-") and current is not None and current["label"] == label:
            current["end"] = end
        else:
            current = {"start": start, "end": end, "label": label}
            entities.append(current)
    return entities
