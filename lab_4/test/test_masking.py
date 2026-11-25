"""Проверки замены по символьным позициям и обработки перекрытий."""

import csv
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from src.masking import main, mask_entities


class MaskingTests(unittest.TestCase):
    def test_replaces_synthetic_identifiers_and_preserves_other_text(self):
        text = "Карта 0000 123456\nТелефон +7 000 1234567. Доза 5 мг."
        masked = mask_entities(text, [
            {"start": 26, "end": 40, "label": "PHONE"},
            {"start": 6, "end": 17, "label": "ID"},
        ])
        self.assertEqual(masked, "Карта [ID]\nТелефон [PHONE]. Доза 5 мг.")

    def test_masks_entire_overlapping_chain(self):
        entities = [
            {"start": 2, "end": 6, "label": "ID"},
            {"start": 5, "end": 8, "label": "PHONE"},
            {"start": 7, "end": 9, "label": "ID"},
        ]
        self.assertEqual(mask_entities("0123456789", entities), "01[REDACTED]9")

    def test_keeps_label_for_same_type_and_does_not_merge_adjacent_ranges(self):
        entities = [
            {"start": 0, "end": 3, "label": "NAME"},
            {"start": 1, "end": 4, "label": "NAME"},
            {"start": 4, "end": 6, "label": "AGE"},
        ]
        self.assertEqual(mask_entities("абвгде", entities), "[NAME][AGE]")

    def test_coalesces_exact_duplicates(self):
        entity = {"start": 0, "end": 5, "label": "AGE"}
        self.assertEqual(mask_entities("5 лет", [entity, entity]), "[AGE]")

    def test_empty_predictions_preserve_text(self):
        self.assertEqual(mask_entities("Температура 37,2 °C.", []), "Температура 37,2 °C.")

    def test_rejects_invalid_range_or_label(self):
        for entity in (
            {"start": -1, "end": 2, "label": "NAME"},
            {"start": 1, "end": 1, "label": "NAME"},
            {"start": 0, "end": 20, "label": "NAME"},
            {"start": 0, "end": 2, "label": "UNKNOWN"},
        ):
            with self.subTest(entity=entity), self.assertRaises(ValueError):
                mask_entities("Текст", [entity])

    def test_cli_saves_original_and_masked_text_to_csv(self):
        text = 'Карта 0000 123456,\nжалоба "боль".'
        record = {"id": "sample", "text": text}
        prediction = {"id": "sample", "entities": [{"start": 6, "end": 17, "label": "ID"}]}
        with TemporaryDirectory() as directory:
            records = Path(directory) / "records.jsonl"
            predictions = Path(directory) / "predictions.jsonl"
            records.write_text(json.dumps(record), encoding="utf-8")
            predictions.write_text(json.dumps(prediction), encoding="utf-8")
            default_output = Path(directory) / "records-masked.csv"
            custom_output = Path(directory) / "custom.csv"
            for output, flags in (
                (default_output, []),
                (custom_output, ["--output", str(custom_output)]),
            ):
                with self.subTest(output=output.name):
                    with redirect_stdout(io.StringIO()) as stdout:
                        main([str(records), str(predictions), *flags])
                    self.assertEqual(stdout.getvalue(), "")
                    with output.open(encoding="utf-8", newline="") as source:
                        result = csv.DictReader(source)
                        self.assertEqual(result.fieldnames, ["id", "original_text", "masked_text"])
                        self.assertEqual(list(result), [{
                            "id": "sample",
                            "original_text": text,
                            "masked_text": 'Карта [ID],\nжалоба "боль".',
                        }])
