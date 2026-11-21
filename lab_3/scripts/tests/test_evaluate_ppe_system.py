"""Проверки подсчёта тревог, статистик задержки и стресс-преобразований без GPU."""

import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_ppe_system import aggregate, load_test_records, perturb, score, timing_summary
import predict_ppe


class SystemEvaluationTest(unittest.TestCase):
    def test_loads_test_data_without_predictions_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "Construction-Site-Safety-PPE-Detection-main"
            image = project / "data/raw/test/images/sample.jpg"
            image.parent.mkdir(parents=True)
            image.touch()
            label = image.parent.parent / "labels/sample.txt"
            label.parent.mkdir()
            label.write_text("2 0.5 0.5 0.2 0.4\n")
            split = project / "data/splits/test.txt"
            split.parent.mkdir()
            split.write_text("data/raw/test/images/sample.jpg\n")
            digest = hashlib.sha256(split.read_bytes()).hexdigest()
            split.with_suffix(".sha256").write_text(digest)
            with patch.object(predict_ppe, "ROOT", root), patch("visualize_ppe_errors.ROOT", root):
                records, actual_digest = load_test_records()
                self.assertEqual(actual_digest, digest)
                self.assertEqual(records, [{
                    "image": str(image.relative_to(root)),
                    "truth": [{"class_id": 2, "box": [0.4, 0.3, 0.6, 0.7]}],
                    "predictions": [],
                }])
                split.write_text("../outside.jpg\n")
                with self.assertRaisesRegex(ValueError, "замороженный список"):
                    load_test_records()

    def test_false_alarms_and_critical_recall(self):
        truth = [{"class_id": 2, "box": [0, 0, 0.5, 0.5]}]
        predictions = [truth[0], {"class_id": 3, "box": [0.6, 0.6, 1, 1]},
                       {"class_id": 5, "box": [0.6, 0.6, 1, 1]}]
        summary = aggregate([score(truth, predictions), score(truth, [])])
        self.assertEqual((summary["tp"], summary["fp"], summary["fn"]), (1, 2, 1))
        self.assertEqual(summary["fp_per_100_frames"], 100)
        self.assertEqual(summary["alarm_fp_per_100_frames"], 50)
        self.assertEqual(summary["critical_recall"], 0.5)
        self.assertIsNone(aggregate([score([], [])])["recall"])

    def test_latency_percentiles(self):
        keys = ("decode_ms", "resize_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "overhead_ms", "total_ms")
        rows = [{key: value for key in keys} for value in (1, 2, 3, 4, 5)]
        self.assertEqual(timing_summary(rows)["total_ms"]["median"], 3)
        self.assertAlmostEqual(timing_summary(rows)["total_ms"]["p95"], 4.8)

    def test_brightness_and_occlusion_do_not_change_source(self):
        image = np.full((100, 100, 3), 100, dtype=np.uint8)
        truth = [{"class_id": 2, "box": [0.2, 0.2, 0.6, 0.6]}]
        before = copy.deepcopy(truth)
        self.assertEqual(perturb(image, truth, "dark")[0][0, 0, 0], 50)
        self.assertEqual(perturb(image, truth, "bright")[0][0, 0, 0], 150)
        changed, labels = perturb(image, truth, "occluded")
        self.assertEqual(np.count_nonzero(changed[:, :, 0] != image[:, :, 0]), 800)
        self.assertEqual(labels, before)
        self.assertTrue(np.all(image == 100))
        self.assertEqual(truth, before)

    def test_rotation_transforms_boxes_and_preserves_classes(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        truth = [{"class_id": 2, "box": [0.4, 0.4, 0.6, 0.6]}]
        changed, labels = perturb(image, truth, "rotated")
        self.assertEqual(changed.shape, image.shape)
        self.assertEqual(labels[0]["class_id"], 2)
        self.assertTrue(all(0 <= coordinate <= 1 for coordinate in labels[0]["box"]))
        self.assertLess(labels[0]["box"][0], 0.4)
        self.assertGreater(labels[0]["box"][2], 0.6)
        self.assertEqual(truth[0]["box"], [0.4, 0.4, 0.6, 0.6])
        with self.assertRaises(ValueError):
            perturb(image, truth, "unknown")


if __name__ == "__main__":
    unittest.main()
