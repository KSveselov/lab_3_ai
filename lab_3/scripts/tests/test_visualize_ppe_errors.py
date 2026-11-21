"""Check diagnostic matching, candidate categories and rendered image size."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("errors", ROOT / "scripts/visualize_ppe_errors.py")
errors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(errors)


def box(cls, coords, confidence=None):
    result = {"class_id": cls, "box": coords}
    if confidence is not None:
        result["confidence"] = confidence
    return result


class ErrorGalleryTest(unittest.TestCase):
    def test_duplicate_prediction_matches_only_once(self):
        truth = [box(2, [0, 0, 1, 1])]
        predictions = [box(2, [0, 0, 1, 1], 0.9), box(2, [0, 0, 1, 1], 0.8)]
        matches, missed, extra = errors.match_boxes(truth, predictions)
        self.assertEqual(len(matches), 1)
        self.assertEqual(missed, set())
        self.assertEqual(len(extra), 1)
        self.assertEqual(errors.iou([0, 0, 1, 1], [1, 1, 2, 2]), 0)

    def test_confusion_miss_and_annotation_candidate(self):
        truth = [box(2, [0, 0, 0.1, 0.1])]
        predictions = [box(0, [0, 0, 0.1, 0.1], 0.9), box(5, [0.5, 0.5, 1, 1], 0.95)]
        events = errors.classify_errors(truth, predictions)
        categories = {event["category"] for event in events}
        self.assertTrue({"critical_misses", "class_confusion", "high_confidence_fp", "annotation_suspects"} <= categories)
        self.assertEqual(sum(e["category"] == "annotation_suspects" for e in events), 1)

    def test_low_iou_small_object_and_renderer(self):
        truth = [box(0, [0.1, 0.1, 0.12, 0.12])]
        predictions = [box(0, [0.105, 0.1, 0.125, 0.12], 0.8)]
        events = errors.classify_errors(truth, predictions)
        self.assertEqual({e["category"] for e in events}, {"low_iou", "small_occluded"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            image = root / "images/sample.jpg"
            Image.new("RGB", (640, 640)).save(image)
            record = {"image": str(image), "truth": truth, "predictions": predictions}
            self.assertEqual(errors.render_example(record, events[0]).size, (1280, 1000))
            self.assertIn("IoU = 0.600", errors.caption(record, events[0]))

    def test_invalid_class_is_rejected(self):
        record = {
            "image": str(ROOT / "scripts/visualize_ppe_errors.py"),
            "truth": [box(10, [0, 0, 1, 1])],
            "predictions": [],
        }
        with self.assertRaises(ValueError):
            errors.validate_record(record)


if __name__ == "__main__":
    unittest.main()
