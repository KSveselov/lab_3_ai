"""Проверить экспорт без загрузки модели и запуска GPU."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("predict_ppe", ROOT / "scripts/predict_ppe.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class PredictionExportTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        project = self.root / "Construction-Site-Safety-PPE-Detection-main"
        self.image = project / "data/raw/test/images/sample.jpg"
        self.image.parent.mkdir(parents=True)
        Image.new("RGB", (640, 640)).save(self.image)
        self.label = self.image.parent.parent / "labels/sample.txt"
        self.label.parent.mkdir()
        self.label.write_text("2 0.5 0.5 0.2 0.4\n")
        self.split = project / "data/splits/test.txt"
        self.split.parent.mkdir()
        self.split.write_text("data/raw/test/images/sample.jpg\n")
        self.split.with_suffix(".sha256").write_text(
            hashlib.sha256(self.split.read_bytes()).hexdigest())
        self.weights = self.root / "best.pt"
        self.weights.write_bytes(b"test weights")
        self.output = self.root / "reports/predictions.json"
        boxes = MagicMock()
        boxes.cpu.return_value = boxes
        boxes.cls.tolist.return_value = [2.0]
        boxes.xyxyn.tolist.return_value = [[0.4, 0.3, 0.6, 0.7]]
        boxes.conf.tolist.return_value = [0.9]
        self.model = MagicMock()
        self.model.names = dict(enumerate(exporter.CLASSES))
        self.model.predict.return_value = [SimpleNamespace(boxes=boxes)]

    def run_export(self):
        with patch.object(exporter, "ROOT", self.root), \
                patch("visualize_ppe_errors.ROOT", self.root), \
                patch.dict(sys.modules, {
                    "torch": SimpleNamespace(__version__="test"),
                    "ultralytics": SimpleNamespace(YOLO=MagicMock(return_value=self.model)),
                }), \
                patch.object(sys, "argv", ["predict_ppe.py", "--weights", str(self.weights),
                                           "--output", str(self.output)]):
            exporter.main()

    def test_exports_compatible_records_and_settings(self):
        self.run_export()
        data = json.loads(self.output.read_text())
        record = data["images"][0]
        self.assertEqual(record["truth"], [
            {"class_id": 2, "box": [0.4, 0.3, 0.6, 0.7]}])
        self.assertEqual(record["predictions"], [
            {"class_id": 2, "box": [0.4, 0.3, 0.6, 0.7], "confidence": 0.9}])
        self.assertEqual(data["settings"]["test_sha256"],
                         hashlib.sha256(self.split.read_bytes()).hexdigest())
        self.model.predict.assert_called_once_with(
            str(self.image), imgsz=960, conf=0.25, iou=0.7,
            device="cpu", save=False, verbose=False)

    def test_changed_split_is_rejected(self):
        self.split.write_text("../outside.jpg\n")
        with self.assertRaisesRegex(ValueError, "замороженный список"):
            self.run_export()
        self.model.predict.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_invalid_label_is_rejected(self):
        self.label.write_text("2.5 0.5 0.5 0.2 0.4\n")
        with self.assertRaisesRegex(ValueError, "Недопустимая разметка"):
            self.run_export()
        self.model.predict.assert_not_called()

    def test_inference_failure_preserves_existing_file(self):
        self.output.parent.mkdir()
        self.output.write_text("previous predictions\n")
        self.model.predict.side_effect = RuntimeError("inference failed")
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.run_export()
        self.assertEqual(self.output.read_text(), "previous predictions\n")


if __name__ == "__main__":
    unittest.main()
