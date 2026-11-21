"""Check E2 configuration and metric export without training a model."""

import contextlib
import io
import json
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


class E2Test(unittest.TestCase):
    def test_only_resolution_changes_between_experiments(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            values = {"metrics/mAP50(B)": 0.5, "metrics/mAP50-95(B)": 0.3}
            metrics = SimpleNamespace(
                save_dir=output,
                results_dict=values,
                box=SimpleNamespace(map50=0.5, map=0.3),
            )
            for experiment in ("e1", "e2"):
                model = Mock()
                model.trainer.save_dir = output
                model.val.return_value = metrics
                factory = Mock(return_value=model)
                with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=factory)}):
                    with contextlib.redirect_stdout(io.StringIO()):
                        runpy.run_path(str(ROOT / "experiments" / f"{experiment}.py"), run_name="__main__")
                factory.assert_called_once_with(
                    str(ROOT / "Construction-Site-Safety-PPE-Detection-main/models/yolov8n.pt")
                )
                model.train.assert_called_once()
                model.val.assert_called_once()
                calls.append((model.train.call_args.kwargs.copy(), model.val.call_args.kwargs.copy()))
                self.assertEqual(json.loads((output / "metrics.json").read_text()), values)

            baseline_train, baseline_val = calls[0]
            changed_train, changed_val = calls[1]
            self.assertEqual(baseline_train.pop("imgsz"), 640)
            self.assertEqual(changed_train.pop("imgsz"), 960)
            self.assertEqual(baseline_train.pop("name"), "e1-opora")
            self.assertEqual(changed_train.pop("name"), "e2-img960")
            self.assertEqual(baseline_train, changed_train)
            self.assertEqual(baseline_val.pop("imgsz"), 640)
            self.assertEqual(changed_val.pop("imgsz"), 960)
            self.assertEqual(baseline_val, changed_val)


if __name__ == "__main__":
    unittest.main()
