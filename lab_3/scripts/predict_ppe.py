"""Сохранить эталоны и предсказания E2 для галерей ошибок.

CPU: python scripts/predict_ppe.py
GPU: env -u PYTHONPATH .venv-rocm/bin/python scripts/predict_ppe.py --device 0
Выход: reports/error-analysis/predictions.json.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

from build_ppe_audit import CLASSES, annotations
from visualize_ppe_errors import ROOT, validate_record


def read_ground_truth(image):
    truth = []
    for number, row in enumerate(annotations(image), 1):
        if len(row) != 5:
            raise ValueError(f"Неверная строка разметки: {image}, строка {number}")
        cls, cx, cy, width, height = row
        if (not cls.is_integer() or int(cls) not in range(len(CLASSES))
                or not all(math.isfinite(x) and 0 <= x <= 1 for x in row[1:])
                or width <= 0 or height <= 0):
            raise ValueError(f"Недопустимая разметка: {image}, строка {number}: {row}")
        truth.append({"class_id": int(cls), "box": [
            max(0, cx - width / 2), max(0, cy - height / 2),
            min(1, cx + width / 2), min(1, cy + height / 2),
        ]})
    return truth


def load_test_records():
    """Прочитать проверенные эталоны из замороженной тестовой выборки."""
    project = ROOT / "Construction-Site-Safety-PPE-Detection-main"
    split = project / "data/splits/test.txt"
    digest = hashlib.sha256(split.read_bytes()).hexdigest()
    if digest != split.with_suffix(".sha256").read_text().strip():
        raise ValueError("Изменён замороженный список теста")
    paths = split.read_text().splitlines()
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Тестовый список пуст или содержит повторяющиеся изображения")
    records = []
    for line in paths:
        relative = Path(line)
        image = project / relative
        if (relative.is_absolute() or ".." in relative.parts
                or not image.resolve().is_relative_to(project.resolve())):
            raise ValueError(f"Недопустимый путь тестового изображения: {line}")
        if not image.is_file():
            raise FileNotFoundError(f"Не найдено тестовое изображение: {image}")
        record = {"image": str(image.relative_to(ROOT)),
                  "truth": read_ground_truth(image), "predictions": []}
        validate_record(record)
        records.append(record)

    return records, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=ROOT / "runs/ppe/e2-img960-colab-2/weights/best.pt")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/error-analysis/predictions.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    weights_digest = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    records, digest = load_test_records()

    import torch
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    if [model.names[i] for i in range(len(model.names))] != list(CLASSES):
        raise ValueError("Словарь классов весов не совпадает с разметкой")
    for number, record in enumerate(records, 1):
        result = model.predict(str(ROOT / record["image"]), imgsz=960, conf=0.25,
                               iou=0.7, device=args.device, save=False, verbose=False)[0]
        boxes = result.boxes.cpu()
        record["predictions"] = [
            {"class_id": int(cls), "box": box, "confidence": confidence}
            for cls, box, confidence in zip(
                boxes.cls.tolist(), boxes.xyxyn.tolist(), boxes.conf.tolist())
        ]
        validate_record(record)
        if number % 20 == 0 or number == len(records):
            print(f"Обработано {number}/{len(records)}", flush=True)
    settings = {"weights": str(args.weights.resolve()), "weights_sha256": weights_digest,
                "test_sha256": digest, "imgsz": 960, "conf": 0.25, "nms_iou": 0.7,
                "match_iou": 0.5, "device": args.device, "torch": torch.__version__}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                         suffix=".json", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump({"settings": settings, "images": records}, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Готово: {len(records)} кадров. Предсказания сохранены: {args.output}")


if __name__ == "__main__":
    main()
