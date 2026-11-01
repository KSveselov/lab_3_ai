"""Пять минимальных тестов лабораторной: данные, разбиение, визуализатор, модель."""

from pathlib import Path
import subprocess
import sys

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_ppe_system import predict, read_image
from visualize_ppe_errors import render_example, validate_record

CONFIG = yaml.safe_load((ROOT / "configs/ppe.yaml").read_text())
FOLDERS = {split: ROOT / CONFIG["path"] / CONFIG[split] for split in ("train", "val", "test")}
IMAGES = {split: sorted(folder.glob("*")) for split, folder in FOLDERS.items()}


def test_yaml_images_and_labels_are_readable():
    for split, folder in FOLDERS.items():
        assert folder.is_dir(), folder
        assert IMAGES[split], f"Пустая выборка: {split}"
        for image in IMAGES[split]:
            with Image.open(image) as source:
                source.load()
                assert min(source.size) > 0, image
            label = image.parent.parent / "labels" / f"{image.stem}.txt"
            assert label.is_file(), label
            label.read_text(encoding="utf-8")
    assert {split: len(images) for split, images in IMAGES.items()} == {
        "train": 1961, "val": 420, "test": 420,
    }


def test_annotation_classes_and_coordinates():
    for images in IMAGES.values():
        for image in images:
            label = image.parent.parent / "labels" / f"{image.stem}.txt"
            for number, line in enumerate(label.read_text().splitlines(), 1):
                location = f"{label}:{number}"
                values = line.split()
                assert len(values) == 5, location
                assert int(values[0]) in CONFIG["names"], location
                coordinates = [float(value) for value in values[1:]]
                assert all(0 <= value <= 1 for value in coordinates), location
                assert coordinates[2] > 0 and coordinates[3] > 0, location


def test_split_groups_and_hashes_do_not_overlap():
    # Переиспользуем проверку групп, имён и реальных SHA-256 из шага 3.
    subprocess.run([sys.executable, str(ROOT / "scripts/validate_ppe_split.py")], check=True)


def test_visualizer_image_size():
    record = {"image": str(IMAGES["test"][0]), "predictions": [],
              "truth": [{"class_id": 2, "box": [0.1, 0.1, 0.5, 0.5]}]}
    event = {"category": "critical_misses", "gt": 0, "pred": None}
    rendered = render_example(record, event)
    assert isinstance(rendered, Image.Image)
    assert rendered.size == (1280, 1000)
    assert rendered.mode == "RGB"


def test_real_model_prediction_schema():
    import torch
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "runs/ppe/e2-img960-colab-2/weights/best.pt"))
    assert model.names == CONFIG["names"]
    image = IMAGES["test"][0]
    device = "0" if torch.cuda.is_available() else "cpu"
    predictions, _ = predict(model, read_image(image), 960, device)
    assert isinstance(predictions, list)
    assert predictions, "На контрольном изображении ожидались обнаружения"
    for item in predictions:
        assert set(item) == {"class_id", "box", "confidence"}
    validate_record({"image": str(image), "truth": [], "predictions": predictions})
    print(f"Контроль модели: {image.name}; объектов: {len(predictions)}; устройство: {device}")
