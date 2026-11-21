import json
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "Construction-Site-Safety-PPE-Detection-main"


def main() -> None:
    model_path = PROJECT / "models/yolov8n.pt"
    data_path = ROOT / "configs/ppe.yaml"
    for path in (model_path, data_path):
        if not path.is_file():
            raise FileNotFoundError(f"Не найден входной файл E1: {path}")

    model = YOLO(str(model_path))
    model.train(
        data=str(data_path),
        epochs=25,
        imgsz=640,
        batch=16,
        seed=42,
        deterministic=True,
        device=0,
        project=str(ROOT / "runs/ppe"),
        name="e1-opora",
    )
    # Ultralytics загружает лучшие сохранённые веса после обучения.
    metrics = model.val(
        data=str(data_path),
        split="test",
        imgsz=640,
        batch=16,
        device=0,
        plots=True,
        project=str(model.trainer.save_dir),
        name="test",
    )
    output = Path(metrics.save_dir)
    (output / "metrics.json").write_text(
        json.dumps(metrics.results_dict, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Результаты E1 сохранены: {model.trainer.save_dir}")
    print(f"Оценка на тесте сохранена: {output}")


if __name__ == "__main__":
    main()
