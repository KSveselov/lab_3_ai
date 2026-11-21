from pathlib import Path
import random

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "Construction-Site-Safety-PPE-Detection-main"


def main() -> None:
    split_path = PROJECT / "data/splits/val.txt"
    images = split_path.read_text(encoding="utf-8").splitlines()
    if len(images) < 10:
        raise ValueError(f"Для E0 нужно минимум 10 изображений: {split_path}")

    selected_images = random.Random(42).sample(images, 10)
    sources = [str(PROJECT / image) for image in selected_images]
    model_path = PROJECT / "models/yolov8n.pt"
    for path in [model_path, *(Path(source) for source in sources)]:
        if not path.is_file():
            raise FileNotFoundError(f"Не найден входной файл E0: {path}")

    model = YOLO(str(model_path))
    results = model.predict(
        source=sources,
        imgsz=640,
        conf=0.25,
        device="cpu",
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(ROOT / "runs/ppe"),
        name="e0",
    )
    assert len(results) == 10, "Модель должна обработать все 10 изображений"

    output = Path(results[0].save_dir)
    (output / "selected_images.txt").write_text(
        "\n".join(selected_images) + "\n", encoding="utf-8"
    )
    print(f"Результаты E0 сохранены: {output}")


if __name__ == "__main__":
    main()
