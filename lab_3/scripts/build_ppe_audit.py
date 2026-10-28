"""Select reproducible PPE annotation audit samples and render labeled contact sheets."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CLASSES = (
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
)
COLORS = (
    "#00a878", "#1380d3", "#e83f3f", "#d649c5", "#f26924",
    "#ffe14b", "#9b62d6", "#00b7a7", "#9d9d9d", "#3773ff",
)
ROOT = Path(__file__).resolve().parents[1] / "Construction-Site-Safety-PPE-Detection-main"
DATA = ROOT / "data" / "raw"


def annotations(image: Path) -> list[tuple[int, float, float, float, float]]:
    label = image.parent.parent / "labels" / f"{image.stem}.txt"
    return [tuple(map(float, line.split())) for line in label.read_text().splitlines()]


def select_images() -> list[tuple[str, Path]]:
    images = sorted(DATA.glob("*/images/*.jpg"))
    rng = random.Random(42)
    random_set = rng.sample(images, 30)
    remaining = [p for p in images if p not in set(random_set)]
    small = [
        p for p in remaining
        if any(0.0001 <= w * h <= 0.001 for _, _, _, w, h in annotations(p))
    ]
    hard_set = rng.sample(small, 10)
    return [("random", p) for p in random_set] + [("small_object", p) for p in hard_set]


def render_tile(image_path: Path) -> Image.Image:
    with Image.open(image_path) as original:
        image = original.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for cls, cx, cy, bw, bh in annotations(image_path):
        cls = int(cls)
        x0, y0 = (cx - bw / 2) * width, (cy - bh / 2) * height
        x1, y1 = (cx + bw / 2) * width, (cy + bh / 2) * height
        color = COLORS[cls]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
        text = CLASSES[cls]
        box = draw.textbbox((x0, y0), text, font=ImageFont.load_default())
        draw.rectangle((box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1), fill=color)
        draw.text((x0, y0), text, fill="black", font=ImageFont.load_default())
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/ppe_audit_sheets"))
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "audit.csv")
    args = parser.parse_args()
    samples = select_images()
    args.output.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("выборка", "изображение", "разметка", "статус", "тип_проблемы", "замечание", "действие"))
        for kind, image in samples:
            label = image.parent.parent / "labels" / f"{image.stem}.txt"
            kind_ru = {"random": "случайная", "small_object": "малый_объект"}[kind]
            writer.writerow((kind_ru, image.relative_to(ROOT), label.relative_to(ROOT), "не_проверено", "", "", ""))
    for sheet_number in range(10):
        group = samples[sheet_number * 4:(sheet_number + 1) * 4]
        sheet = Image.new("RGB", (1280, 1360), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (kind, path) in enumerate(group):
            x, y = (i % 2) * 640, (i // 2) * 680
            sheet.paste(render_tile(path), (x, y + 40))
            title = f"{sheet_number * 4 + i + 1:02d} {kind} {path.parent.parent.name}/{path.name}"
            draw.text((x + 8, y + 10), title[:85], fill="black")
        sheet.save(args.output / f"sheet_{sheet_number + 1:02d}.jpg", quality=92)
    print(f"Selected {len(samples)} images; wrote {args.csv} and {args.output}")


if __name__ == "__main__":
    main()
