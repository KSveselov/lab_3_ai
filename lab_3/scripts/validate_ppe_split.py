"""Validate paths, annotation schema, hashes, and group isolation of the PPE split."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Construction-Site-Safety-PPE-Detection-main" / "data"
SPLITS = DATA / "splits"
PROCESSED = DATA / "processed"
FOLDERS = {"обучение": "train", "проверка": "val", "тест": "test"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-near-duplicates", action="store_true")
    args = parser.parse_args()
    with (DATA / "split_manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2801:
        raise ValueError(f"Expected 2801 images, got {len(rows)}")
    by_group: dict[str, set[str]] = defaultdict(set)
    by_hash: dict[str, set[str]] = defaultdict(set)
    by_name: dict[str, set[str]] = defaultdict(set)
    per_class: dict[str, Counter[int]] = defaultdict(Counter)
    images = Counter()
    perceptual_hashes: list[tuple[int, str, str]] = []
    for row in rows:
        split = row["новая_часть"]
        if split not in FOLDERS:
            raise ValueError(f"Unknown split: {split}")
        image = DATA.parent / row["изображение"]
        label = DATA.parent / row["разметка"]
        if not image.is_file() or not label.is_file():
            raise ValueError(f"Missing image or label: {image}")
        if hashlib.sha256(image.read_bytes()).hexdigest() != row["контрольная_сумма_SHA256"]:
            raise ValueError(f"Image hash changed: {image}")
        if args.check_near_duplicates:
            with Image.open(image) as source:
                pixels = list(source.convert("L").resize((9, 8)).getdata())
            image_hash = 0
            for y in range(8):
                for x in range(8):
                    image_hash = (image_hash << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
            perceptual_hashes.append((image_hash, split, row["изображение"]))
        folder = FOLDERS[split]
        if not (PROCESSED / folder / "images" / image.name).is_file():
            raise ValueError(f"Missing processed image link: {image.name}")
        if not (PROCESSED / folder / "labels" / label.name).is_file():
            raise ValueError(f"Missing processed label link: {label.name}")
        by_group[row["группа"]].add(split)
        by_hash[row["контрольная_сумма_SHA256"]].add(split)
        by_name[image.name].add(split)
        images[split] += 1
        for line_number, line in enumerate(label.read_text().splitlines(), 1):
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO line: {label}:{line_number}")
            cls = int(parts[0])
            box = [float(x) for x in parts[1:]]
            if cls not in range(10) or not all(0 <= x <= 1 for x in box) or box[2] <= 0 or box[3] <= 0:
                raise ValueError(f"Invalid YOLO values: {label}:{line_number}")
            per_class[split][cls] += 1
    for title, index in (("groups", by_group), ("SHA-256", by_hash), ("names", by_name)):
        overlaps = [key for key, splits in index.items() if len(splits) > 1]
        if overlaps:
            raise ValueError(f"{title} overlap between splits: {overlaps[:5]}")
    if any(per_class[split][cls] == 0 for split in FOLDERS for cls in range(10)):
        raise ValueError("A class is absent from a split")
    for split, folder in FOLDERS.items():
        listed = (SPLITS / f"{folder}.txt").read_text().splitlines()
        expected = [row["изображение"] for row in rows if row["новая_часть"] == split]
        if listed != expected:
            raise ValueError(f"{folder}.txt differs from frozen manifest")
    digest = hashlib.sha256((SPLITS / "test.txt").read_bytes()).hexdigest()
    if digest != (SPLITS / "test.sha256").read_text().strip():
        raise ValueError("Frozen test list SHA-256 changed")
    if args.check_near_duplicates:
        close = []
        for i, (left_hash, left_split, left_path) in enumerate(perceptual_hashes):
            for right_hash, right_split, right_path in perceptual_hashes[i + 1:]:
                if left_split != right_split and (left_hash ^ right_hash).bit_count() <= 5:
                    close.append((left_path, right_path))
        if close:
            raise ValueError(f"Potential near-duplicate images across splits: {close[:5]}")
        print("Между частями нет пар с расстоянием dHash не более 5 бит")
    print("Проверка пройдена: 2801 изображение и разметка читаются")
    print("Пересечений имён, SHA-256 и групп между частями: 0")
    for split in FOLDERS:
        print(split, "изображений", images[split], "объектов по классам", dict(sorted(per_class[split].items())))
    print("SHA-256 списка теста:", digest)


if __name__ == "__main__":
    main()
