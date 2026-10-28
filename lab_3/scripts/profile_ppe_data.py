"""Create a reproducible inventory of the downloaded PPE dataset."""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1] / "Construction-Site-Safety-PPE-Detection-main"
DATA = ROOT / "data"
RAW = DATA / "raw"
FRAME = re.compile(r"(?i)^(.+?)_(?:mp4|mov)[-_]\d+$")
AUGMENT = re.compile(r"\.rf\.[0-9a-f]+$")


def group_for(stem: str) -> tuple[str, str]:
    original = AUGMENT.sub("", stem)
    original = re.sub(r"(?i)_(?:jpg|jpeg|png)$", "", original)
    match = FRAME.match(original)
    if match:
        return f"видео_по_имени:{match.group(1).lower()}", "выведено_из_имени_файла"
    if original.startswith("youtube-"):
        return "источник_ютуб_неизвестен", "источник_неизвестен_объединено_консервативно"
    return f"исходное_изображение_по_имени:{original.lower()}", "выведено_из_имени_изображения"


def main() -> None:
    images = sorted(RAW.glob("*/images/*.jpg"))
    rows = []
    class_counts = Counter()
    problems = []
    for image in images:
        label = image.parent.parent / "labels" / f"{image.stem}.txt"
        with Image.open(image) as im:
            width, height = im.size
            im.verify()
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        group, quality = group_for(image.stem)
        counts = Counter()
        if not label.exists():
            problems.append(f"Missing label: {image}")
        else:
            for n, line in enumerate(label.read_text().splitlines(), 1):
                values = line.split()
                try:
                    if len(values) != 5:
                        raise ValueError("expected 5 fields")
                    cls = int(values[0])
                    cx, cy, bw, bh = map(float, values[1:])
                    if cls not in range(10) or not all(0 <= v <= 1 for v in (cx, cy, bw, bh)):
                        raise ValueError("class or coordinate out of range")
                    if bw <= 0 or bh <= 0:
                        raise ValueError("empty box")
                    counts[cls] += 1
                    class_counts[cls] += 1
                except ValueError as exc:
                    problems.append(f"{label}:{n}: {exc}")
        split_names = {"train": "обучение", "valid": "проверка", "test": "тест"}
        rows.append((
            image.relative_to(ROOT), label.relative_to(ROOT), split_names[image.parent.parent.name],
            width, height, digest, group, quality,
            ";".join(f"{cls}:{counts[cls]}" for cls in sorted(counts)),
        ))
    with (DATA / "manifest.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("изображение", "разметка", "исходная_часть", "ширина", "высота", "контрольная_сумма_SHA256",
                         "предварительная_группа", "основание_группировки", "число_объектов_по_классам"))
        writer.writerows(rows)
    print(f"Images: {len(rows)}; groups: {len(set(r[6] for r in rows))}; issues: {len(problems)}")
    print("Class counts:", dict(sorted(class_counts.items())))
    for issue in problems[:20]:
        print(issue)


if __name__ == "__main__":
    main()
