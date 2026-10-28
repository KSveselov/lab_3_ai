"""Create a frozen, group-disjoint PPE split and materialize YOLO paths."""

from __future__ import annotations

import csv
import hashlib
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "Construction-Site-Safety-PPE-Detection-main"
DATA = PROJECT / "data"
MANIFEST = DATA / "manifest.csv"
SPLIT_MANIFEST = DATA / "split_manifest.csv"
SPLITS = DATA / "splits"
PROCESSED = DATA / "processed"
CONFIG = ROOT / "configs" / "ppe.yaml"
TEST_DIGEST = SPLITS / "test.sha256"
NAMES = (
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
)


def counts_for(row: dict[str, str]) -> Counter[int]:
    return Counter({int(k): int(v) for k, v in
                    (part.split(":") for part in row["число_объектов_по_классам"].split(";") if part)})


def choose_split(rows: list[dict[str, str]]) -> dict[str, str]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["предварительная_группа"]].append(row)
    unknown = "источник_ютуб_неизвестен"
    assert unknown in groups
    candidates = sorted(set(groups) - {unknown})
    totals = Counter()
    per_group = {}
    for group, items in groups.items():
        counts = Counter()
        for item in items:
            counts.update(counts_for(item))
        per_group[group] = counts
        totals.update(counts)
    target = round(len(rows) * 0.15)
    best_score = float("inf")
    best = None
    for attempt in range(2000):
        shuffled = candidates[:]
        random.Random(42 + attempt).shuffle(shuffled)
        picked: dict[str, set[str]] = {}
        available = shuffled
        for split in ("тест", "проверка"):
            selected = set()
            size = 0
            for group in available:
                group_size = len(groups[group])
                if size + group_size <= target:
                    selected.add(group)
                    size += group_size
                if size == target:
                    break
            picked[split] = selected
            available = [g for g in available if g not in selected]
        score = 0.0
        for split in ("тест", "проверка"):
            selected = picked[split]
            size = sum(len(groups[g]) for g in selected)
            score += abs(size - target) / target * 100
            counts = Counter()
            for group in selected:
                counts.update(per_group[group])
            for cls in range(10):
                expected = totals[cls] * 0.15
                score += abs(counts[cls] - expected) / max(expected, 1)
                if counts[cls] == 0:
                    score += 1000
        if score < best_score:
            best_score = score
            best = picked
    assert best is not None
    assignment = {group: "обучение" for group in groups}
    for split in ("тест", "проверка"):
        assignment.update({group: split for group in best[split]})
    return assignment


def write_frozen_split(rows: list[dict[str, str]], assignment: dict[str, str]) -> None:
    with SPLIT_MANIFEST.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("изображение", "разметка", "исходная_часть", "новая_часть",
                         "группа", "контрольная_сумма_SHA256", "число_объектов_по_классам"))
        for row in rows:
            writer.writerow((row["изображение"], row["разметка"], row["исходная_часть"],
                             assignment[row["предварительная_группа"]],
                             row["предварительная_группа"], row["контрольная_сумма_SHA256"],
                             row["число_объектов_по_классам"]))


def materialize() -> None:
    with SPLIT_MANIFEST.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for split, folder in (("обучение", "train"), ("проверка", "val"), ("тест", "test")):
        items = [r for r in rows if r["новая_часть"] == split]
        list_path = SPLITS / f"{folder}.txt"
        # The manifest is immutable after creation; lists are deterministic derivatives.
        list_path.write_text("".join(r["изображение"] + "\n" for r in items))
        for row in items:
            image = PROJECT / row["изображение"]
            label = PROJECT / row["разметка"]
            for source, kind in ((image, "images"), (label, "labels")):
                target = PROCESSED / folder / kind / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                link = os.path.relpath(source, target.parent)
                if target.is_symlink():
                    if os.readlink(target) != link:
                        raise ValueError(f"Conflicting link: {target}")
                elif target.exists():
                    raise ValueError(f"Unexpected existing file: {target}")
                else:
                    target.symlink_to(link)
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    yaml = [f"path: {PROCESSED}", "train: train/images", "val: val/images", "test: test/images", "names:"]
    yaml.extend(f"  {i}: {name}" for i, name in enumerate(NAMES))
    CONFIG.write_text("\n".join(yaml) + "\n")


def main() -> None:
    with MANIFEST.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    SPLITS.mkdir(parents=True, exist_ok=True)
    if not SPLIT_MANIFEST.exists():
        assignment = choose_split(rows)
        write_frozen_split(rows, assignment)
        print("Created frozen split_manifest.csv")
    else:
        print("Using existing frozen split_manifest.csv")
        if TEST_DIGEST.exists():
            with SPLIT_MANIFEST.open(newline="") as stream:
                frozen = list(csv.DictReader(stream))
            proposed = "".join(
                row["изображение"] + "\n" for row in frozen if row["новая_часть"] == "тест"
            )
            if hashlib.sha256(proposed.encode()).hexdigest() != TEST_DIGEST.read_text().strip():
                raise ValueError("Frozen test assignment has changed")
    materialize()
    with SPLIT_MANIFEST.open(newline="") as stream:
        frozen = list(csv.DictReader(stream))
    sizes = Counter(r["новая_часть"] for r in frozen)
    print("Image counts:", dict(sizes))
    digest = hashlib.sha256((SPLITS / "test.txt").read_bytes()).hexdigest()
    if TEST_DIGEST.exists():
        if TEST_DIGEST.read_text().strip() != digest:
            raise ValueError("Frozen test list has changed")
    else:
        TEST_DIGEST.write_text(digest + "\n")
    print("Test list SHA-256:", digest)


if __name__ == "__main__":
    main()
