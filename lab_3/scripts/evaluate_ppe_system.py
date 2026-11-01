"""Шаг 7: задержка, память, ложные тревоги и устойчивость E1/E2.

Запуск: env -u PYTHONPATH .venv-rocm/bin/python scripts/evaluate_ppe_system.py
Пороги фиксированы: confidence=0.25, NMS IoU=0.7, сопоставление IoU>=0.5.
"""

import argparse
import csv
import gc
import hashlib
import json
import platform
import random
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import psutil
from PIL import Image, ImageDraw, ImageFont

from visualize_ppe_errors import ROOT, match_boxes, validate_record

MODELS = (("E1", "e1-opora-colab-2", 640), ("E2", "e2-img960-colab-2", 960))
VARIANTS = {
    "original": "Исходное изображение",
    "dark": "Яркость × 0.5",
    "bright": "Яркость × 1.5",
    "occluded": "Закрыта половина выбранной рамки",
    "rotated": "Наклон на 15 градусов",
}


def score(truth, predictions):
    matches, missed, extra = match_boxes(truth, predictions)
    alarm_errors = [p for p in extra if predictions[p]["class_id"] in (2, 3, 4)]
    return {
        "frames": 1, "tp": len(matches), "fn": len(missed), "fp": len(extra),
        "frames_with_fp": int(bool(extra)), "alarm_fp": len(alarm_errors),
        "frames_with_alarm_fp": int(bool(alarm_errors)),
        "critical_gt": sum(gt["class_id"] == 2 for gt in truth),
        "critical_tp": sum(truth[g]["class_id"] == 2 for g, _, _ in matches),
    }


def aggregate(scores):
    counts = {key: sum(row[key] for row in scores) for key in scores[0]}
    tp, fp, fn, frames = (counts[key] for key in ("tp", "fp", "fn", "frames"))
    return {**counts, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "critical_recall": counts["critical_tp"] / counts["critical_gt"] if counts["critical_gt"] else None,
            "fp_per_100_frames": fp * 100 / frames,
            "alarm_fp_per_100_frames": counts["alarm_fp"] * 100 / frames,
            "alarm_frames_per_100": counts["frames_with_alarm_fp"] * 100 / frames}


def timing_summary(rows):
    keys = ("decode_ms", "resize_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "overhead_ms", "total_ms")
    return {key: {"median": float(np.median([row[key] for row in rows])),
                  "p95": float(np.percentile([row[key] for row in rows], 95))} for key in keys}


def perturb(image, truth, variant):
    """Создать проверочное ухудшение; исходное изображение и разметка не меняются."""
    height, width = image.shape[:2]
    if variant in ("dark", "bright"):
        factor = 0.5 if variant == "dark" else 1.5
        return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8), truth
    if variant == "occluded":
        candidates = [gt for gt in truth if gt["class_id"] == 2]
        candidates = candidates or [gt for gt in truth if gt["class_id"] == 5] or truth
        target = max(candidates, key=lambda gt: (gt["box"][2] - gt["box"][0]) *
                     (gt["box"][3] - gt["box"][1]))["box"] if candidates else [0.3, 0.3, 0.7, 0.7]
        x0, y0, x1, y1 = target
        changed = image.copy()
        changed[int(y0 * height):int(y1 * height), int(x0 * width):int((x0 + x1) * width / 2)] = 114
        return changed, truth
    if variant == "rotated":
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 15, 1)
        changed = cv2.warpAffine(image, matrix, (width, height), borderValue=(114, 114, 114))
        transformed = []
        for gt in truth:
            x0, y0, x1, y1 = gt["box"]
            corners = np.array([[x0 * width, y0 * height, 1], [x1 * width, y0 * height, 1],
                                [x1 * width, y1 * height, 1], [x0 * width, y1 * height, 1]]) @ matrix.T
            low = np.clip(corners.min(axis=0) / [width, height], 0, 1)
            high = np.clip(corners.max(axis=0) / [width, height], 0, 1)
            if np.all(high > low):
                transformed.append({"class_id": gt["class_id"], "box": [*low.tolist(), *high.tolist()]})
        return changed, transformed
    if variant != "original":
        raise ValueError(f"Неизвестное преобразование: {variant}")
    return image, truth


def predict(model, image, size, device):
    result = model.predict(image, imgsz=size, batch=1, conf=0.25, iou=0.7,
                           quantize=None, rect=False, device=device, save=False, verbose=False)[0]
    boxes = result.boxes.cpu()
    predictions = [{"class_id": int(cls), "box": box, "confidence": confidence}
                   for cls, box, confidence in zip(boxes.cls.tolist(), boxes.xyxyn.tolist(), boxes.conf.tolist())]
    return predictions, result.speed


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Не удалось декодировать изображение: {path}")
    return image


def evaluate(model, records, size, device, torch):
    process = psutil.Process()
    timings, scores = [], []
    rss_peak = process.memory_info().rss
    for record in records[:10]:
        predict(model, read_image(ROOT / record["image"]), size, device)
    if model.predictor.model.fp16:
        raise ValueError("Для сравнения требуется FP32")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for number, record in enumerate(records, 1):
        torch.cuda.synchronize()
        start = perf_counter()
        image = read_image(ROOT / record["image"])
        decoded = perf_counter()
        if image.shape[:2] != (size, size):
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        resized = perf_counter()
        predictions, stages = predict(model, image, size, device)
        torch.cuda.synchronize()
        end = perf_counter()
        row = {"image": record["image"], "decode_ms": (decoded - start) * 1000,
               "resize_ms": (resized - decoded) * 1000,
               **{key + "_ms": value for key, value in stages.items()}, "total_ms": (end - start) * 1000}
        row["overhead_ms"] = row["total_ms"] - row["decode_ms"] - row["resize_ms"] - sum(stages.values())
        timings.append(row)
        scores.append(score(record["truth"], predictions))
        rss_peak = max(rss_peak, process.memory_info().rss)
        if number % 100 == 0:
            print(f"  Замеры: {number}/{len(records)}", flush=True)
    return {"timings_ms": timing_summary(timings),
            "serial_fps": len(timings) * 1000 / sum(row["total_ms"] for row in timings),
            "memory_mib": {"gpu_allocated_peak": torch.cuda.max_memory_allocated() / 2**20,
                           "gpu_reserved_peak": torch.cuda.max_memory_reserved() / 2**20,
                           "rss_sampled_peak": rss_peak / 2**20},
            "detections": aggregate(scores)}, timings


def robustness(model, records, size, device):
    results, rows = {}, []
    for variant in VARIANTS:
        scores = []
        for record in records:
            image, truth = perturb(read_image(ROOT / record["image"]), record["truth"], variant)
            predictions, _ = predict(model, image, size, device)
            counts = score(truth, predictions)
            scores.append(counts)
            rows.append({"variant": variant, "image": record["image"], **counts})
        results[variant] = aggregate(scores)
        print(f"  Устойчивость: {variant}, {len(records)} кадров", flush=True)
    return results, rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def preview(record, output):
    canvas = Image.new("RGB", (960, 700), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("DejaVuSans.ttf", 15)
    original = read_image(ROOT / record["image"])
    for index, (variant, title) in enumerate(VARIANTS.items()):
        image, _ = perturb(original, record["truth"], variant)
        x, y = index % 3 * 320, index // 3 * 350
        canvas.paste(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).resize((320, 320)), (x, y + 30))
        draw.text((x + 4, y + 4), title, font=font, fill="black")
    canvas.save(output, quality=92)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/system-evaluation")
    args = parser.parse_args()
    import torch
    import ultralytics

    if not torch.cuda.is_available():
        raise RuntimeError("Для этого протокола требуется GPU; используйте .venv-rocm вне песочницы")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    cache = json.loads((ROOT / "reports/error-analysis/predictions.json").read_text())
    records = cache["images"]
    split = ROOT / "Construction-Site-Safety-PPE-Detection-main/data/splits/test.txt"
    if hashlib.sha256(split.read_bytes()).hexdigest() != cache["settings"]["test_sha256"]:
        raise ValueError("Изменён замороженный тестовый список")
    expected = {str(Path("Construction-Site-Safety-PPE-Detection-main") / line) for line in split.read_text().splitlines()}
    if len(records) != len(expected) or {r["image"] for r in records} != expected:
        raise ValueError("Состав сохранённых эталонов не совпадает с тестовым списком")
    for record in records:
        validate_record(record)
        if read_image(ROOT / record["image"]).shape[:2] != (640, 640):
            raise ValueError("Протокол масштабирования рассчитан на квадратные кадры 640×640")
    sample = sorted(random.Random(42).sample(records, 60), key=lambda record: record["image"])
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"environment": {"python": platform.python_version(), "torch": torch.__version__,
              "ultralytics": ultralytics.__version__, "opencv": cv2.__version__,
              "gpu": torch.cuda.get_device_name(0), "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
              "cpu_threads": torch.get_num_threads(), "opencv_threads": cv2.getNumThreads()},
              "protocol": {"frames": len(records), "warmup_calls": 10, "batch": 1, "precision": "FP32",
              "confidence": 0.25, "nms_iou": 0.7, "match_iou": 0.5, "robustness_frames": 60, "seed": 42,
              "test_sha256": cache["settings"]["test_sha256"]}, "models": {}}
    all_timings, all_robustness = [], []
    for name, folder, size in MODELS:
        weights = ROOT / "runs/ppe" / folder / "weights/best.pt"
        print(f"{name}, imgsz={size}", flush=True)
        model = ultralytics.YOLO(str(weights))
        if [model.names[i] for i in range(10)] != ["Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
                                                "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle"]:
            raise ValueError("Словарь классов модели не совпадает с эталонами")
        summary, timings = evaluate(model, records, size, "0", torch)
        stress, stress_rows = robustness(model, sample, size, "0")
        result["models"][name] = {"weights": str(weights.relative_to(ROOT)),
            "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "imgsz": size,
            **summary, "robustness": stress}
        all_timings.extend({"model": name, **row} for row in timings)
        all_robustness.extend({"model": name, **row} for row in stress_rows)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write_csv(args.output / "timings.csv", all_timings)
    write_csv(args.output / "robustness.csv", all_robustness)
    (args.output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    preview_record = max(sample, key=lambda record: max(
        ((gt["box"][2] - gt["box"][0]) * (gt["box"][3] - gt["box"][1])
         for gt in record["truth"] if gt["class_id"] == 2), default=0))
    preview(preview_record, args.output / "robustness-preview.jpg")
    print(f"Готово: {args.output}")


if __name__ == "__main__":
    main()
