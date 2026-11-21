"""Автоматически собрать шесть галерей из сохранённых предсказаний E2.

Запуск: python scripts/visualize_ppe_errors.py
Вход: reports/error-analysis/predictions.json (эталоны и ответы модели).
Выход: шесть папок с изображениями, HTML-галереи и selection.json.
"""

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {
    "high_confidence_fp": "1. Ложные срабатывания с высокой уверенностью",
    "critical_misses": "2. Пропуски критичного класса — без каски",
    "low_iou": "3. Сопоставленные рамки с низким IoU",
    "class_confusion": "4. Путаница похожих классов",
    "small_occluded": "5. Малые и потенциально перекрытые объекты",
    "annotation_suspects": "6. Предположительные ошибки разметки",
}
RULES = {
    "high_confidence_fp": "Нет сопоставленного эталона, уверенность ≥ 0.75.",
    "critical_misses": "Для эталона NO-Hardhat нет ответа того же класса с IoU ≥ 0.5.",
    "low_iou": "Класс совпал, но 0.5 ≤ IoU < 0.75: рамка локализована неточно.",
    "class_confusion": "IoU ≥ 0.5 у несопоставленных рамок похожих, но разных классов.",
    "small_occluded": "Площадь < 0.25% кадра или перекрытие рамкой того же класса ≥ 50%; есть пропуск или низкий IoU. Перекрытие требует ручной проверки.",
    "annotation_suspects": "Уверенность ≥ 0.85, IoU со всеми эталонами < 0.1. Это подозрение, а не доказанная ошибка разметки.",
}
CLASS_NAMES = ("каска", "маска", "без каски", "без маски", "без жилета",
               "человек", "конус", "жилет", "техника", "транспорт")
SIMILAR_PAIRS = ({0, 2}, {1, 3}, {4, 7}, {4, 5}, {5, 7}, {8, 9})


def intersection(left, right):
    return max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def iou(left, right):
    overlap = intersection(left, right)
    union = area(left) + area(right) - overlap
    return overlap / union if union > 0 else 0.0


def match_boxes(truth, predictions):
    """Сопоставить рамки одного класса по убыванию IoU, каждую не более одного раза."""
    candidates = sorted(
        ((iou(gt["box"], pred["box"]), g, p)
         for g, gt in enumerate(truth) for p, pred in enumerate(predictions)
         if gt["class_id"] == pred["class_id"]), reverse=True,
    )
    matched_gt, matched_pred, matches = set(), set(), []
    for overlap, g, p in candidates:
        if overlap >= 0.5 and g not in matched_gt and p not in matched_pred:
            matches.append((g, p, overlap))
            matched_gt.add(g)
            matched_pred.add(p)
    return matches, set(range(len(truth))) - matched_gt, set(range(len(predictions))) - matched_pred


def classify_errors(truth, predictions):
    matches, missed, extra = match_boxes(truth, predictions)
    events = []

    def add(category, score, gt=None, pred=None, **details):
        events.append({"category": category, "score": score, "gt": gt, "pred": pred, **details})

    # 1 и 6: уверенный ответ без эталона; отдельно — подозрение на пропущенную метку.
    for p in sorted(extra):
        pred = predictions[p]
        if pred["confidence"] >= 0.75:
            add("high_confidence_fp", pred["confidence"], pred=p)
        max_overlap = max((iou(pred["box"], gt["box"]) for gt in truth), default=0)
        if pred["confidence"] >= 0.85 and max_overlap < 0.1:
            add("annotation_suspects", pred["confidence"], pred=p, max_iou=max_overlap)
    # 2: пропуск критичного класса «без каски».
    for g in sorted(missed):
        if truth[g]["class_id"] == 2:
            add("critical_misses", area(truth[g]["box"]), gt=g)
    # 3: класс совпал, но рамки пересеклись недостаточно точно.
    for g, p, overlap in matches:
        if overlap < 0.75:
            add("low_iou", 1 - overlap, gt=g, pred=p, iou=overlap)
    # 4: ищем похожие классы только среди ещё не сопоставленных рамок.
    confusion_pairs = sorted(
        ((iou(truth[g]["box"], predictions[p]["box"]), g, p)
         for g in missed for p in extra
         if {truth[g]["class_id"], predictions[p]["class_id"]} in SIMILAR_PAIRS), reverse=True,
    )
    used_gt, used_pred = set(), set()
    for overlap, g, p in confusion_pairs:
        if overlap >= 0.5 and g not in used_gt and p not in used_pred:
            add("class_confusion", overlap, gt=g, pred=p, iou=overlap)
            used_gt.add(g)
            used_pred.add(p)
    # 5: ошибки на малых объектах и при перекрытии рамок одного класса.
    matched = {g: (p, overlap) for g, p, overlap in matches}
    for g, gt in enumerate(truth):
        size = area(gt["box"])
        # Вложенные рамки человека и СИЗ не считаем признаком перекрытия.
        overlap = max((intersection(gt["box"], other["box"]) / size
                       for j, other in enumerate(truth)
                       if j != g and other["class_id"] == gt["class_id"]), default=0)
        if (size < 0.0025 or overlap >= 0.5) and (g in missed or matched[g][1] < 0.75):
            add("small_occluded", 1 - min(size, 1), gt=g,
                pred=matched[g][0] if g in matched else None,
                area=size, same_class_overlap=overlap)
    return events


def validate_record(record):
    path = ROOT / record["image"]
    if not path.resolve().is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"Недоступное изображение: {path}")
    for key in ("truth", "predictions"):
        for item in record[key]:
            cls, box = item["class_id"], item["box"]
            if (not isinstance(cls, int) or cls not in range(len(CLASS_NAMES))
                    or len(box) != 4 or not all(math.isfinite(x) and 0 <= x <= 1 for x in box)
                    or box[0] >= box[2] or box[1] >= box[3]):
                raise ValueError(f"Неверная рамка в {path}: {item}")
            if key == "predictions" and not 0 <= item["confidence"] <= 1:
                raise ValueError(f"Неверная уверенность в {path}: {item}")


def caption(record, event):
    gt = record["truth"][event["gt"]] if event["gt"] is not None else None
    pred = record["predictions"][event["pred"]] if event["pred"] is not None else None
    text = f"Эталон: {CLASS_NAMES[gt['class_id']]}" if gt else "Эталон: нет сопоставленной рамки"
    text += (f". Ответ: {CLASS_NAMES[pred['class_id']]}, уверенность {pred['confidence']:.3f}"
             if pred else ". Ответ: подходящая рамка не найдена")
    if gt and pred:
        text += f". IoU = {iou(gt['box'], pred['box']):.3f}"
    elif "max_iou" in event:
        text += f". Максимальный IoU с эталонами = {event['max_iou']:.3f}"
    if "area" in event:
        text += f". Площадь = {event['area'] * 100:.4f}%, перекрытие = {event['same_class_overlap'] * 100:.1f}%"
    return text + "."


def render_example(record, event):
    """Слева эталон, справа ответ; внизу — увеличенный фрагмент без подписей поверх объекта."""
    with Image.open(ROOT / record["image"]) as source:
        original = source.convert("RGB").resize((640, 640))
    font = ImageFont.truetype("DejaVuSans.ttf", 16)
    canvas = Image.new("RGB", (1280, 1000), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), CATEGORIES[event["category"]], font=font, fill="black")
    draw.text((12, 35), "\n".join(textwrap.wrap(caption(record, event), 115)), font=font, fill="black")
    draw.text((12, 93), Path(record["image"]).name, font=font, fill="black")
    focus = (record["truth"][event["gt"]] if event["gt"] is not None
             else record["predictions"][event["pred"]])["box"]
    cx, cy = (focus[0] + focus[2]) * 320, (focus[1] + focus[3]) * 320
    span = max(64, (focus[2] - focus[0]) * 960, (focus[3] - focus[1]) * 960)
    crop = (max(0, int(cx - span / 2)), max(0, int(cy - span / 2)),
            min(640, int(cx + span / 2)), min(640, int(cy + span / 2)))
    for offset, key, index, title, color in (
        (0, "truth", event["gt"], "Эталонная рамка — зелёная", "green"),
        (640, "predictions", event["pred"], "Рамка модели — красная", "red"),
    ):
        panel = original.copy()
        if index is not None:
            box = record[key][index]["box"]
            ImageDraw.Draw(panel).rectangle([x * 640 for x in box], outline=color, width=3)
        draw.text((offset + 12, 117), title, font=font, fill=color)
        canvas.paste(panel, (offset, 140))
        draw.text((offset + 12, 790), "Увеличенный фрагмент", font=font, fill="black")
        zoom = panel.crop(crop)
        scale = 180 / max(zoom.size)
        zoom = zoom.resize((max(1, int(zoom.width * scale)), max(1, int(zoom.height * scale))),
                           Image.Resampling.NEAREST)
        canvas.paste(zoom, (offset + 12, 820))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "reports/error-analysis/predictions.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/error-analysis")
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    split = ROOT / "Construction-Site-Safety-PPE-Detection-main/data/splits/test.txt"
    if hashlib.sha256(split.read_bytes()).hexdigest() != data["settings"]["test_sha256"]:
        raise ValueError("Тестовый список отличается от сохранённых предсказаний")
    records = {record["image"]: record for record in data["images"]}
    if not records or len(records) != len(data["images"]):
        raise ValueError("Предсказания пусты или содержат повторяющиеся изображения")
    candidates = {key: [] for key in CATEGORIES}
    for record in records.values():
        validate_record(record)
        for event in classify_errors(record["truth"], record["predictions"]):
            candidates[event["category"]].append({"image": record["image"], **event})
    selected, used_images, links = [], set(), []
    for category, title in CATEGORIES.items():
        folder = args.output / category
        folder.mkdir(parents=True, exist_ok=True)
        ranked = sorted(candidates[category], key=lambda e: (-e["score"], e["image"]))
        ranked.sort(key=lambda e: e["image"] in used_images)
        if category == "small_occluded":
            overlapping = sorted((e for e in ranked if e["same_class_overlap"] >= 0.5
                and e["area"] >= 0.0025 and records[e["image"]]["truth"][e["gt"]]["class_id"] == 5),
                key=lambda e: -e["area"])
            ranked = overlapping[:1] + ranked
        examples, category_images = [], set()
        for event in ranked:
            if event["image"] in category_images:
                continue
            category_images.add(event["image"])
            used_images.add(event["image"])
            record = records[event["image"]]
            filename = f"{len(examples) + 1:02d}.jpg"
            render_example(record, event).save(folder / filename, quality=92)
            selected.append({**event, "gallery": f"{category}/{filename}"})
            examples.append(f'<figure><img src="{filename}" alt="{html.escape(title)}">'
                f'<figcaption>{html.escape(Path(event["image"]).name)}<br>'
                f'{html.escape(caption(record, event))}</figcaption></figure>')
            if len(examples) == 4:
                break
        (folder / "index.html").write_text('<!doctype html><html lang="ru"><meta charset="utf-8">'
            f'<title>{title}</title><style>body{{font:18px sans-serif;margin:24px}}'
            'img{max-width:100%}figcaption{overflow-wrap:anywhere}</style>'
            f'<a href="../index.html">Все галереи</a><h1>{title}</h1><p>{RULES[category]}</p>'
            '<p>Слева эталон, справа ответ модели. Показаны только рассматриваемые рамки.</p>'
            f'<p>Кандидатов: {len(candidates[category])}; показано кадров: {len(examples)}.</p>'
            + ''.join(examples) + '</html>')
        links.append(f'<li><a href="{category}/index.html">{title}</a><p>{RULES[category]}</p></li>')
    (args.output / "index.html").write_text('<!doctype html><html lang="ru"><meta charset="utf-8">'
        '<title>Ошибки E2</title><h1>Шесть галерей ошибок E2</h1>'
        '<p>Зелёный — эталон, красный — модель. IoU — пересечение рамок / их объединение.</p>'
        '<p>По одному событию на кадр, до четырёх кадров на категорию. Категории могут пересекаться.</p>'
        '<ul>' + ''.join(links) + '</ul></html>')
    (args.output / "selection.json").write_text(json.dumps({"settings": data["settings"],
        "candidate_counts": {key: len(value) for key, value in candidates.items()},
        "examples": selected}, ensure_ascii=False, indent=2) + "\n")
    print(f"Готово: {len(records)} кадров, {len(selected)} примеров. Откройте {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
