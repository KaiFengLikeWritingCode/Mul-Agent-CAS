import argparse
import os
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont

from ner_deepseek import extract_entities          # 你的多模态 NER
from detector      import detect_boxes
from matcher       import match_entities


ALLOWED_LABELS = {"vehicle", "aircraft", "vessel", "weapon"}


# ──────────────────────────── 辅助 ────────────────────────────
def crop_regions(image: Image.Image, boxes: List[List[int]]) -> List[Image.Image]:
    """根据 bbox 列表裁剪图像区域。"""
    return [image.crop((xmin, ymin, xmax, ymax)).convert("RGB") for xmin, ymin, xmax, ymax in boxes]


# ──────────────────────────── 核心流程 ─────────────────────────
def process(image_path: str, caption: str):
    """
    1. NER → entities
    2. detect_boxes + match_entities → 绑定唯一 bbox
    3. 返回原图和带 bbox/score 的结果列表
    """
    img   = Image.open(image_path).convert("RGB")
    ents  = extract_entities(image_path, caption)

    # —— 1) 先分组 —— #
    aircraft_ents: List[Dict] = [e for e in ents if e["label"] == "aircraft"]
    other_ents:     List[Dict] = [e for e in ents if e["label"] in ALLOWED_LABELS and e["label"] != "aircraft"]
    invalid_ents:   List[Dict] = [e for e in ents if e["label"] not in ALLOWED_LABELS]

    results: List[Dict] = []

    # —— 2) aircraft 一次性检测 + 匹配 —— #
    if aircraft_ents:
        ac_boxes = detect_boxes(image_path, "aircraft")          # 只跑一次 Grounding-DINO
        if ac_boxes:
            ac_crops   = crop_regions(img, ac_boxes)
            matched_ac = match_entities(aircraft_ents, ac_crops, ac_boxes, thr=0.5)
            results.extend(matched_ac)
        else:
            # 整张图没检测到飞机
            results.extend(
                {"name": e["name"], "label": e["label"], "bbox": None, "score": None}
                for e in aircraft_ents
            )

    # —— 3) 其余可定位类别：逐实体按 name 检测 —— #
    for ent in other_ents:
        name, label = ent["name"], ent["label"]
        boxes = detect_boxes(image_path, name)        # 用专有名词提高准确率
        if not boxes:
            results.append({"name": name, "label": label, "bbox": None, "score": None})
            continue
        matched = match_entities([ent], crop_regions(img, boxes), boxes, thr=0.5)[0]
        results.append(matched)

    # —— 4) 不可定位类别直接 None —— #
    for ent in invalid_ents:
        results.append({"name": ent["name"], "label": ent["label"], "bbox": None, "score": None})

    return img, results


# ──────────────────────────── 可视化 ──────────────────────────
def draw_and_save(img: Image.Image, results: list, output_path: str):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except IOError:
        font = ImageFont.load_default()

    for res in results:
        if res["bbox"] is None:
            continue
        xmin, ymin, xmax, ymax = res["bbox"]
        label_txt = f'{res["name"]} ({res["label"]}:{res["score"]:.2f})'
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)

        tb = draw.textbbox((xmin, ymin), label_txt, font=font)
        text_w, text_h = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle([xmin, ymin - text_h, xmin + text_w, ymin], fill="red")
        draw.text((xmin, ymin - text_h), label_txt, fill="white", font=font)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"Saved annotated image to {output_path}")


# ──────────────────────────── CLI ────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--text",  required=True, help="Caption / text")
    parser.add_argument("--out",   default="outputs/annotated.jpg")
    args = parser.parse_args()

    image, res = process(args.image, args.text)

    print("\nDetection + Matching Results")
    for r in res:
        print(r)

    draw_and_save(image, res, args.out)
