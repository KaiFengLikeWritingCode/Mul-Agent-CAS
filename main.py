import argparse
import os
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont

from ner_deepseek import extract_entities    # 你的多模态 NER 模块
from detector      import detect_boxes
from matcher       import match_entities

ALLOWED_LABELS = {"vehicle", "aircraft", "vessel", "weapon"}


def crop_regions(image: Image.Image, boxes: List[List[int]]) -> List[Image.Image]:
    """根据 bbox 列表裁剪图像区域。"""
    return [
        image.crop((xmin, ymin, xmax, ymax)).convert("RGB")
        for xmin, ymin, xmax, ymax in boxes
    ]


def process(image_path: str, caption: str):
    """
    1. NER → entities
    2. 按 label 分组，每组一次 detect_boxes + match_entities
    3. invalid_ents 全部 bbox=None
    """
    img  = Image.open(image_path).convert("RGB")
    ents = extract_entities(image_path, caption)

    # 分组
    grouped: Dict[str, List[Dict]] = {lbl: [] for lbl in ALLOWED_LABELS}
    invalid_ents: List[Dict] = []

    for e in ents:
        lbl = e["label"]
        if lbl in ALLOWED_LABELS:
            grouped[lbl].append(e)
        else:
            invalid_ents.append(e)

    results: List[Dict] = []

    # 对每个合法类别一次性检测
    for lbl, ents_of_lbl in grouped.items():
        if not ents_of_lbl:
            continue

        # 1) 检测该类别所有框
        boxes = detect_boxes(image_path, lbl)
        if not boxes:
            # 全部视作未检测到
            for e in ents_of_lbl:
                results.append({
                    "name":  e["name"],
                    "label": e["label"],
                    "bbox":  None,
                    "score": None
                })
            continue

        # 2) 裁剪 & 匹配
        crops   = crop_regions(img, boxes)
        matched = match_entities(ents_of_lbl, crops, boxes, thr=0.1)
        results.extend(matched)

    # invalid_ents 全部 None
    for e in invalid_ents:
        results.append({
            "name":  e["name"],
            "label": e["label"],
            "bbox":  None,
            "score": None
        })

    return img, results


def draw_and_save(img: Image.Image, results: List[Dict], output_path: str):
    """
    在 img 上画出 results 里的 bbox 和标签，然后保存到 output_path。
    results 每项: { "name","label","bbox":[xmin,ymin,xmax,ymax], "score" }
    """
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except IOError:
        font = ImageFont.load_default()

    for res in results:
        bbox = res.get("bbox")
        if bbox is None:
            continue
        xmin, ymin, xmax, ymax = bbox
        label_txt = f'{res["name"]} ({res["label"]}:{res["score"]:.2f})'

        # 画框
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)

        # 画文字背景
        tb = draw.textbbox((xmin, ymin), label_txt, font=font)
        text_w, text_h = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle([xmin, ymin - text_h, xmin + text_w, ymin], fill="red")

        # 写文字
        draw.text((xmin, ymin - text_h), label_txt, fill="white", font=font)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"Saved annotated image to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--text",  required=True, help="Caption / text")
    parser.add_argument("--out",   default="outputs/annotated.jpg")
    args = parser.parse_args()

    img, res = process(args.image, args.text)

    print("Detection + Matching Results:")
    for r in res:
        print(r)

    draw_and_save(img, res, args.out)
