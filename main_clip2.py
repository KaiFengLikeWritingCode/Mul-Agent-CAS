# main.py

import argparse
from PIL import Image, ImageDraw, ImageFont
from ner_deepseek import extract_entities
# from ner_clip import extract_entities
from detector import detect_boxes
from matcher import match_entities
from collections import defaultdict
import os

from generate_description import generate_description_with_llm

def crop_regions(image: Image.Image, boxes):
    """
    根据给定的 boxes 裁剪区域并返回图像列表。
    boxes: List[List[xmin,ymin,xmax,ymax]] 或 [(bbox, score)] 形式
    """
    crops = []
    W, H = image.size

    for b in boxes:
        # 兼容 [(bbox, score)] 或 [xmin,ymin,xmax,ymax]
        if isinstance(b, (tuple, list)) and len(b) == 2 and isinstance(b[0], (list, tuple)):
            xmin, ymin, xmax, ymax = b[0]
        else:
            xmin, ymin, xmax, ymax = b

        # 防止越界
        xmin = max(0, int(xmin))
        ymin = max(0, int(ymin))
        xmax = min(W, int(xmax))
        ymax = min(H, int(ymax))

        # 忽略无效框
        if xmax <= xmin or ymax <= ymin:
            continue

        crop = image.crop((xmin, ymin, xmax, ymax)).convert("RGB")
        crops.append(crop)

    return crops


# def process(image_path, text):
#     img = Image.open(image_path).convert("RGB")
#     # ents = extract_entities(text)
#     ents = extract_entities(image_path, text)
#     all_results = []
#     for ent in ents:
#         # 用实体 name 做零样本检测
#         boxes = detect_boxes(image_path, ent["name"])
#         if not boxes:
#             continue
#         # 裁剪并匹配
#         crops = crop_regions(img, boxes)
#         matched = match_entities([ent], crops, boxes)
#         all_results.extend(matched)
#     return img, all_results
def process(image_path, text):
    img = Image.open(image_path).convert("RGB")
    ents = extract_entities(image_path, text)  # [{'name':..., 'label':...}]


    for e in ents:
        e["description"] = generate_description_with_llm(e["name"], text)

    print(ents)
    all_results = []

    # 按 label 分组
    from collections import defaultdict
    label_groups = defaultdict(list)

    print("\n=== Label Groups ===")
    for label, group in label_groups.items():
        print(f"Label: {label}")
        for entity in group:
            print(f"  - {entity['name']}")
    for e in ents:
        label_groups[e["label"]].append(e)

    # 每个 label 单独检测
    for label, group in label_groups.items():
        if label.lower() == "location":  # 🔥 跳过 location
            print(f"Skipping detection for label: {label}")
            continue
        print(f"Detecting label: {label}")
        det_results = detect_boxes(image_path, label)  # [(bbox, score), ...]

        print("=== Detection Results ===")
        print(f"Found {len(det_results)} boxes for label '{label}':")
        for i, (bbox, score) in enumerate(det_results, 1):
            print(f"  Box {i}: bbox={bbox}, score={score:.2f}")
        if not det_results:
            print(f"No boxes detected for {label}")
            continue

        boxes, scores = zip(*det_results)
        crops = crop_regions(img, boxes)

        # 多实体-多检测框匹配
        matched = match_entities(group, crops, boxes)
        # 把score补回去
        for m in matched:
            idx = boxes.index(m["bbox"])
            m["score"] = scores[idx]
        all_results.extend(matched)

    return img, all_results


def draw_and_save(img: Image.Image, results: list, output_path: str):
    """
    在 img 上画出 results 里的 bbox 和 label，然后保存到 output_path
    results 每项: { "name","label","bbox":[xmin,ymin,xmax,ymax], "score" }
    """
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except IOError:
        font = ImageFont.load_default()

    for res in results:
        xmin, ymin, xmax, ymax = res["bbox"]
        label = f'{res["name"]} ({res["label"]}:{res["score"]:.2f})'

        # 画检测框
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)

        # 计算文字框尺寸
        # textbbox 返回 (x0, y0, x1, y1)
        tb = draw.textbbox((xmin, ymin), label, font=font)
        text_width  = tb[2] - tb[0]
        text_height = tb[3] - tb[1]

        # 在框顶上留出高度，画背景矩形
        text_bg = [xmin, ymin - text_height, xmin + text_width, ymin]
        draw.rectangle(text_bg, fill="red")

        # 画文字
        draw.text((xmin, ymin - text_height), label, fill="white", font=font)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"Saved annotated image to {output_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True, help="Path to input image")
    p.add_argument("--text",  type=str, required=True, help="Input text for NER")
    p.add_argument("--out",   type=str, default="outputs/annotated.jpg",
                     help="Path to save annotated image")
    args = p.parse_args()

    img, results = process(args.image, args.text)

    print("Detection+Matching Results:")
    for r in results:
        print(r)

    draw_and_save(img, results, args.out)
