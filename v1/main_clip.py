# main.py

import argparse
from PIL import Image, ImageDraw, ImageFont
from ner_deepseek import extract_entities
# from ner_clip import extract_entities
from detector import detect_boxes
from matcher import match_entities
import os

def crop_regions(image: Image.Image, boxes):
    crops = []
    for xmin, ymin, xmax, ymax in boxes:
        crops.append(image.crop((xmin, ymin, xmax, ymax)).convert("RGB"))
    return crops

def process(image_path, text):
    img = Image.open(image_path).convert("RGB")
    # ents = extract_entities(text)
    ents = extract_entities(image_path, text)
    all_results = []
    for ent in ents:
        # 用实体 name 做零样本检测
        boxes = detect_boxes(image_path, ent["name"])
        if not boxes:
            continue
        # 裁剪并匹配
        crops = crop_regions(img, boxes)
        matched = match_entities([ent], crops, boxes)
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
