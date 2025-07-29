# main.py

import argparse
import os
from PIL import Image, ImageDraw, ImageFont
from ner_deepseek import extract_entities            # 注意：改为你新的多模态 ner 模块
from detector import detect_boxes
from matcher import match_entities

def crop_regions(image: Image.Image, boxes):
    """根据 bbox 列表裁剪图像区域。"""
    crops = []
    for xmin, ymin, xmax, ymax in boxes:
        crops.append(image.crop((xmin, ymin, xmax, ymax)).convert("RGB"))
    return crops

ALLOWED_LABELS = {"vehicle", "aircraft", "vessel", "weapon"}
def process(image_path, text):
    img = Image.open(image_path).convert("RGB")
    ents = extract_entities(image_path, text)
    results = []
    for ent in ents:
        name, label = ent["name"], ent["label"]
        if label not in ALLOWED_LABELS:
            results.append({"name":name, "label":label, "bbox":None, "score":None})
            continue
        # boxes = detect_boxes(image_path, name)
        boxes = detect_boxes(image_path, label)
        if not boxes:
            results.append({"name":name, "label":label, "bbox":None, "score":None})
            continue
        crops = crop_regions(img, boxes)
        matched = match_entities([ent], crops, boxes)[0]
        results.append(matched)
    return img, results

def draw_and_save(img: Image.Image, results: list, output_path: str):
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
            # 无 bbox，就跳过绘制
            continue
        xmin, ymin, xmax, ymax = bbox
        label_text = f'{res["name"]} ({res["label"]}:{res["score"]:.2f})'

        # 1) 画框
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)

        # 2) 计算文字区域并画背景
        tb = draw.textbbox((xmin, ymin), label_text, font=font)
        text_w = tb[2] - tb[0]
        text_h = tb[3] - tb[1]
        draw.rectangle([xmin, ymin - text_h, xmin + text_w, ymin], fill="red")

        # 3) 写文字
        draw.text((xmin, ymin - text_h), label_text, fill="white", font=font)

    # 确保目录存在，保存文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"Saved annotated image to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--text",  type=str, required=True, help="Caption or text for entity extraction")
    parser.add_argument("--out",   type=str, default="outputs/annotated.jpg",
                        help="Where to save the annotated image")
    args = parser.parse_args()

    # 1) 处理得到原图和识别结果
    img, results = process(args.image, args.text)

    # 2) 在控制台打印结果
    print("Detection + Matching Results:")
    for r in results:
        print(r)

    # 3) 可视化并保存
    draw_and_save(img, results, args.out)
