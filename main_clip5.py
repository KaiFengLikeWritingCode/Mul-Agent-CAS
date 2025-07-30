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


import json
from openai import OpenAI

# 初始化通义千问客户端（兼容 OpenAI SDK 调用）
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 设置环境变量 DASHSCOPE_API_KEY
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def evaluate_detections_with_llm_full(image_path: str, detections: list, context_text: str):
    """
    使用阿里云通义千问 Qwen Plus 对整体图像和检测结果进行评估与清洗（使用URL形式加载图像）。
    """
    # 从 image_path 提取 idx (文件名去扩展名)
    idx = os.path.splitext(os.path.basename(image_path))[0]
    image_url = f"https://raw.githubusercontent.com/KaiFengLikeWritingCode/Mul-Agent-CAS/refs/heads/main/datasets_0/sample_image/{idx}.jpg"

    detections_text = "\n".join(
        [f"- Name: {d['name']}, Label: {d['label']}, Score: {d['score']:.2f}, BBox: {d['bbox']}" for d in detections]
    )

    user_prompt = f"""
You are a military image analysis expert.

Context Text:
{context_text}

Detections:
{detections_text}

Instructions:
1. Visually check the image and verify which detections are correct.
2. Return ONLY a JSON list of valid detections in this format:
[
  {{"name": "...", "label": "...", "bbox": [xmin,ymin,xmax,ymax], "score": ...}}
]
If none are valid, return [].
    """

    # 调用通义千问 Qwen Plus
    completion = client.chat.completions.create(
        model="qwen-plus",  # 支持文本推理
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that evaluates object detections in images."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        temperature=0,
        max_tokens=512
    )

    content = completion.choices[0].message.content.strip()

    if content.startswith("```"):
        content = content.strip("`")  # 去掉所有反引号
        # 去掉可能的json关键字
        content = content.replace("json", "", 1).strip()

    try:
        cleaned = json.loads(content)
        if isinstance(cleaned, list):
            print(f"[QWEN EVAL] Cleaned {len(detections)} → {len(cleaned)} valid detections")
            return cleaned
        else:
            print("[WARNING] Qwen did not return list. Keeping original detections.")
            return detections
    except Exception as e:
        print(f"[ERROR] JSON parse failed after cleanup: {content}\nError: {e}")
        return detections


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


def iou(box1, box2):
    """计算两个框的IoU"""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    inter_x1 = max(x1, x1g)
    inter_y1 = max(y1, y1g)
    inter_x2 = min(x2, x2g)
    inter_y2 = min(y2, y2g)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)  # 修复这里
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def merge_boxes(boxes, scores, iou_threshold=0.5):
    """
    合并重叠率高的框，确保最终保留的框之间完全不重叠
    对有重叠的框，保留分数最高的那个
    """
    # 按分数排序
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    final_boxes, final_scores = [], []

    while sorted_indices:
        current = sorted_indices.pop(0)
        current_box, current_score = boxes[current], scores[current]

        # 保留当前框
        final_boxes.append(current_box)
        final_scores.append(current_score)

        # 去掉与当前框重叠超过阈值的框
        sorted_indices = [
            i for i in sorted_indices
            if iou(current_box, boxes[i]) < iou_threshold
        ]

    return final_boxes, final_scores


def process(image_path, text):
    img = Image.open(image_path).convert("RGB")
    ents = extract_entities(image_path, text)  # [{'name':..., 'label':...}]

    # 生成每个实体的描述
    for e in ents:
        e["description"] = generate_description_with_llm(e["name"], text)

    print("=====Extracted entities with descriptions=====")
    for e in ents:
        print(e)
    print("\n")

    # 按label分组实体
    grouped_entities = defaultdict(list)
    for e in ents:
        grouped_entities[e["label"].lower()].append(e)

    all_boxes, all_scores, all_entities = [], [], []

    # 遍历分组进行检测
    for label, entity_list in grouped_entities.items():
        if label == "location":
            # 跳过地理位置实体检测
            for e in entity_list:
                print(f"Skipping detection for location entity: {e['name']}")
            continue

        if label == "aircraft":
            # 对飞机实体只用label进行检测
            detect_prompt = label  # 直接使用类别作为检测提示词
            print(f"Detecting aircrafts with prompt: {detect_prompt}")
            det_results = detect_boxes(image_path, detect_prompt)

            # 所有检测到的框统一分配给当前label下的所有实体（后续通过matcher区分）
            for bbox, score in det_results:
                all_boxes.append(bbox)
                all_scores.append(score)
                # 这里保留实体列表，每个检测框后续通过 match_entities 做语义匹配
                for e in entity_list:
                    all_entities.append(e)

        else:
            # 其他实体：用name+description进行检测
            for e in entity_list:
                detect_prompt = f"{e['name']} : {e['description']}"
                print(f"Detecting entity: {detect_prompt}")
                det_results = detect_boxes(image_path, detect_prompt)
                for bbox, score in det_results:
                    all_boxes.append(bbox)
                    all_scores.append(score)
                    all_entities.append(e)

    # 合并重叠的框
    merged_boxes, merged_scores = merge_boxes(all_boxes, all_scores, iou_threshold=0.5)
    print(f"Merged {len(all_boxes)} boxes into {len(merged_boxes)} boxes")

    print("\n========== Detection Summary ==========")
    for i in range(len(merged_boxes)):
        print(f"     BBox : {merged_boxes[i]}")
        print(f"     Score: {merged_scores[i]:.4f}\n")

    # 裁剪合并后的区域
    crops = crop_regions(img, merged_boxes)

    # 匹配：将去重框与实体按语义匹配
    matched = match_entities(ents, crops, merged_boxes)
    for m in matched:
        idx = merged_boxes.index(m["bbox"])
        m["score"] = merged_scores[idx]

    # === LLM整体评估与清洗 ===
    print("\n=== Evaluating all detections with LLM ===")
    matched_filtered = evaluate_detections_with_llm_full(image_path, matched, text)

    # 将 label 为 location 的实体补充进 matched_filtered，bbox 为 null
    location_entities = grouped_entities.get("location", [])
    for loc in location_entities:
        matched_filtered.append({
            "name": loc["name"],
            "label": loc["label"],
            "bbox": None,       # 地理位置实体没有对应框
            "score": None       # 分数可设为 None 或省略
        })

    # 找出所有未匹配到 bbox 的其他实体（非 location）
    matched_names = {m["name"] for m in matched_filtered}  # 已在结果中的实体名
    for ent in ents:
        if ent["name"] not in matched_names:
            print(f"Entity '{ent['name']}' has no matched bbox, adding with bbox=None.")
            matched_filtered.append({
                "name": ent["name"],
                "label": ent["label"],
                "bbox": None,
                "score": None
            })

    # return img, matched
    return img, matched_filtered



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
        # 跳过没有定位框的实体（例如 location）
        if res["bbox"] is None:
            continue
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
