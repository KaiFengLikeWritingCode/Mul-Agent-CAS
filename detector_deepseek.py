# detector_api.py
import os
import re
import time
import json
import base64
import requests
from typing import List

# Replicate API 配置
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise EnvironmentError("请设置环境变量 REPLICATE_API_TOKEN")
HEADERS = {
    "Authorization": f"Token {REPLICATE_API_TOKEN}",
    "Content-Type": "application/json",
}
REPLICATE_URL     = "https://www.chataiapi.com/v1"
MODEL_VERSION     = "deepseek-vl2"  # 或指定 small/tiny 版

def detect_boxes(image_path: str, phrase: str, timeout: int = 120) -> List[List[int]]:
    """
    用 DeepSeek-VL2 模型做零样本目标检测，返回 [[xmin,ymin,xmax,ymax], …]。
    若超时、报错或无框，均返回 []。
    （模型 prompt 用 <|grounding|> + <|ref|>…<|/ref|> 格式触发返框）
    """
    # 1) 读图 + base64 编码
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    # 2) 构造 prompt
    prm = f"<|grounding|><|ref|>{phrase}<|/ref|>."
    payload = {
        "version": MODEL_VERSION,
        "input": {
            "prompt": prm,
            "image":   f"data:image/jpeg;base64,{b64}"
        }
    }

    # 3) 创建预测任务
    resp = requests.post(REPLICATE_URL, json=payload, headers=HEADERS).json()
    pred_id = resp.get("id")
    if not pred_id:
        return []

    # 4) 轮询直到完成或超时
    start = time.time()
    while True:
        time.sleep(1)
        status = requests.get(f"{REPLICATE_URL}/{pred_id}", headers=HEADERS).json()
        if status.get("status") == "succeeded":
            break
        if status.get("status") in ("failed",):
            return []
        if time.time() - start > timeout:
            print("-----超时----")
            return []

    # 5) 解析 output 列表，找到含 <|det|> 的那一行
    outputs = status.get("output", [])
    print("-----outputs----")
    print(outputs)
    for raw in reversed(outputs):
        m = re.search(r"<\|det\|>(\[\[.*?\]\])", raw)
        if m:
            # 可能是 [[x1,y1,x2,y2], …]，也可能多级嵌套，取第一维
            boxes = json.loads(m.group(1))
            # 强制转 int
            return [[int(c) for c in box] for box in boxes]
    return []

import argparse
from PIL import Image, ImageDraw
def draw_boxes_on_image(image_path: str, boxes, output_path: str):
    """
    简单地把 detect_boxes 输出的 rectangles 画到原图上，并保存到 output_path。
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"Annotated image saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test detector_api.detect_boxes() with an image and a phrase"
    )
    parser.add_argument(
        "--image", "-i", required=True,default="./datasets/sample_image/15.jpg",
        help="Path to the input image file"
    )
    parser.add_argument(
        "--phrase", "-p", required=True,default="Eurofighter",
        help="Text phrase to ground (e.g. 'Eurofighter')"
    )
    parser.add_argument(
        "--out", "-o", default="./outputs/test_annotated_15.jpg",
        help="Where to save the annotated image (optional)"
    )
    args = parser.parse_args()

    print(f"Running detect_boxes on image: {args.image!r} with phrase: {args.phrase!r}")
    boxes = detect_boxes(args.image, args.phrase)
    if not boxes:
        print("No boxes detected.")
    else:
        print("Detected boxes:")
        for idx, box in enumerate(boxes):
            print(f"  [{idx}] {box}")

    # 可视化
    draw_boxes_on_image(args.image, boxes, args.out)

if __name__ == "__main__":
    main()



# python detector_deepseek.py     --image ./datasets/sample_image/15.jpg     --phrase "aircraft"     --out ./outputs/test_annotated_15.jpg