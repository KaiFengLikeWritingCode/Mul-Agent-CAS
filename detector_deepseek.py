# detector_api.py
import os
import re
import time
import json
import base64
import requests
from typing import List

# ———— 配置 ————
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise EnvironmentError("请先设置环境变量 REPLICATE_API_TOKEN")

# 一定要用官方的 API 域名
REPLICATE_URL = "https://api.replicate.com/v1/predictions"
HEADERS = {
    "Authorization": f"Token {REPLICATE_API_TOKEN}",
    "Content-Type":  "application/json",
}

# 也可以指定 small、tiny 版：chenxwh/deepseek-vl2:small 等
MODEL_VERSION = "chenxwh/deepseek-vl2:latest"


def detect_boxes(image_path: str, phrase: str, timeout: int = 120) -> List[List[int]]:
    """
    用 DeepSeek-VL2 （Replicate）做零样本检测。
    返回 [[xmin,ymin,xmax,ymax], …]。超时、报错或无命中均返回 []。
    """
    # 1) 读取 & base64 编码
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    # 2) Prompt: grounding 格式
    prm = f"<|grounding|><|ref|>{phrase}<|/ref|>."

    payload = {
        "version": MODEL_VERSION,
        "input": {
            "prompt": prm,
            "image":   f"data:image/jpeg;base64,{image_b64}"
        }
    }

    # 3) 发起预测任务
    create_resp = requests.post(REPLICATE_URL, json=payload, headers=HEADERS)
    create_resp.raise_for_status()
    job = create_resp.json()
    job_id = job.get("id")
    if not job_id:
        return []

    # 4) 轮询状态
    start = time.time()
    status = None
    while time.time() - start < timeout:
        time.sleep(1)
        status_resp = requests.get(f"{REPLICATE_URL}/{job_id}", headers=HEADERS)
        status_resp.raise_for_status()
        status = status_resp.json()
        st = status.get("status")
        if st == "succeeded":
            break
        if st in ("failed", "canceled"):
            return []
    else:
        # 超时
        return []

    # 5) 从 output 列表里抽带 <|det|> 的行
    outputs = status.get("output") or []
    # 倒序找最新的那个
    for entry in reversed(outputs):
        m = re.search(r"<\|det\|>(\[\[.*?\]\])", entry)
        if m:
            boxes = json.loads(m.group(1))
            # 确保转 int
            return [[int(x) for x in box] for box in boxes]

    return []
