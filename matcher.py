from typing import List, Dict

import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from config import CLIP_MODEL_NAME
from scipy.optimize import linear_sum_assignment  # 匈牙利算法

_clip_model = None
_processor = None

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_clip():
    global _clip_model, _processor
    if _clip_model is None or _processor is None:
        _processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
        _clip_model.eval()
    return _clip_model, _processor

def encode_texts(texts):
    model, processor = get_clip()
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    return feats / feats.norm(dim=-1, keepdim=True)

def encode_images(crops):
    model, processor = get_clip()
    inputs = processor(images=crops, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_image_features(pixel_values=inputs["pixel_values"])
    return feats / feats.norm(dim=-1, keepdim=True)

# def match_entities(entities, crops, boxes):
#     if not entities or not crops:
#         return []
#
#     t_feats = encode_texts([e["name"] for e in entities])
#     v_feats = encode_images(crops)
#     sim = (t_feats @ v_feats.T).cpu().numpy()  # [N, M]
#
#     # 匈牙利算法，确保一对一匹配
#     row_ind, col_ind = linear_sum_assignment(-sim)
#
#     results = []
#     for r, c in zip(row_ind, col_ind):
#         results.append({
#             "name": entities[r]["name"],
#             "label": entities[r]["label"],
#             "bbox": boxes[c],
#             "score": float(sim[r, c])  # 匹配相似度
#         })
#     return results

def match_entities(entities, crops, boxes):
    """
    改进：确保每个检测框都能匹配到一个 name。
    entities: [{'name','label'}]
    crops: 裁剪后的检测框图像
    boxes: 对应检测框的坐标
    """
    if not entities or not crops:
        return []

    # 1. 计算相似度矩阵 [num_entities, num_boxes]
    # t_feats = encode_texts(['a ' + e["name"] + " " +e['label'] for e in entities])
    t_feats = encode_texts([f"{e['name']}: {e['description']}" for e in entities])
    v_feats = encode_images(crops)
    sim = (t_feats @ v_feats.T).cpu().numpy()

    results = []

    # 2. 框驱动：对每个框，选择相似度最高的 name
    for j, bbox in enumerate(boxes):
        i = int(np.argmax(sim[:, j]))   # 找与此框最相似的实体索引
        score = sim[i, j]
        if score < 0.45:
            continue
        results.append({
            "name":  entities[i]["name"],
            "label": entities[i]["label"],
            "bbox":  bbox,
            "score": float(score)
        })

    return results

