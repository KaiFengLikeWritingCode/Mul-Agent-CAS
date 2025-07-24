# matcher.py

import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from config import CLIP_MODEL_NAME

# 全局变量，确保只加载一次
_clip_model = None
_processor = None

# 选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_clip():
    """
    懒加载 CLIPProcessor 和 CLIPModel，返回 (model, processor)
    """
    global _clip_model, _processor
    if _clip_model is None or _processor is None:
        _processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
        _clip_model.eval()
    return _clip_model, _processor

def encode_texts(texts):
    """
    texts: List[str]
    返回: Tensor[N, D] 已归一化的文本特征
    """
    model, processor = get_clip()
    # tokenizer part
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    # 归一化
    return feats / feats.norm(dim=-1, keepdim=True)

def encode_images(crops):
    """
    crops: List[PIL.Image]
    返回: Tensor[M, D] 已归一化的图像特征
    """
    model, processor = get_clip()
    # image preprocessing
    inputs = processor(images=crops, return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"]  # shape [M, C, H, W]
    with torch.no_grad():
        feats = model.get_image_features(pixel_values=pixel_values)
    # 归一化
    return feats / feats.norm(dim=-1, keepdim=True)

# def match_entities(entities, crops, boxes):
#     """
#     entities: List[{"name": str, "label": str}]
#     crops: List[PIL.Image]     # 与 boxes 一一对应
#     boxes: List[List[int]]     # [[xmin,ymin,xmax,ymax], ...]
#     返回: List[{"label","label","bbox","score"}]
#     """
#     # 文本和图像特征
#     t_feats = encode_texts([e["name"] for e in entities])  # [N, D]
#     v_feats = encode_images(crops)                           # [M, D]
#
#     # 计算余弦相似度矩阵 [N, M]
#     sim = (t_feats @ v_feats.T).cpu().numpy()
#
#     results = []
#     for i, ent in enumerate(entities):
#         # top-1 匹配
#         j = int(np.argmax(sim[i]))
#         results.append({
#             "name": ent["name"],
#             "label": ent["label"],
#             "bbox": boxes[j],
#             "score": float(sim[i, j])
#         })
#     return results
from scipy.optimize import linear_sum_assignment
from vqa_score import score_pair        # 你已有的 BLIP-2/VL2 评分函数，返回 0~1
SCORE_NEG = 1e3     # 匹配不到时的大成本

def match_entities(ents, crops, boxes, thr: float = 0.5):
    """
    给定实体列表、crop 图和 boxes，返回与 ents 等长的结果 list。
    每个实体最多分配一个 box；同一 box 也只能给一个实体。
    """
    if not boxes:
        return [{"name":e["name"], "label":e["label"], "bbox":None, "score":None}
                for e in ents]

    # 1) 计算 cost 矩阵 = 1 - score
    n, m = len(ents), len(boxes)
    cost = np.full((n, m), SCORE_NEG, dtype=np.float32)

    for i, ent in enumerate(ents):
        for j, crop in enumerate(crops):
            s = score_pair(crop, ent["name"])     # BLIP-2 VQA 相似度分
            cost[i, j] = 1.0 - s                  # 分数越高成本越小

    # 2) 匈牙利算法求最小成本匹配
    row_idx, col_idx = linear_sum_assignment(cost)

    # 3) 组装输出，未匹配或低于阈值 → bbox=None
    out = []
    used_boxes = set(col_idx)
    for i, ent in enumerate(ents):
        if i in row_idx:
            j = col_idx[np.where(row_idx == i)[0][0]]
            score = 1.0 - cost[i, j]
            if score >= thr:          # 达标，赋框
                out.append({
                    "name": ent["name"],
                    "label": ent["label"],
                    "bbox": [int(x) for x in boxes[j]],
                    "score": float(score)
                })
                continue
        # 未匹配或分数低
        out.append({"name": ent["name"], "label": ent["label"],
                    "bbox": None, "score": None})
    return out


