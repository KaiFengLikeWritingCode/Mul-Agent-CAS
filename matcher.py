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

def match_entities(entities, crops, boxes):
    """
    多实体 <-> 多检测框 用匈牙利算法匹配
    entities: [{'name','label'}]
    crops: 与 boxes 一一对应
    boxes: [[xmin,ymin,xmax,ymax], ...]
    """
    if not entities or not crops:
        return []

    # 用 name 生成文本特征
    t_feats = encode_texts([e["name"] for e in entities])
    v_feats = encode_images(crops)
    sim = (t_feats @ v_feats.T).cpu().numpy()  # [N, M]

    # 匈牙利算法最大化相似度 → 最小化负相似度
    row_ind, col_ind = linear_sum_assignment(-sim)

    results = []
    for r, c in zip(row_ind, col_ind):
        results.append({
            "name":  entities[r]["name"],
            "label": entities[r]["label"],
            "bbox":  boxes[c],
            "score": float(sim[r, c])
        })

    return results
