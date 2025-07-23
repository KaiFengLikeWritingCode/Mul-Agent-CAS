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

def match_entities(entities, crops, boxes):
    """
    entities: List[{"name": str, "label": str}]
    crops: List[PIL.Image]     # 与 boxes 一一对应
    boxes: List[List[int]]     # [[xmin,ymin,xmax,ymax], ...]
    返回: List[{"label","label","bbox","score"}]
    """
    # 文本和图像特征
    t_feats = encode_texts([e["name"] for e in entities])  # [N, D]
    v_feats = encode_images(crops)                           # [M, D]

    # 计算余弦相似度矩阵 [N, M]
    sim = (t_feats @ v_feats.T).cpu().numpy()

    results = []
    for i, ent in enumerate(entities):
        # top-1 匹配
        j = int(np.argmax(sim[i]))
        results.append({
            "name": ent["name"],
            "label": ent["label"],
            "bbox": boxes[j],
            "score": float(sim[i, j])
        })
    return results
