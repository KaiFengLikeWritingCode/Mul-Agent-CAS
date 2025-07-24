import os

# LLM NER 模型
# LLM_MODEL_NAME = "Salesforce/blip2-flan-t5-xl"

LLM_MODEL_NAME = "google/flan-t5-base"
from pathlib import Path

# GroundingDINO 配置
# GDINO_CONFIG = os.path.expanduser("models/GroundingDINO/config.py")
# GDINO_CHECKPOINT = os.path.expanduser("models/GroundingDINO/checkpoint.pth")
ROOT = Path(__file__).resolve().parent

# GroundingDINO
GDINO_CONFIG     = str(ROOT / "models" / "GroundingDINO_SwinT_OGC.py")
GDINO_CHECKPOINT = str(ROOT / "models" / "groundingdino_swint_ogc.pth")
GDINO_BOX_THRESHOLD = 0.3

# CLIP
# CLIP_MODEL_NAME = "ViT-B/32"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

BLIP2_MODEL_NAME = "Salesforce/blip2-opt-2.7b"
# BLIP2_MODEL_NAME = "Salesforce/blip2-flan-t5-xl"


# 允许的实体类别
ENTITY_TYPES = ["vehicle", "aircraft", "vessel", "weapon", "location", "other"]