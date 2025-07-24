# vqa_score.py
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch, re
from PIL import Image, ImageDraw
device = "cuda" if torch.cuda.is_available() else "cpu"
from config import BLIP2_MODEL_NAME, ENTITY_TYPES   # ENTITY_TYPES = ["vehicle","aircraft","vessel","weapon","location","other"]


_processor = Blip2Processor.from_pretrained(BLIP2_MODEL_NAME)
_model     = Blip2ForConditionalGeneration.from_pretrained(BLIP2_MODEL_NAME).to(device)
def score_pair(crop_img: Image.Image, name: str) -> float:
    """
    问 BLIP-2:“Is this aircraft {name}? yes/no”，取生成概率里 'yes' 的概率。
    """
    prompt = f"Is this aircraft {name}? Answer yes or no."
    inputs = _processor(images=crop_img, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True,
                              output_scores=True)
    # 取 logits 分布估计 "yes" 概率
    logits = out.scores[0][0]          # (vocab, )
    yes_id  = _processor.tokenizer(" yes", add_special_tokens=False)["input_ids"][0]
    no_id   = _processor.tokenizer(" no",  add_special_tokens=False)["input_ids"][0]
    prob_yes = torch.softmax(logits[[yes_id, no_id]], dim=-1)[0].item()
    return prob_yes            # 0~1
