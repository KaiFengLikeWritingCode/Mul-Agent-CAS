# ner.py ────────────────────────────────────────────────────────────
import json
import re
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from config import BLIP2_MODEL_NAME, ENTITY_TYPES   # ENTITY_TYPES = ["vehicle","aircraft","vessel","weapon","location","other"]

# 1. 设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 全局加载 BLIP-2 Processor 和 Model
_processor = Blip2Processor.from_pretrained(BLIP2_MODEL_NAME)
_model     = Blip2ForConditionalGeneration.from_pretrained(BLIP2_MODEL_NAME).to(device)
_model.eval()

from transformers import pipeline

# vqa_pipe = pipeline(
#     task="vqa",            # 或 "visual-question-answering"
#     model=_model,
#     processor=_processor,
#     device=0 if device.startswith("cuda") else -1,
# )
vqa_pipe = pipeline(
    task="visual-question-answering",    # 或 "vqa"
    model=_model,
    tokenizer=_processor.tokenizer,          # 文本部分
    image_processor=_processor.image_processor,  # 图像部分
    device=0 if device.startswith("cuda") else -1,
)


def extract_entities(image_path: str, text: str):
    """
    多模态实体抽取：
      Input:  image_path, text
      Output: [ {"name": str, "label": str}, … ]
    """
    # 1) 读取图像
    image = Image.open(image_path).convert("RGB")

    # 2) 构造 Prompt，强调多词实体和输出 JSON
    # prompt = (
    #     "You are shown an image and its caption. "
    #     "Identify all military-related entities, which may be multi-word phrases "
    #     "(e.g. \"armoured infantry fighting vehicle PUMA\"). "
    #     "Return ONLY a JSON array enclosed in <json>…</json>, "
    #     "each item with keys 'name' and 'label', where label is one of "
    #     "[vehicle, aircraft, vessel, weapon, location, other]."
    #     f"\n\nCaption: \"{text}\""
    #     "\n\n<json>"
    # )
    prompt = f"""
    Example:
    Input: "Tornado of Tactical Air Force Squadron 51 takes off from airbase for mission in Syria."
    <json>
    [
      {{ "name": "Tornado", "label": "aircraft" }},{{ "name": "Tactical Air Force Squadron 51", "label": "other" }},{{ "name": "Syria", "label": "location" }}
    ]
    </json>

    Now do the same for the following sentence. 
    Sentence: "{text}"
    Note that the extracted entities should belong to the class of vehicle, aircraft, vessel, weapon, location, other
    Output must be enclosed in a single <json>...</json> block, without any commentary.
    Note that name is not necessarily a word, do not extract the word as you see it, but extract the name according to the meaning of the sentence, e.g. Taktisches Luftwaffengeschwader 51 Immelmann
    """

    # 3) 编码输入并生成
    # inputs = _processor(images=image, text=prompt, return_tensors="pt").to(device)
    # print("=== [NER] inputs  ===")
    # print(inputs)
    # outputs = _model.generate(**inputs, max_new_tokens=256)
    # print("=== [NER] outputs ===")
    # print(outputs)
    # raw = _processor.decode(outputs[0], skip_special_tokens=True)

    # raw = mm_pipe(image, prompt=prompt, max_new_tokens=256)[0]["generated_text"]
    # raw = mm_pipe(image, text=prompt, max_new_tokens=256)[0]["generated_text"]
    output = vqa_pipe(image=image, question=prompt)
    raw = output["answer"]

    print("=== [NER] Raw model output ===")
    print(raw)
    print("=== [NER] End of raw output ===")

    # 4) 提取 JSON 块
    m = re.search(r"<json>(.*?)</json>", raw, re.S|re.I)
    json_str = m.group(1).strip() if m else raw.strip()

    # 5) 解析 JSON
    try:
        ents = json.loads(json_str)
    except json.JSONDecodeError:
        # 如果一不留神输出没闭合，就找第一个[...]块
        m2 = re.search(r"\[.*\]", json_str, re.S)
        ents = json.loads(m2.group(0)) if m2 else []

    # 6) 过滤异常并返回
    cleaned = []
    for e in ents:
        name = e.get("name")
        label= e.get("label")
        if isinstance(name,str) and label in ENTITY_TYPES:
            cleaned.append({"name": name, "label": label})
    return cleaned
# ───────────────────────────────────────────────────────────────────
