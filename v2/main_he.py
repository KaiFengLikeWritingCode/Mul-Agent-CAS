# ner.py — 调用 OpenAI 多模态 API（Markdown+JSON-mode）
import os, json, base64
from typing import List, Dict, Optional
import openai

# ———————————— 配置 ————————————
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("请设置 OPENAI_API_KEY")

# 使用支持 vision + JSON-mode 的轻量模型
_MODEL = "gpt-4o-mini"
# _MODEL = "o4-mini"

ENTITY_TYPES = ["vehicle","aircraft","vessel","weapon","location","other"]
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# ———————————— 核心函数 ————————————
def extract_entities(
    image_url: str,
    caption:    str,
    n:          int = 3
) -> List[Dict[str, Optional[List[int]]]]:
    # 将图像转 base64，并用 Markdown 嵌入
    # img_b64 = base64.b64encode(open(image_path,"rb").read()).decode()
    # md_image = f"![img](data:image/png;base64,{img_b64})"
    # image_url = "https://github.com/KaiFengLikeWritingCode/Mul-Agent-CAS/blob/main/datasets/sample_image/15.jpg"
    # image_url = "https://github.com/KaiFengLikeWritingCode/Mul-Agent-CAS/blob/main/datasets/sample_image/15.jpg"
    md_image = f"![img]({image_url})"

    # 系统提示 + few-shot 示例，全部写在一个 content 里
    system_msg = (
        "You are a multimodal military-domain extractor. "
        "Given an image and its caption, extract ALL military-related entities "
        "and their types, and locate each in the image if visible.\n\n"
        "Output ONLY a JSON array. Each item must have:\n"
        "- name: the proper name (keep only model names, unit names, place names).\n"
        f"- label: one of {ENTITY_TYPES}\n"
        "- bbox: [xmin,ymin,xmax,ymax] if the object appears, or null otherwise.\n\n"
        "Ignore generic or descriptive terms like 'armoured infantry fighting vehicle' or 'gas mask'.\n"
        "If no valid entities, output `[]`.\n\n"
        "### Examples\n\n"
        "**Input:**\n" + md_image + "\n\nCaption: \"Tornado of Tactical Air Force Squadron 51 takes off for mission in Syria.\"\n\n"
        "**Output:**\n"
        "```json\n"
        "[\n"
        "  {\"name\":\"Tornado\",\"label\":\"aircraft\",\"bbox\":[100,50,400,200]},\n"
        "  {\"name\":\"Tactical Air Force Squadron 51\",\"label\":\"other\",\"bbox\":null},\n"
        "  {\"name\":\"Syria\",\"label\":\"location\",\"bbox\":null}\n"
        "]\n"
        "```\n\n"
        "**Now** extract for the following image+caption:\n"
        + md_image + "\n\nCaption: \"" + caption + "\""
    )

    resp = openai.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "system", "content": system_msg}],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=512,
        n=n,
    )

    # 取最短那条
    cands = []
    for choice in resp.choices:
        out = choice.message.content
        if isinstance(out, list):
            cands.append(out)
        else:
            try:
                cands.append(json.loads(out))
            except:
                continue
    if not cands:
        return []
    best = min(cands, key=lambda x: len(x))
    return best

# ———————————— CLI 测试 ————————————
if __name__=="__main__":
    import argparse, pprint
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="https://raw.githubusercontent.com/KaiFengLikeWritingCode/Mul-Agent-CAS/refs/heads/main/datasets/sample_image/15.jpg")
    p.add_argument("--caption", default="A Eurofighter in a special livery and another with a different marking fly together with an A400M transport plane and a Transall transport plane in a special retro livery.")
    args = p.parse_args()

    res = extract_entities(args.image, args.caption)
    pprint.pp(res)


# python detector_deepseek.py     --image ./datasets/sample_image/15.jpg     --phrase "aircraft"     --out ./outputs/test_annotated_15.jpg

# python main_he.py   --image ./datasets/sample_image/15.jpg   --caption "A Eurofighter in a special livery and another with a different marking fly together with an A400M transport plane and a Transall transport plane in a special retro livery."

