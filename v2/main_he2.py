# ner_detect.py — 先 NER 再定位
import os, json
from typing import List, Dict, Optional
import openai

# ———————————— 配置 ————————————
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("请设置 OPENAI_API_KEY 环境变量")

# 文本 NER 用模型（文本+JSON-mode）
_NER_MODEL = "gpt-4o"
# 定位用模型（多模态+JSON-mode）
_LOC_MODEL = "gpt-4o"

ENTITY_TYPES = ["vehicle","aircraft","vessel","weapon","location","other"]
LOCATABLE = {"vehicle","aircraft","vessel","weapon"}


# ———————————— 1. 文本抽取 NER ————————————
def _ner_text(caption: str, n: int = 3) -> List[Dict[str,str]]:
    # print(caption)
    sys = (
        "You are a military-domain information extractor.\n"
        "◆ 任务: 从给定英文句子中找出所有【专有名词】军用实体或地名，"
        "对每个实体输出 {\"name\",\"label\"}，其中 label ∈ "
        f"{ENTITY_TYPES}.\n"
        "◆ 只保留具体型号、正式部队/机构名称、固有地名；"
        "忽略纯描述性或通用类别词 (e.g. \"armoured infantry fighting vehicle\", "
        "\"gas mask\", \"overall system demonstrator\").\n"
        "◆ 同一对象若出现多种说法，只保留最常用/最具体的那一个。\n"
        "◆ 绝不抽取 \"soldier\", \"soldiers\" 或其同义词。\n"
        "◆ 若句子里不存在任何符合条件的实体，输出空数组 [].\n"
        "◆ 只输出合法 JSON，不要解释，不要多余字段。"
    )
    # few = [
    #     ("Tornado of Tactical Air Force Squadron 51 takes off for mission in Syria.",
    #      [{"name":"Tornado","label":"aircraft"},
    #       {"name":"Tactical Air Force Squadron 51","label":"other"},
    #       {"name":"Syria","label":"location"}]),
    #     ("The armoured infantry fighting vehicle PUMA was showcased at the exposition.",
    #      [{"name":"PUMA","label":"vehicle"}]),
    # ]

    few = [
        {
            "input": "Tornado of Tactical Air Force Squadron 51 takes off from airbase for mission in Syria.",
            "output": [
                {"name": "Tornado", "label": "aircraft"},
                {"name": "Tactical Air Force Squadron 51", "label": "other"},
                {"name": "Syria", "label": "location"}
            ]
        },
        {
            "input": "The armoured infantry fighting vehicle PUMA was showcased at the exposition.",
            "output": [
                {"name": "PUMA", "label": "vehicle"}
            ]
        },
        {
            "input": "A soldier puts on a gas mask inside the chemical plant.",
            "output": []
        }
    ]

    msgs = [{"role":"system","content":sys}]
    # for cap, ents in few:
    #     msgs.append({"role":"user","content":cap})
    #     msgs.append({
    #         "role":"assistant",
    #         "content": json.dumps(ents, ensure_ascii=False)
    #     })
    for shot in few:
        msgs.append({"role": "user",      "content": shot["input"]})
        msgs.append({"role": "assistant", "content": json.dumps(shot["output"])})

    msgs.append({"role":"user","content":caption})

    resp = openai.chat.completions.create(
        model=_NER_MODEL,
        messages=msgs,
        response_format={"type":"json_object"},
        temperature=0,
        max_tokens=256,
        n=n
    )

    # print(resp)

    candidates: List[List[Dict[str,str]]] = []
    for ch in resp.choices:
        data = ch.message.content

        # 如果是字符串，尝试 json.loads
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                continue

        # 如果是单个 dict，包装成 list
        if isinstance(data, dict):
            data = [data]

        # 现在 data 应该是 list
        if (
            isinstance(data, list)
            and all(isinstance(x, dict) for x in data)
            and all("name" in x and "label" in x for x in data)
        ):
            candidates.append(data)

    if not candidates:
        return []

    # 取最短（最干净）
    best = min(candidates, key=len)
    return best

# ———————————— 2. 多模态定位 ————————————
def _locate_entity(image_url: str, name: str, n: int = 2) -> Optional[List[int]]:
    """
    对单个实体 name，询问模型：“这个对象在哪里？”
    返回 bbox or None
    """
    sys = (
        "你是多模态定位助手。给定一张图片和一个实体名称，"
        "返回该实体在图中的边界框 bbox，格式 {\"bbox\":[xmin,ymin,xmax,ymax]}，"
        "如果未出现则返回 {\"bbox\":null}，只要 JSON，不要其它。"
    )
    user = f"Caption: Here is the object {name}."
    # 用 Markdown 嵌入图像
    user_img = f"![]({image_url})"
    msgs = [
        {"role":"system","content":sys},
        {"role":"user","content":user},
        {"role":"user","content":user_img}
    ]
    resp = openai.chat.completions.create(
        model=_LOC_MODEL,
        messages=msgs,
        response_format={"type":"json_object"},
        temperature=0,
        max_tokens=64,
        n=n
    )
    # 取第一条合法输出
    for ch in resp.choices:
        content = ch.message.content
        if isinstance(content,dict) and "bbox" in content:
            return content["bbox"]
        if isinstance(content,str):
            try:
                j = json.loads(content)
                if "bbox" in j:
                    return j["bbox"]
            except:
                pass
    return None

# ———————————— 3. 组合接口 ————————————
def extract_entities(
    image_url: str,
    caption: str
) -> List[Dict[str, Optional[List[int]]]]:
    """
    1. 调用 _ner_text 得到 [{name,label},…]
    2. 对 label ∈ LOCATABLE 的实体调用 _locate_entity 得 bbox
    3. 返回带 bbox 的列表
    """
    ents = _ner_text(caption)
    print(ents)
    results = []
    for e in ents:
        name, label = e["name"], e["label"]
        if label in LOCATABLE:
            bbox = _locate_entity(image_url, name)
        else:
            bbox = None
        results.append({"name":name, "label":label, "bbox":bbox})
    return results

# ———————————— CLI 测试 ————————————
if __name__=="__main__":
    import argparse, pprint
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="https://raw.githubusercontent.com/KaiFengLikeWritingCode/Mul-Agent-CAS/main/datasets/sample_image/15.jpg")
    p.add_argument("--caption", default="A Eurofighter in a special livery and another with a different marking fly together with an A400M transport plane and a Transall transport plane in a special retro livery.")
    args = p.parse_args()
    pprint.pp(extract_entities(args.image, args.caption))
