# ner.py ────────────────────────────────────────────────────────────
import json, re, time
import torch
from transformers import pipeline
from config import LLM_MODEL_NAME, ENTITY_TYPES   # ["vehicle", …, "other"]

# 1. 设备
device = 0 if torch.cuda.is_available() else -1

# 2. 通用大模型，用于生成 JSON & 兜底分类
gen_pipe = pipeline(
    "text2text-generation",
    model=LLM_MODEL_NAME,
    device=device
)

# 3. 单实体分类的小 Prompt（兜底用）
def classify_type_with_llm(name: str) -> str:
    prompt = (
        f'name: "{name}"\n'
        "Which category does it belong to? "
        "Choose exactly one: vehicle, aircraft, vessel, weapon, location, other.\n"
        "Here are some sample categories: In our annotations, the following mappings are typical: PUMA, Dachs, Jaguar, Leopard 2 A7, Leopard 2 A6, Marder, M113, GTK Boxer, and Iveco Trekker are labeled as vehicle; Boeing E-3, Tornado, Eurofighter, Airbus A400M, NH-90, CH-53, F-16C/D, F-18 Hornet, Sea Lynx MK 88A, H145M, Tiger, and C-130J Hercules are aircraft; MARS, PATRIOT, MILAN, and P8 are weapon; Syria, Mediterranean Sea, Alps, Gaza Strip, and Havel are location; and UN, ABC-Defense, Combo Pen Training Simulator, Orion, Daesh, Explosive Ordnance Disposal, Heron 1, Tactical Air Force Squadron 51, Medium Girder Bridge, and Forward Air MedEvac are other."
        "Answer with only the category word."
    )
    out = gen_pipe(prompt, max_new_tokens=5)[0]["generated_text"].strip().lower()
    return out if out in ENTITY_TYPES else "other"

# 4. 主函数：抽实体 + 类型
def extract_entities(text: str):
    """返回 [{'name': 'PUMA', 'label': 'vehicle'}, …]，始终合法"""
    prompt = f"""
Example:
Input: "Soldiers test the overall system demonstrator armoured infantry fighting vehicle PUMA."
<json>
[
  {{ "name": "PUMA", "label": "vehicle" }}
]
</json>

Now do the same for the following sentence. 
Note that the extracted entities should belong to the class of vehicle, aircraft, vessel, weapon, location, other
Output must be enclosed in a single <json>...</json> block, without any commentary.
Sentence: "{text}"
"""

    raw = gen_pipe(prompt, max_new_tokens=256)[0]["generated_text"]
    # 提取 <json> … </json>
    m = re.search(r"<json>(.*?)</json>", raw, re.S | re.I)
    if m:
        json_str = m.group(1).strip()
        try:
            ents = json.loads(json_str)
            # 再次过滤非法类别
            ents = [e for e in ents if e.get("label") in ENTITY_TYPES]
            if ents:
                return ents
        except json.JSONDecodeError:
            pass  # 继续走兜底

    # ========= 兜底流程 =========
    # 1) 先用非常宽松的正则把所有连续大写/首字母词串出来
    #    这只是示例，你可以换成专业 NER
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9\-_/]+\b", text)
    seen, ents = set(), []
    for name in candidates:
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        ent_type = classify_type_with_llm(name)
        ents.append({"name": name, "label": ent_type})
        # 避免对长句过多请求，可限制最多 10 个实体
        if len(ents) >= 10:
            break
        # 可适当 sleep 避免速率限制
        time.sleep(0.1)
    return ents
# ───────────────────────────────────────────────────────────────────
