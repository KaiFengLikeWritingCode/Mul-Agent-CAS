# ner.py — DeepSeek‑only implementation (no local GPU needed)
"""
Extract military‑related entities (name + label) from an image caption.
The heavy lifting is done by DeepSeek‑Chat’s JSON‑mode, yielding state‑of‑the‑art
accuracy while avoiding local BLIP‑2 checkpoints.

Requirements
------------
$ pip install openai pillow requests

Environment variables
---------------------
DEEPSEEK_API_KEY   – your DeepSeek Chat API token.

Usage
-----
>>> from ner import extract_entities
>>> ents = extract_entities("img.jpg", "Tornado of Tactical Air Force Squadron 51 …")
>>> print(ents)
[{"name": "Tornado", "label": "aircraft"}, ...]
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

from openai import OpenAI

# ---------------------------------------------------------------------------
# Constants & client
# ---------------------------------------------------------------------------
ENTITY_TYPES: List[str] = [
    "vehicle",
    "aircraft",
    "vessel",
    "weapon",
    "location",
    "other",
]

# _DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_DEEPSEEK_BASE_URL = "https://www.chataiapi.com/v1"
_DEEPSEEK_MODEL = "deepseek-chat"
_DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not _DEEPSEEK_API_KEY:
    raise EnvironmentError("Please set the DEEPSEEK_API_KEY environment variable.")

_client = OpenAI(api_key=_DEEPSEEK_API_KEY, base_url=_DEEPSEEK_BASE_URL)

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

# def _call_deepseek_chat(prompt: str) -> str:
#     """Low‑level helper – calls DeepSeek Chat in JSON‑mode and returns raw text."""
#
#     # # One‑shot example to stabilise output
#     # example_input = (
#     #     "Tornado of Tactical Air Force Squadron 51 takes off from airbase for mission in Syria."
#     # )
#     # example_output = [
#     #     {"name": "Tornado", "label": "aircraft"},
#     #     {"name": "Tactical Air Force Squadron 51", "label": "other"},
#     #     {"name": "Syria", "label": "location"},
#     # ]
#     # system_msg = (
#     #     "You are an information extractor specialising in military context. "
#     #     "Return *only* a valid JSON array. Each item must have fields 'name' and 'label', "
#     #     f"and label must be one of {ENTITY_TYPES}.Be careful not to extract the name SOLDIERS or SOLDIER!"
#     # )
#
#     # example_input = (
#     #     "A Luftwaffe Eurofighter flies as Quick Reaction Alert over southern Germany."
#     # )
#     # example_output = [
#     #     {"name": "Eurofighter", "label": "aircraft"},
#     #     {"name": "Luftwaffe", "label": "other"},
#     #     {"name": "Quick Reaction Alert", "label": "other"},
#     #     {"name": "southern Germany", "label": "location"},
#     # ]
#     system_msg = (
#         "You are a military-domain information extractor.\n"
#         "◆ 任务: 从给定英文句子中找出所有【专有名词】军用实体或地名，"
#         "对每个实体输出 {\"name\",\"label\"}，其中 label ∈ "
#         f"{ENTITY_TYPES}。\n"
#         "◆ 只保留具体型号、正式部队/机构名称、固有地名；"
#         "忽略纯描述性或通用类别词 (e.g. \"armoured infantry fighting vehicle\", "
#         "\"gas mask\", \"overall system demonstrator\").\n"
#         "◆ 同一对象若出现多种说法，只保留最常用/最具体的那一个。\n"
#         "◆ 绝不抽取 \"soldier\", \"soldiers\" 或其同义词。\n"
#         "◆ 若句子里不存在任何符合条件的实体，输出空数组 []。\n"
#         "◆ 只输出合法 JSON，不要解释，不要多余字段。"
#     )
#
#     few_shots = [
#         # ★ 正向示例
#         {
#             "input": "Tornado of Tactical Air Force Squadron 51 takes off from airbase for mission in Syria.",
#             "output": [
#                 {"name": "Tornado", "label": "aircraft"},
#                 {"name": "Tactical Air Force Squadron 51", "label": "other"},
#                 {"name": "Syria", "label": "location"}
#             ]
#         },
#         # ★ 反向示例：丢弃通用类别词
#         {
#             "input": "The armoured infantry fighting vehicle PUMA was showcased at the exposition.",
#             "output": [
#                 {"name": "PUMA", "label": "vehicle"}
#             ]
#         },
#         # ★ 反向示例：无专有名词 → 返回 []
#         {
#             "input": "A soldier puts on a gas mask inside the chemical plant.",
#             "output": []
#         }
#     ]
#
#     # messages = [
#     #     {"role": "system", "content": system_msg},
#     #     {"role": "user", "content": example_input},
#     #     {"role": "assistant", "content": json.dumps(example_output)},
#     #     {"role": "user", "content": prompt},
#     # ]
#
#     messages = [{"role": "system", "content": system_msg}]
#     for shot in few_shots:
#         messages.append({"role": "user", "content": shot["input"]})
#         messages.append({"role": "assistant", "content": json.dumps(shot["output"])})
#     # 当前要处理的句子
#     messages.append({"role": "user", "content": prompt})
#
#     response = _client.chat.completions.create(
#         model=_DEEPSEEK_MODEL,
#         messages=messages,
#         response_format={"type": "json_object"},  # forces JSON
#         temperature=0,
#         max_tokens=256,
#         n=3
#     )
#     candidates = [json.loads(c.message.content) for c in response.choices]
#     return response.choices[0].message.content.strip()

import json
from typing import List, Dict

def _call_deepseek_chat(prompt: str) -> List[Dict]:
    """
    Low-level helper – calls DeepSeek Chat in JSON-mode with few-shot examples,
    uses n=3 outputs + voting (choose shortest valid list) to stabilize,
    and returns the parsed JSON list of {"name","label"} dicts.
    """
    # 1. system prompt describing strict extraction rules
    system_msg = (
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

    # 2. few-shot examples (positive + negative)
    few_shots = [
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
            "input": "Soldiers operate a medium artillery rocket system II (MARS) on the training ground.",
            "output": [
                {"name": "MARS", "label": "weapon"}
            ]
        },
        {
            "input": "Soldiers test the MARS II rocket launcher on a military test site.",
            "output": [
                {"name": "MARS", "label": "weapon"}
            ]
        },
        {
            "input": "A soldier puts on a gas mask inside the chemical plant.",
            "output": []
        },
        {
            "input": "External landing of a helicopter on a naval vessel with 12 soldiers.",
            "output": []
        },

    ]

    # 3. assemble messages
    messages = [{"role": "system",  "content": system_msg}]
    for shot in few_shots:
        messages.append({"role": "user",      "content": shot["input"]})
        messages.append({"role": "assistant", "content": json.dumps(shot["output"])})
    messages.append({"role": "user",      "content": prompt})

    # 4. call DeepSeek Chat with n=3 for multiple candidates
    response = _client.chat.completions.create(
        model=_DEEPSEEK_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=256,
        n=6
    )

    # 5. parse and collect valid outputs (handle both str and list types)
    candidates: List[List[Dict]] = []
    for choice in response.choices:
        content = choice.message.content
        # if already list, use directly
        if isinstance(content, list):
            data = content
        else:
            raw = content.strip()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
        # validate structure
        if isinstance(data, list) and all(
            isinstance(e, dict) and e.get("label") in ENTITY_TYPES
            for e in data
        ):
            candidates.append(data)

    # 6. choose the “cleanest” output: the shortest list often has fewer spurious items
    if not candidates:
        return []
    best = min(candidates, key=lambda lst: len(lst))
    return json.dumps(best, ensure_ascii=False)

def _clean_entities(raw_json: str) -> List[Dict[str, str]]:
    """Parse and sanity‑check DeepSeek's JSON string."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        # Fallback: try to locate the *first* JSON array in the response
        import re

        m = re.search(r"\[.*?]", raw_json, re.S)
        data = json.loads(m.group(0)) if m else []

    cleaned: List[Dict[str, str]] = []
    for item in data:
        name = item.get("name")
        label = item.get("label")
        if isinstance(name, str) and label in ENTITY_TYPES:
            cleaned.append({"name": name, "label": label})
    return cleaned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_entities(image_path: str, text: str) -> List[Dict[str, str]]:  # noqa: D401
    """Extract entities from *caption* `text`.

    The `image_path` parameter is kept for API compatibility with the old BLIP‑2
    version, but the current implementation does **not** inspect the image –
    DeepSeek‑Chat achieves higher recall from text alone. Replace BLIP‑2 with
    DeepSeek‑VL2 if bounding‑box grounding is required later.
    """
    raw = _call_deepseek_chat(text)
    entities = _clean_entities(raw)
    return entities


# ---------------------------------------------------------------------------
# CLI helper (optional)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, pprint

    parser = argparse.ArgumentParser(description="DeepSeek‑based entity extractor")
    parser.add_argument("caption", help="Image caption / sentence")
    parser.add_argument("image", nargs="?", help="Dummy path (kept for compatibility)")
    args = parser.parse_args()

    pprint.pp(_call_deepseek_chat(args.caption))
