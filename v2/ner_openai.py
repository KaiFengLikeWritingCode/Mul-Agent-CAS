# ner.py — OpenAI-only implementation
"""
Extract military-related entities (name + label) from an image caption,
using OpenAI’s ChatCompletion API in a few-shot, JSON-only style.

Requirements
------------
$ pip install openai pillow

Environment variables
---------------------
OPENAI_API_KEY   – your OpenAI API token.

Usage
-----
>>> from ner import extract_entities
>>> ents = extract_entities("img.jpg", "Tornado of Tactical Air Force Squadron 51 …")
>>> print(ents)
[{"name": "Tornado", "label": "aircraft"}, ...]
"""

import os
import re
import json
from typing import List, Dict

import openai

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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")
openai.api_key = OPENAI_API_KEY

# You can swap to "gpt-4o-mini" or another capable model
_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Core helper: call OpenAI Chat with few-shot + parse JSON
# ---------------------------------------------------------------------------
def _call_openai_chat(prompt: str) -> str:
    """
    Low-level helper – calls OpenAI ChatCompletion in JSON-only mode with
    few-shots, temperature=0, n=3 for voting, returns the chosen JSON text.
    """
    # 1. System prompt with strict rules
    system_msg = (
        "You are a military-domain information extractor.\n"
        "◆ Task: From the given English sentence, identify ALL proper-noun military entities or place names.\n"
        "◆ For each, output exactly {\"name\",\"label\"}, where label ∈ "
        f"{ENTITY_TYPES}.\n"
        "◆ Only keep specific model names, formal unit/organization names, or proper place names;\n"
        "   ignore generic/descriptive terms (e.g. \"armoured infantry fighting vehicle\",\n"
        "   \"gas mask\", \"overall system demonstrator\").\n"
        "◆ If an object appears multiple ways, keep only the most common/concise name.\n"
        "◆ Never extract \"soldier\" or \"soldiers\" or synonyms.\n"
        "◆ If there are NO valid entities, output an empty array [].\n"
        "◆ Output MUST be valid JSON and NOTHING else."
    )

    # 2. Few-shot examples (positive + negative)
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
            "input": "A soldier puts on a gas mask inside the chemical plant.",
            "output": []
        }
    ]

    # 3. Assemble messages
    messages = [{"role": "system", "content": system_msg}]
    for shot in few_shots:
        messages.append({"role": "user",      "content": shot["input"]})
        messages.append({"role": "assistant", "content": json.dumps(shot["output"])})
    messages.append({"role": "user", "content": prompt})

    # 4. Call OpenAI ChatCompletion with n=3 for voting
    resp = openai.ChatCompletion.create(
        model=_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=256,
        n=3
    )

    # 5. Collect & validate JSON candidates
    candidates: List[List[Dict]] = []
    for choice in resp.choices:
        raw = choice.message.content.strip()
        # Extract first JSON array if wrapped
        if not raw.startswith("["):
            m = re.search(r"\[.*\]", raw, re.S)
            raw = m.group(0) if m else raw
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(data, list)
            and all(isinstance(e, dict) and e.get("label") in ENTITY_TYPES for e in data)
        ):
            candidates.append(data)

    # 6. Pick the “cleanest” (shortest) candidate, fallback to empty list
    best = min(candidates, key=len) if candidates else []
    return json.dumps(best, ensure_ascii=False)


def _clean_entities(raw_json: str) -> List[Dict[str, str]]:
    """Parse and sanity-check the JSON string from OpenAI."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw_json, re.S)
        data = json.loads(m.group(0)) if m else []
    cleaned: List[Dict[str, str]] = []
    for item in data:
        n = item.get("name")
        l = item.get("label")
        if isinstance(n, str) and l in ENTITY_TYPES:
            cleaned.append({"name": n, "label": l})
    return cleaned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_entities(image_path: str, text: str) -> List[Dict[str, str]]:
    """
    Extract entities from the caption `text`. `image_path` is kept for API
    compatibility but not used (text-only).
    """
    raw = _call_openai_chat(text)
    return _clean_entities(raw)


# ---------------------------------------------------------------------------
# CLI helper (optional)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, pprint
    parser = argparse.ArgumentParser(description="OpenAI-based entity extractor")
    parser.add_argument("caption", help="Image caption / sentence")
    args = parser.parse_args()
    pprint.pp(extract_entities("", args.caption))
