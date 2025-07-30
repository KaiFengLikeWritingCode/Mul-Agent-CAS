import openai

_DESC_MODEL = "gpt-4o"  # 使用 GPT-4o 生成描述

def generate_description_with_llm(name: str, caption: str) -> str:
    """
    调用 GPT-4o 生成实体 name 在 caption 中的简短语义描述。
    """
    sys_prompt = (
        "你是军事领域信息提取助手。\n"
        "任务：根据实体名称和整句文本，生成该实体的简短上下文描述，用于图像匹配。\n"
        "要求：\n"
        "1. 以简洁、专业的方式描述。\n"
        "2. 不超过 1 句。\n"
        "3. 语言为英文。"
    )
    user_prompt = f"Entity: {name}\nText: {caption}\n\n请生成该实体的简短上下文描述。"

    resp = openai.chat.completions.create(
        model=_DESC_MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        max_tokens=64
    )

    return resp.choices[0].message.content.strip()
