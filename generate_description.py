import os
import json
from openai import OpenAI

# DeepSeek API 配置
_DEEPSEEK_BASE_URL = "https://www.chataiapi.com/v1"
_DEEPSEEK_MODEL = "deepseek-chat"
_DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not _DEEPSEEK_API_KEY:
    raise EnvironmentError("Please set the DEEPSEEK_API_KEY environment variable.")

_client = OpenAI(api_key=_DEEPSEEK_API_KEY, base_url=_DEEPSEEK_BASE_URL)

def generate_description_with_llm(name: str, caption: str) -> str:
    """
    调用 DeepSeek Chat 生成实体 name 在 caption 中的简短上下文描述。
    """
    # "任务：根据实体名称和整句文本，生成该实体的简短上下文描述，用于图像匹配。\n"
    system_msg = (
        "你是军事领域信息提取助手。\n"
        "任务：根据实体名称和整句文本，生成该实体的简短上下文描述，介绍一下这个实体名是什么，注意不要引入新的实体名，避免干扰目标检测。\n"
        "要求：\n"
        "1. 以简洁、专业的方式描述。\n"
        "2. 不超过 1 句。\n"
        "3. 语言为英文。"
    )

    user_msg = f"Entity: {name}\nText: {caption}\n\n请生成该实体的简短上下文描述。"

    # 调用 DeepSeek Chat
    response = _client.chat.completions.create(
        model=_DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0,
        max_tokens=64,
        n=1
    )

    # 提取生成的内容
    return response.choices[0].message.content.strip()


def generate_description_with_llm_2(name: str, caption: str) -> str:
    """
    调用 DeepSeek Chat 生成实体 name 在 caption 中的简短上下文描述，
    严格限制避免引入新的实体名。
    """
    system_msg = (
        "You are a military information extraction assistant.\n"
        "Task: Generate a short contextual description of the given entity based ONLY on the provided text.\n"
        "Strict requirements:\n"
        "1. Only describe the given entity itself using context from the text.\n"
        "2. Do NOT introduce any new entities, objects, or names beyond the given entity.\n"
        "3. Use a single concise sentence in English.\n"
        "4. The description must be directly derived from the text and NOT infer or invent extra details.\n"
        "5. The output must ONLY focus on the given entity and avoid referencing any other objects.\n"
        "6. Do NOT repeat the entity name more than once.\n"
    )

    user_msg = (
        f"Entity: {name}\n"
        f"Text: {caption}\n\n"
        "Generate ONLY a short description of this entity based strictly on the text."
    )

    response = _client.chat.completions.create(
        model=_DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0,
        max_tokens=64,
        n=1
    )

    return response.choices[0].message.content.strip()


def generate_description_with_llm_3(name: str, caption: str) -> str:
    """
    调用 DeepSeek Chat 生成实体 name 在 caption 中的简短上下文描述，
    严格禁止引入新的实体或主体。
    """
    system_msg = (
        "You are a military information extraction assistant.\n"
        "Task: Generate a short description ONLY about the given entity based strictly on the provided text.\n"
        "Hard constraints:\n"
        "1. Only describe the given entity itself. Do NOT mention any other person, group, object, or entity.\n"
        "2. Do NOT introduce or refer to subjects (e.g., 'soldiers', 'pilots', 'crew').\n"
        "3. Use adjectives or functional roles ONLY for the given entity.\n"
        "4. Use a single concise sentence in English.\n"
        "5. The description must NOT include any new or inferred names/terms other than the given entity.\n"
        "6. Do NOT repeat the entity name more than once.\n"
        "7. If unsure, output only a generic attribute (e.g., 'It is a military vehicle.')\n"
    )

    user_msg = (
        f"Entity: {name}\n"
        f"Text: {caption}\n\n"
        "Generate ONLY a short description of this entity. Exclude any mention of people or other entities."
    )

    response = _client.chat.completions.create(
        model=_DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0,
        max_tokens=64,
        n=1
    )

    return response.choices[0].message.content.strip()

# 示例调用
if __name__ == "__main__":
    desc = generate_description_with_llm("Eurofighter",
                                         "A Eurofighter in a special livery flies alongside a transport plane.")
    print(desc)
