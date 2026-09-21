"""
Gemini compatibility wrapper using OpenRouter.
"""

from .constants import COMMON_SYSTEM_PROMPT
from .openrouter import call_openrouter


def call_gemini(
    user_prompt: str,
    system_prompt: str = COMMON_SYSTEM_PROMPT,
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.7
) -> str:

    # 保持原 AutoGEO Gemini prompt 的行为：
    # 原代码也是把 system/user 拼成一个字符串再发给 Gemini
    messages = (
        f"system_prompt: {system_prompt}\n\n"
        f"user_prompt: {user_prompt}"
    )

    return call_openrouter(
        user_prompt=messages,
        model_name=model_name,
        system_prompt=None,
        temperature=temperature,
    )