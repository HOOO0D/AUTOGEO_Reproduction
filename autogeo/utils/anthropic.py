"""
Anthropic compatibility wrapper using OpenRouter.
"""

from .constants import COMMON_SYSTEM_PROMPT
from .openrouter import call_openrouter


def call_claude(
    user_prompt: str,
    model_name: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.7,
    system_prompt: str = COMMON_SYSTEM_PROMPT
) -> str:

    return call_openrouter(
        user_prompt=user_prompt,
        model_name=model_name,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=4096,
    )