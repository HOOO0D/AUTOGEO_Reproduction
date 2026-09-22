"""
Unified OpenRouter client for AutoGEO.
"""

import os
import time
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

from .constants import MAX_RETRIES, RETRY_DELAY_SECONDS


load_dotenv("keys.env")

_client: Optional[OpenAI] = None


# AutoGEO 原仓库模型名 -> OpenRouter model slug
MODEL_ALIASES = {
    # Google
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    "gemini-2.5-flash-lite": "google/gemini-2.5-flash-lite",

    # OpenAI
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",

    # Anthropic
    "claude-3-haiku-20240307": "anthropic/claude-3-haiku",
    "claude-3-5-sonnet-20241022": "anthropic/claude-3.5-sonnet",
}


def normalize_model_name(model_name: str) -> str:
    """
    Convert original AutoGEO model names to OpenRouter model slugs.
    """

    # 已经是 OpenRouter 格式
    if "/" in model_name:
        return model_name

    if model_name in MODEL_ALIASES:
        return MODEL_ALIASES[model_name]

    # Gemini / GPT 的命名规律比较稳定
    if model_name.startswith("gemini-"):
        return f"google/{model_name}"

    if model_name.startswith("gpt-"):
        return f"openai/{model_name}"

    if model_name.startswith("claude-"):
        return f"anthropic/{model_name}"

    raise ValueError(
        f"Unknown model name: {model_name}. "
        "Please add it to MODEL_ALIASES in openrouter.py."
    )


def get_openrouter_client() -> OpenAI:
    global _client

    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in keys.env"
            )

        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=120.0,
        )

    return _client


def call_openrouter(
    user_prompt: str,
    model_name: str,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = 0.7,
    max_tokens: Optional[int] = None,
    provider: Optional[Dict[str, Any]] = None,
    response_format: Optional[Dict[str, Any]] = None,
    max_retries: int = MAX_RETRIES,
) -> str:

    model = normalize_model_name(model_name)

    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    messages.append({
        "role": "user",
        "content": user_prompt
    })

    client = get_openrouter_client()

    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
            }

            if temperature is not None:
                kwargs["temperature"] = temperature

            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            if response_format is not None:
                kwargs["response_format"] = response_format

            # OpenRouter provider routing，可选
            if provider is not None:
                kwargs["extra_body"] = {
                    "provider": provider
                }

            response = client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content

            if content is None:
                raise RuntimeError(
                    f"OpenRouter returned empty content for {model}"
                )

            return content

        except Exception as e:
            if attempt < max_retries - 1:
                print(
                    f"[OpenRouter] {model} failed "
                    f"({attempt + 1}/{max_retries}): {e}"
                )
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                raise
