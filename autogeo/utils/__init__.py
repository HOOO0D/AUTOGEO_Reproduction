"""
Utilities module for AutoGEO.

Contains LLM clients and utility functions.
"""
from .gemini import call_gemini
from .openai import call_openai
from .anthropic import call_claude
from .logger import get_logger


def call_hf_model(*args, **kwargs):
    """
    Lazily import HuggingFace dependencies only when a local model is used.
    """
    from .hf_model import call_hf_model as _call_hf_model
    return _call_hf_model(*args, **kwargs)


__all__ = [
    "call_gemini",
    "call_openai",
    "call_claude",
    "get_logger",
    "call_hf_model",
]