"""
Rewriters module for AutoGEO.

Provides different implementations of document rewriters:
- Core: Core rewriting logic with rule loading and prompt generation
- API rewriter: Uses LLM APIs for rewriting
- Mini rewriter: Uses local models for rewriting
"""
from .core import rewrite_document, get_rewrite_prompt_template
from .api import api_rewrite_documents


def mini_rewrite_documents(*args, **kwargs):
    """
    Lazily import AutoGEO Mini dependencies only when Mini is actually used.
    """
    from .mini import mini_rewrite_documents as _mini_rewrite_documents
    return _mini_rewrite_documents(*args, **kwargs)


__all__ = [
    "rewrite_document",
    "get_rewrite_prompt_template",
    "api_rewrite_documents",
    "mini_rewrite_documents",
]
