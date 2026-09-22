from typing import List

from ..utils.openrouter import call_openrouter


query_prompt = """Write an accurate and concise answer for the given user question, using _only_ the provided summarized web search results. The answer should be correct, high-quality, and written by an expert using an unbiased and journalistic tone. The user's language of choice such as English, Français, Español, Deutsch, or 日本語 should be used. The answer should be informative, interesting, and engaging. The answer's logic and reasoning should be rigorous and defensible. Every sentence in the answer should be _immediately followed_ by an in-line citation to the search result(s). The cited search result(s) should fully support _all_ the information in the sentence. Search results need to be cited using [index]. When citing several search results, use [1][2][3] format rather than [1, 2, 3]. You can use multiple search results to respond comprehensively while avoiding irrelevant search results.

Question: {query}

Search Results:
{source_text}
"""


def generate_answer_gemini(query: str, sources: List[str], model_name: str = 'gemini-2.5-flash-lite') -> str:
    """Generate answer using Gemini with RAG.

    Args:
        query: User query text
        sources: List of source document texts
        model_name: Gemini model name (default: 'gemini-2.5-flash-lite')

    Returns:
        Generated answer text with citations

    Raises:
        Exception: If API call fails after all retries
    """
    source_text = '\n\n'.join([f'### Source {idx}:\n{source}' for idx, source in enumerate(sources)])
    prompt = query_prompt.format(query=query, source_text=source_text)
    return call_openrouter(
        user_prompt=prompt,
        model_name=model_name,
        # The original Gemini SDK call left sampling at the model default.
        temperature=None,
    )

def generate_answer_gpt(query: str, sources: List[str], model_name: str = 'gpt-4o-mini') -> str:
    """Generate answer using GPT with RAG.

    Args:
        query: User query text
        sources: List of source document texts
        model_name: OpenAI model name (default: 'gpt-4o-mini')

    Returns:
        Generated answer text with citations

    Raises:
        Exception: If API call fails after all retries
    """
    source_text = '\n\n'.join([f'### Source {idx}:\n{source}\n\n' for idx, source in enumerate(sources)])
    prompt = query_prompt.format(query=query, source_text=source_text)
    return call_openrouter(
        user_prompt=prompt,
        model_name=model_name,
        temperature=0.5,
    )

def generate_answer_claude(query: str, sources: List[str], model_name: str = 'claude-3-haiku-20240307') -> str:
    """Generate answer using Claude with RAG.

    Args:
        query: User query text
        sources: List of source document texts
        model_name: Claude model name (default: 'claude-3-haiku-20240307')

    Returns:
        Generated answer text with citations

    Raises:
        Exception: If API call fails after all retries
    """
    source_text = '\n\n'.join([f'### Source {idx}:\n{source}\n\n' for idx, source in enumerate(sources)])
    system_prompt = query_prompt.split("Question: {query}")[0].strip()
    user_content = f"Question: {query}\n\nSearch Results:\n{source_text}"
    return call_openrouter(
        user_prompt=user_content,
        model_name=model_name,
        system_prompt=system_prompt,
        max_tokens=4096,
        temperature=0.5,
    )

