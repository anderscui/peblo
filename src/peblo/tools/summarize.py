# coding=utf-8
from peblo.providers import BaseLlmProvider


def summarize(provider: BaseLlmProvider, text: str) -> dict[str, str]:
    # prompt = f"Summarize the following text in 3-5 sentences and give 5 keywords.\n\n{text}"
    prompt = f"请总结以下文本，尽量使用三到五句话，同时给出3-5个的关键词，输出的语言尽量与以下文本之语言一致。\n\n{text}"
    print('prompt:', prompt)
    resp = provider.generate(prompt)
    return {"summary": resp}


def simple_summarizer(text: str, max_summary_chars: int = 2000) -> str:
    """
    A very naive summarizer: returns the first N characters.
    Designed as a temporary fallback so the whole pipeline works.
    """
    text = text.strip()
    if len(text) <= max_summary_chars:
        return text

    head = text[: max_summary_chars].rstrip()

    return (
        head
        + '\n\n...[TRUNCATED]...\n\n'
        + f'(Original length = {len(text)} chars)'
    )
