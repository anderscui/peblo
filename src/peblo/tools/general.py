# coding=utf-8
from peblo.providers import BaseLlmProvider


def what_is_this(provider: BaseLlmProvider, text: str) -> dict[str, str]:
    prompt = f"请总结以下文本，尽量使用三到五句话，同时给出3-5个的关键词，输出的语言尽量与以下文本之语言一致。\n\n{text}"
    print('prompt:', prompt)
    resp = provider.generate(prompt)
    return {"summary": resp}
