# coding=utf-8


def summarize(provider, text: str) -> dict[str, str]:
    prompt = f"Summarize the following text in 3-5 sentences and give 5 keywords.\n\n{text}"
    resp = provider.generate(prompt)
    return {"summary": resp}
