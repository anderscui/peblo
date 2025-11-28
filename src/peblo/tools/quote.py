# coding=utf-8
import json

VERIFY_PROMPT = """你是一个严格的名言考证助手。

任务：
判断下面这句话是否确实出自所指定的作者。

规则：
1. 只有在你非常确定时才可以回答“true”或“false”。
2. 若确认属实，必须给出明确出处（书名、文章或演讲）。
3. 若确认不是该作者所说，但能确定真实作者，也必须给出真实出处。
4. 若无法确认，一律回答“unknown”。
5. 严禁编造任何作者或出处。
6. 只允许输出 JSON，不允许输出解释、说明或多余文字。

输出格式（严格 JSON）：
{{
  "type": "verify",
  "result": "true | false | unknown",
  "quote": "",
  "author": "",
  "source": ""
}}

待验证语句：
{text}

指定作者：
{author}
"""

SEARCH_PROMPT = """你是一个严格的名言检索助手。

任务：
根据给出的“部分原文或大意”，检索真实存在的名言。
如果提供了作者，仅在高度匹配且确定时才返回结果。

规则：
1. 只有在你非常确定真实存在时才返回。
2. 必须提供明确出处。
3. 若无法确定，一律返回“not_found”。
4. 严禁编造。
5. 最多只返回一条。
6. 只允许输出 JSON，不允许输出解释、说明或多余文字。

输出格式（严格 JSON）：
{{
  "type": "search",
  "result": "found | not_found",
  "quote": "",
  "author": "",
  "source": ""
}}

输入内容：
{text}

限定作者（如有）：
{author}
"""


def quote_check(provider, mode: str, quote: str, author: str | None = None) -> dict[str, str]:
    if mode == 'verify':
        prompt = VERIFY_PROMPT.format(text=quote, author=author)
    else:
        if author is None:
            author = ''
        prompt = SEARCH_PROMPT.format(text=quote, author=author)

    messages = [
        {
            'role': 'user',
            "content": prompt,
        }
    ]
    resp = provider.chat(messages)
    # print(resp)

    if isinstance(resp, dict):
        resp_text = (
            resp.get("message", {}).get("content")
            or resp.get("content")
            or resp.get("text")
            or ""
        )
    else:
        resp_text = str(resp)

    try:
        data = json.loads(resp_text)
    except json.JSONDecodeError:
        if mode == "verify":
            return {
                "type": "verify",
                "result": "unknown",
                "quote": "",
                "author": "",
                "source": ""
            }
        else:
            return {
                "type": "search",
                "result": "not_found",
                "quote": "",
                "author": "",
                "source": ""
            }

    if mode == 'verify':
        return {
            "type": mode,
            "result": data.get("result", "unknown"),
            "quote": data.get("quote", ""),
            "author": data.get("author", ""),
            "source": data.get("source", "")
        }
    else:
        return {
            "type": "search",
            "result": data.get("result", "not_found"),
            "quote": data.get("quote", ""),
            "author": data.get("author", ""),
            "source": data.get("source", "")
        }


if __name__ == '__main__':
    from peblo.providers import OllamaProvider

    llm = OllamaProvider(model='qwen3-vl:8b-instruct')
    print(quote_check(llm, 'verify', '我什么都能抗拒，除了诱惑。', '鲁迅'))
    print(quote_check(llm, 'search', 'Be yourself; everyone else is already taken.', '王尔德'))
    print(quote_check(llm, 'search', 'A room without books is like a body without a soul.', '王尔德'))
