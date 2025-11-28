# coding=utf-8
from pathlib import Path
import json

from peblo.utils.files import is_safe_text_file

MAX_FILE_SIZE = 5 * 1024 * 1024   # 5 MB
MAX_TEXT_CHARS = 10000            # Max chars sent to models


PEEK_PROMPT = """你是一个严格的内容分析助手。

任务：
对输入的内容进行高层次分析，输出以下三部分信息：
1. 大致分类（如：code / log / config / natural_language / error / other）
2. 它在表达什么（简要概括）
3. 可能的用途 / 问题 / 注意点（数组，最多 5 条）

规则：
1. 只允许输出 JSON，不允许任何解释性文字。
2. 只基于给定内容本身进行判断，不要臆测来源。
3. 若内容信息不足，可在 summary 中说明“不足以判断”。

输出格式（严格 JSON）：
{{
  "category": "",
  "summary": "",
  "notes": []
}}

输入内容：
{content}
"""


def read_peek_text(input_arg: str) -> dict:
    """
    Normalize peek input (text or file).

    Returns:
        {
            "input_type": "file" | "text",
            "origin": str,
            "text": str,
            "truncated": bool,
            "file_size": int | None
        }
    """

    p = Path(input_arg)
    # ---------- Case 1: File ----------
    if p.is_file() and p.exists():
        file_size = p.stat().st_size

        if file_size > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large for peek (>{MAX_FILE_SIZE / 1024 / 1024:.1f} MB)"
            )

        if not is_safe_text_file(str(p)):
            raise ValueError("File is not a safe text file for peek")

        # read file contents
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        truncated = False
        if len(content) > MAX_TEXT_CHARS:
            content = content[:MAX_TEXT_CHARS]
            truncated = True

        return {
            "input_type": "file",
            "origin": str(p),
            "text": content.strip(),
            "truncated": truncated,
            "file_size": file_size,
        }

    # ---------- Case 2: Plain Text ----------
    text = input_arg.strip()
    truncated = False

    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
        truncated = True

    return {
        "input_type": "text",
        "origin": "inline",
        "text": text,
        "truncated": truncated,
        "file_size": None,
    }


def peek_analyze(provider, content: str) -> dict:
    """
    Call LLM to analyze content. If parsing fails, return a safe fallback.
    """
    prompt = PEEK_PROMPT.format(content=content)

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    resp = provider.chat(messages)
    print(resp)

    try:
        data = json.loads(resp)
    except json.JSONDecodeError:
        return {
            "category": "other",
            "summary": "Failed to parse model output.",
            "notes": []
        }

    return {
        "category": data.get("category", "other"),
        "summary": data.get("summary", ""),
        "notes": data.get("notes", []) if isinstance(data.get("notes"), list) else []
    }


def peek(provider, target: str) -> dict:
    """
    Main entry for peek tool.
    """
    info = read_peek_text(target)

    analysis = peek_analyze(provider, info['text'])

    return {
        "input": {
            "type": info["input_type"],
            "origin": info["origin"],
            "truncated": info["truncated"],
            "file_size": info["file_size"],
        },
        "analysis": analysis,
    }


if __name__ == '__main__':
    from peblo.providers import OllamaProvider

    llm = OllamaProvider(model='qwen3-vl:8b-instruct')
    print(peek(llm, 'peek.py'))
    print(peek(llm, 'ERROR: FileNotFoundError: config.yaml not found'))
    # print(peek(llm, 'ERROR: FileNotFoundError: config.yaml not found', force_file=True))
