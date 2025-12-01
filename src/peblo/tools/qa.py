# coding=utf-8
import json
import logging

from peblo.tools.text_input import read_text_input

logger = logging.getLogger(__name__)

QA_PROMPT = """You are a concise one-shot QA assistant.

Context:
{context}

Question:
{question}

Instructions:
- Answer only based on the Context. Do not invent facts.
- If the answer is not present or not determinable, reply: "UNKNOWN".
- If you can answer, provide a short answer (max 60 words).
- Optionally, include a 1-2 line "Source" with the exact snippet from the context that supports your answer.

Output format (STRICT JSON):
{{
  "answer": "...",
  "source_snippet": "..."
}}
"""

def qa(provider, question: str, target: str | None = None) -> dict:
    """
    One-shot QA: ask a question against a text or text file.
    """
    context = ""
    origin = "inline"
    truncated = False

    if target:
        info = read_text_input(target)
        context = info["text"]
        origin = info["origin"]
        truncated = info.get("truncated", False)

    prompt = QA_PROMPT.format(context=context, question=question)

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    resp = provider.chat(messages)
    logger.debug(f'llm resp: {resp}')

    try:
        data = json.loads(resp)
    except json.JSONDecodeError:
        return {"answer": "UNKNOWN", "source_snippet": "", "origin": origin}

    return {
        "answer": data.get("answer", "UNKNOWN"),
        "source_snippet": data.get("source_snippet", ""),
        "origin": origin,
        "truncated": truncated,
    }


if __name__ == '__main__':
    from peblo.providers import OllamaProvider

    llm = OllamaProvider(model='qwen3-vl:8b-instruct')
    # print(qa(llm, 'Tell me the programming language used in this file.', 'text_input.py'))
    print(qa(llm, 'List the functions defined in this file.', 'text_input.py'))
    print(qa(llm, 'How to fix this error?', 'ERROR: FileNotFoundError: config.yaml not found'))
    print(qa(llm, '这句话的关键词是（以列表显示）？', ' 原生多模态  统一框架支持文本、图像的检索与生成，新增 VisRAG Pipeline 实现 PDF 到多模态问答的闭环。  而且内置的多模态 Benchmark 覆盖视觉问答等任务，并提供统一的评估体系，方便研究者快速对比实验效果。'))
