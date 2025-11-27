# coding=utf-8
from pathlib import Path

from peblo.providers import BaseLlmProvider
from peblo.utils.images import image_to_base64


def ocr_by_llm(provider: BaseLlmProvider, file_path: str) -> dict[str, str]:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(file_path)

    img_b64 = image_to_base64(file_path)

    messages = [
        {
            'role': 'user',
            # "content": "Extract all readable text from this image. Only return the text.",
            "content": "请从此图片中提取所有文字，并仅返回文字内容。",
            "images": [img_b64],
        }
    ]

    text = provider.chat(messages)
    return {
        'file': file_path,
        'engine': 'llm',
        'text': text
    }


if __name__ == '__main__':
    from peblo.providers import OllamaProvider

    llm = OllamaProvider(model='deepseek-ocr:3b')
    # llm = OllamaProvider(model='qwen3-vl:8b-instruct')
    print(ocr_by_llm(llm, '/Users/andersc/Downloads/images/出梁庄记-26.png'))
