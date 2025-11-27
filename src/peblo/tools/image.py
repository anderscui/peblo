# coding=utf-8
from pathlib import Path

from peblo.providers import BaseLlmProvider
from peblo.utils.images import image_to_base64


def describe_image(provider: BaseLlmProvider, file_path: str) -> dict[str, str]:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(file_path)

    img_b64 = image_to_base64(file_path)

    messages = [
        {
            'role': 'user',
            "content": "请以简洁准确的文字描述该图片的内容，仅返回文字。",
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
    # 咖啡馆2.0 (108) 群聊截图，群成员包括“梁山108位群友”、“贝壳”、“看得见的森林”、“清风翻《诗经》”等。聊天内容主要为轻松幽默的互动，如“能8.6日”、“哈哈哈”、“三星自带搜狗”、“戳中了我的笑点”、“眼泪笑出来了”、“はちろく”、“本来准备午睡的，笑得在床上打滚”等，配有表情符号，气氛欢乐。
    print(describe_image(llm, '/Users/andersc/Downloads/images/咖啡馆2.0.png'))
