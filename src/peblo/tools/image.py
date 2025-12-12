# coding=utf-8
from pathlib import Path

from peblo.providers import BaseLlmProvider
from peblo.commons.io.images import image_to_base64


def describe_image(provider: BaseLlmProvider, file_path: str) -> dict[str, str]:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(file_path)

    img_b64 = image_to_base64(file_path)

    # prompt = '请以简洁准确的文字描述该图片的内容，仅返回文字。'
    prompt = '请以简洁准确的文字描述该图片的内容，使用3-5句话。请聚焦于主要可视对象和场景，仅关注明确可见的内容。不使用 markdown，不添加额外解释，仅返回文字。'
#     prompt = """
# Describe the image in 3-5 concise sentences.
# Focus on the main visible objects and scene.
# Do not speculate beyond what is clearly visible.
# Do not use markdown. Do not add explanations.
# """.strip()

    messages = [
        {
            'role': 'user',
            "content": prompt,
            "images": [img_b64],
        }
    ]

    text = provider.chat(messages)
    return {
        'file': file_path,
        'engine': 'llm',
        'caption': text
    }


if __name__ == '__main__':
    from peblo.providers import OllamaProvider

    # deepseek-ocr:3b：处理稍长的 prompt 似乎有问题
    # llm = OllamaProvider(model='deepseek-ocr:3b')
    llm = OllamaProvider(model='qwen3-vl:8b-instruct')
    # 咖啡馆2.0 (108) 群聊截图，群成员包括“梁山108位群友”、“贝壳”、“看得见的森林”、“清风翻《诗经》”等。聊天内容主要为轻松幽默的互动，如“能8.6日”、“哈哈哈”、“三星自带搜狗”、“戳中了我的笑点”、“眼泪笑出来了”、“はちろく”、“本来准备午睡的，笑得在床上打滚”等，配有表情符号，气氛欢乐。
    print(describe_image(llm, '/Users/andersc/Downloads/images/9月下半期活动日程表.jpg'))
