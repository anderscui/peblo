# coding=utf-8
from peblo.providers import BaseLlmProvider
from peblo.commons.culture import detect_lang


def translate_text(provider: BaseLlmProvider, text: str, target_lang: str = 'zh'):
    lang_map = {
        'zh': 'Chinese',
        'en': 'English',
        'ja': 'Japanese',
        'ko': 'Korean',
        'es': 'Spanish',
        'pt': 'Portuguese',
        'de': 'German',
        'fr': 'French',
        'it': 'Italian',
        # 'ru': 'Russian',
        # 'hi': 'Hindi',
        # 'bn': 'Bengali',
        # 'vi': 'Vietnamese',
        # 'id': 'Indonesian',
        # 'ar': 'Arabic',
    }

    src_lang = detect_lang(text)
    if src_lang == target_lang:
        return {
            'translation': text,
            'skipped': True,
            'src_lang': src_lang
        }

    target_lang_name = lang_map.get(target_lang)
    if target_lang_name is None:
        raise ValueError(f'Unsupported target lang: {target_lang}. Use one of [{', '.join(lang_map.keys())}]')

    prompt = f"""Translate the following text into {target_lang_name}.
Keep the original meaning. Do not add explanations.

Text:
{text}
"""

    messages = [
        {"role": "user", "content": prompt}
    ]

    result = provider.chat(messages)
    return {
        "translation": result.strip(),
        'skipped': False,
        'src_lang': src_lang
    }


if __name__ == '__main__':
    from peblo.providers import OllamaProvider

    llm = OllamaProvider(model='qwen3-vl:8b-instruct')
    print(translate_text(llm, 'hello, world!'))
    print(translate_text(llm, '你好哇，李银河!'))  # skipped
    print(translate_text(llm, '東京の天気はどうですか？!', target_lang='en'))
    print(translate_text(llm, '東京の天気はどうですか？!', target_lang='unknown'))  # error
