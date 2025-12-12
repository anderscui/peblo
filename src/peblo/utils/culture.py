# coding=utf-8
from langdetect import detect

def detect_lang(text: str, use_iso639: bool=True) -> str:
    lang = detect(text)
    if lang is not None and use_iso639:
        return 'zh' if lang in {'zh-cn', 'zh-tw'} else lang
    return lang


if __name__ == '__main__':
    cases = [('Hello, world!', 'en'),
             ('你好，世界', 'zh'),
             ('不要问我从哪里来', 'zh'),
             ('こんにちは', 'ja'),
             ('Bonjour, comment ça va?', 'fr'),
             ('Hello 世界', None)]
    for text, expected in cases:
        detected = detect_lang(text)
        if detected != expected:
            print(f'{expected} -> {detected}: {text}')
