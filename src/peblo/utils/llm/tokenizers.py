# coding=utf-8
import tiktoken

_tokenizer_cache = {}

import re


class SimpleTokenizer:
    """
    Tokenization rules:
    - English letters & digits: consecutive sequences become ONE token
    - Everything else: each character is ONE token
    """
    def encode(self, text: str):
        tokens = re.findall(r'[A-Za-z0-9]+|\s+|.', text, flags=re.DOTALL)
        return tokens

    def decode(self, tokens):
        return ''.join(tokens)


def get_tokenizer(model: str):
    """
    Try to get a tokenizer for the model.
    - If model contains "gpt", "o3", "o1", use tiktoken
    - Otherwise fallback to a simple whitespace tokenizer
    """

    key = model.lower()
    if key in _tokenizer_cache:
        return _tokenizer_cache[key]

    # models like openai/gpt-5
    if '/' in model:
        model = model.rsplit('/', maxsplit=1)[-1]

    # OpenAI-like models
    if any(x in key for x in ['gpt', 'o3', 'o1']):
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception as e:
            enc = tiktoken.get_encoding('cl100k_base')
        _tokenizer_cache[key] = enc
        return enc

    # enc = SimpleTokenizer()
    enc = tiktoken.get_encoding('cl100k_base')
    _tokenizer_cache[key] = enc
    return enc


def count_tokens(messages, tokenizer) -> int:
    """
    Count tokens in a list of ChatMessage.
    Rough but reliable enough for context truncation.
    """
    total = 0
    for msg in messages:
        text = f'{msg.role}:{msg.content}\n'
        total += len(tokenizer.encode(text))
    return total


if __name__ == '__main__':
    s = 'ChatGPT   和 Gemini 真不错，version3.1很好。'
    # t = SimpleTokenizer()
    # print(t.encode("ChatGPT   和 Gemini 真不错，version3.1很好。"))

    t = get_tokenizer('openai/gpt-5')
    print(t.encode(s))
    for token_id in t.encode(s):
        print(t.decode([token_id]))
