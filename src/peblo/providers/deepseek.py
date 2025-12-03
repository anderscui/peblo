# coding=utf-8
import json
import logging
import requests
import os

from peblo.providers import BaseLlmProvider
from peblo.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)
ProviderRegistry.register("deepseek", lambda **kwargs: DeepSeekProvider(**kwargs))

# models
# deepseek-chat
# deepseek-reasoner


class DeepSeekProvider(BaseLlmProvider):
    def __init__(self, model='deepseek-chat', api_key=None):
        super().__init__('deepseek')

        self.model = model
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY', '')
        self.base_url = 'https://api.deepseek.com'
        logger.debug(f'{self.name} model {self.model} initialized')

    @property
    def capabilities(self):
        return {'chat'}

    def chat(self,
             messages,
             stream=False,
             extra_headers: dict = None,
             temperature: float = 1.0,
             max_tokens: int = 4096,
             thinking: bool = False):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream' if stream else 'application/json'
        }
        if extra_headers and isinstance(extra_headers, dict):
            headers.update(extra_headers)

        thinking_type = 'enabled' if thinking else 'disabled'
        payload = {
            'model': self.model,
            'messages': messages,
            'stream': stream,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'thinking': {'type': thinking_type},
        }

        resp = requests.post(
            f'{self.base_url}/chat/completions',
            headers=headers,
            json=payload,
            stream=stream,
            timeout=30
        )
        resp.raise_for_status()

        if stream:
            return self._stream_chat(resp)
        else:
            data = resp.json()
            return data['choices'][0]['message']['content']

    def _stream_chat(self, resp):
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue

            line = raw_line.strip()

            # SSE comments or keep-alive
            if line.startswith(b':'):
                continue

            # SSE data line
            if line.startswith(b'data:'):
                line = line[len(b'data:'):].strip()
                if line == b'[DONE]':
                    break

            try:
                data = json.loads(line.decode('utf-8'))
            except Exception as e:
                logger.error(f'[{self.model}] [stream] json decode error: {e}')
                continue

            if 'choices' in data:
                # not delta stream
                msg = data['choices'][0].get('message')
                if msg and msg.get('content'):
                    yield msg['content']
                    break

                # delta stream
                delta = data['choices'][0].get('delta')
                if delta and delta.get('content'):
                    yield delta['content']

    async def achat(self, messages, stream=False):
        raise NotImplementedError('Async DeepSeek not implemented yet')

    def embed(self, text: str):
        raise NotImplementedError('Embedding not implemented for DeepSeek')


if __name__ == '__main__':
    llm = DeepSeekProvider()
    # not stream
    resp = llm.chat(messages=[
        {'role': 'system', 'content': 'You are a helpful assistant'},
        {'role': 'user', 'content': 'Hello，世界。'}
    ], stream=False)
    print(resp)

    # stream
    resp_stream = llm.chat(messages=[
        {'role': 'system', 'content': 'You are a helpful assistant'},
        {'role': 'user', 'content': 'Hello，世界。'}
    ], stream=True)
    for chunk in resp_stream:
        print(chunk, end='')
