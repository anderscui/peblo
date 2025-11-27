# coding=utf-8
import json

import requests

from peblo.providers import BaseLlmProvider
from peblo.providers.registry import ProviderRegistry

ProviderRegistry.register("ollama", lambda **kwargs: OllamaProvider(**kwargs))


class OllamaProvider(BaseLlmProvider):
    def __init__(self, model="qwen3:4b-instruct", host="http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")
        print(f'model {self.model} initialized')

    @property
    def capabilities(self):
        return {"chat", "embed"}

    def _request(self, endpoint, payload, stream=False):
        url = f"{self.host}/{endpoint}"
        resp = requests.post(url, json=payload, stream=stream)
        resp.raise_for_status()
        return resp

    def _stream_chat(self, resp):
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode())
            data_chunk = data.get('message', {}).get('content')
            if data_chunk:
                yield data_chunk

    def chat(self, messages, stream=False):
        payload = {"model": self.model, "messages": messages, "stream": stream}
        resp = self._request("api/chat", payload, stream=stream)
        if stream:
            return self._stream_chat(resp)
        else:
            return resp.json()["message"]["content"]

    async def achat(self, messages, stream=False):
        raise NotImplementedError("Async Ollama not implemented yet")

    def embed(self, text: str):
        payload = {"model": self.model, "prompt": text}
        resp = self._request("api/embeddings", payload)
        return resp.json().get("embedding", [])


if __name__ == '__main__':
    llm = OllamaProvider()
    resp = llm.chat(messages=[{'role': 'user', 'content': 'hello，世界。'}], stream=False)
    print(resp)

    # print(llm.generate('1+1=?'))

    # for chunk in llm.chat([{'role': 'user', 'content': 'Tell me which programming language is the best.'}], stream=True):
    #     print(chunk, end='')

    # import time
    # time.sleep(10)
