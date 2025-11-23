# coding=utf-8
import requests

from peblo.providers import BaseLlmProvider
from peblo.providers.registry import ProviderRegistry

ProviderRegistry.register("ollama", lambda **kwargs: OllamaProvider(**kwargs))


class OllamaProvider(BaseLlmProvider):
    def __init__(self, model="deepseek-r1:1.5b", host="http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

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
            if line:
                yield line.decode()

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
    resp = llm.chat(messages=[{'role': 'user', 'content': 'hello'}], stream=False)
    print(resp)

    print(llm.generate('1+1=?'))
