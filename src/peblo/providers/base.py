# coding=utf-8
import math
from abc import ABC, abstractmethod
from typing import Any


class BaseProvider(ABC):
    """Top-level abstract provider."""

    @property
    def capabilities(self) -> set[str]:
        return set()


class BaseLlmProvider(BaseProvider):
    """Abstract base provider for all LLM backends."""

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], stream: bool = False) -> Any:
        raise NotImplementedError

    @abstractmethod
    async def achat(self, messages: list[dict[str, str]], stream: bool = False) -> Any:
        raise NotImplementedError

    def generate(self, text: str) -> str:
        messages = [{"role": "user", "content": text}]
        return self.chat(messages, stream=False)

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def count_tokens(self, text: str, bytes_per_token: int = 4) -> int:
        """
        Estimate token count by UTF-8 length.
        Works for most LLMs (OpenAI, Anthropic, Gemini, Ollama models, etc.)

        :param text: Input text
        :param bytes_per_token: Average bytes per token (default 4)
        :return: Estimated token count
        """
        if not text:
            return 0

        length = len(text.encode("utf-8"))
        return math.ceil(length / bytes_per_token)
