# coding=utf-8
from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


class PricingInfo(BaseModel):
    unit: Literal['1K', '1M'] = '1M'
    input: Optional[float] = None
    output: Optional[float] = None
    currency: str = 'USD'


class ModelInfo(BaseModel):
    """
    Schema for various LLM models.
    """
    id: str                          # openai/gpt-4o-mini
    name: str                        # GPT-4o Mini
    description: Optional[str] = None
    modified_at: Optional[datetime] = None

    # Model family identifier normalized by Peblo,
    # e.g. 'gpt-5', 'deepseek-v3', 'qwen3'
    family: Optional[str] = None

    parameter_size: Optional[str] = None  # '8B', '30B', etc.
    context_length: Optional[int] = None
    modality: list[Literal['text', 'image', 'audio', 'video']] = Field(default_factory=lambda: ['text'])
    tokenizer: Optional[str] = None
    disk_size: Optional[int] = None  # for local models only

    pricing: Optional[PricingInfo] = None

    providers: list[str]             # ['openai', 'openrouter']
    capabilities: list[Literal[
        'chat',
        'vision',
        'function_call',
        'embedding',
        'reasoning',
        'tool'
    ]] = Field(default_factory=lambda: ['chat'])


if __name__ == '__main__':
    model = ModelInfo()
