# coding=utf-8
import json
import logging
import requests
import os

from peblo.providers import BaseLlmProvider
from peblo.providers.registry import ProviderRegistry
from peblo.schemas.models import ModelInfo, PricingInfo

logger = logging.getLogger(__name__)
ProviderRegistry.register("openrouter", lambda **kwargs: OpenRouterProvider(**kwargs))


class OpenRouterProvider(BaseLlmProvider):
    def __init__(self, model='openai/gpt-4o-mini', api_key=None):
        super().__init__('openrouter')

        self.model = model
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY', '')
        self.base_url = 'https://openrouter.ai/api/v1'
        logger.debug(f'{self.name} model {self.model} initialized')

    @property
    def capabilities(self):
        return {'chat'}

    def chat(self, messages, stream=False, extra_headers: dict=None):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream' if stream else 'application/json'
        }
        if extra_headers is not None and isinstance(extra_headers, dict):
            headers.update(extra_headers)

        payload = {
            'model': self.model,
            'messages': messages,
            'stream': stream
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

            # If SSE is used
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
        raise NotImplementedError('Async OpenRouter not implemented yet')

    def embed(self, text: str):
        raise NotImplementedError(f'Embedding not implemented for {self.name}')

    def list_models(self) -> list[ModelInfo]:
        def _parse_pricing(pricing: dict[str, str]) -> PricingInfo:
            return PricingInfo(
                unit='1M',
                input=PricingInfo.norm_price_per_token(pricing.get('prompt')),
                output=PricingInfo.norm_price_per_token(pricing.get('completion')),
            )

        def _parse_caps(raw_model_id: str, all_mods: list[str]):
            caps = set()
            model_id_lower = raw_model_id.lower()

            if any(kw in model_id_lower for kw in ['embed', 'vector', 'bge']):
                caps.add('embedding')

            if (any(kw in model_id_lower for kw in ['vision', 'multimodal'])
                    or any(m in {'image', 'video'} for m in all_mods)):
                caps.add('vision')

            # reasoning (R1 / o1 / deepseek-r1 etc.)
            if any(kw in model_id_lower for kw in ['r1', 'reason', 'thinking', 'o1']):
                caps.add('reasoning')

            # function / tool
            if any(kw in model_id_lower for kw in ['tool', 'function', 'json']):
                caps.update({'tool', 'function_call'})

            # ---- chat fallback ----
            if not caps or caps <= {'vision'}:
                caps.add('chat')

            return sorted(caps)

        try:
            headers = {'Authorization': f'Bearer {self.api_key}'}
            resp = requests.get(f'{self.base_url}/models', headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            models = []
            for model in data.get('data', []):
                model_id = model['id']
                name = model['name']

                arch = model.get('architecture', {})
                input_mods = arch.get('input_modalities', [])
                output_mods = arch.get('output_modalities', [])
                modality = '+'.join(input_mods) + '->' + '+'.join(output_mods) if input_mods or output_mods else None
                all_mods = input_mods + output_mods

                caps = _parse_caps(model_id, all_mods)

                models.append(ModelInfo(
                    id=f'{self.name}:{model_id}',
                    name=name,
                    description=model.get('description'),
                    modified_at=model.get('created'),

                    family=model.get('family'),

                    parameter_size=model.get('parameter_size'),
                    context_length=model.get('context_length'),
                    modality=modality,
                    input_modality=input_mods,
                    output_modality=output_mods,
                    tokenizer=arch.get('tokenizer'),
                    disk_size=None,

                    pricing=_parse_pricing(model.get('pricing', {})),

                    providers=[self.name],
                    capabilities=caps,
                    supported_parameters=model.get('supported_parameters', [])
                ))

            return models
        except Exception as e:
            logger.error(f'{self.name} list models failed: {e}')
            return []


class OpenRouterModels:
    # gpt_5_chat = 'openai/gpt-5-chat'  # 不建议通过 API 调用
    gpt_5_1 = 'openai/gpt-5.1'  # $1.25-10, $10/K web search
    gpt_5 = 'openai/gpt-5'  # coding  # $1.25-10
    gpt_5_mini = 'openai/gpt-5-mini'  # $0.25-2
    gpt_5_nano = 'openai/gpt-5-nano'  # $0.05-0.4
    gpt_4_1 = 'openai/gpt-4.1'  # $2-8
    gpt_4_1_mini = 'openai/gpt-4.1-mini'  # translation, $0.4-1.6
    gpt_4_1_nano = 'openai/gpt-4.1-nano'  # translation, $0.1-0.4
    gpt_4o = 'openai/gpt-4o-2024-11-20'
    gpt_4o_mini = 'openai/gpt-4o-mini'  # $0.15-0.6

    gpt_5_pro = 'openai/gpt-5-pro'  # $15-120, 2025.10
    gpt_5_codex = 'openai/gpt-5-codex'  # $1.25-10, 2025.09
    gpt_4o_audio = 'openai/gpt-4o-audio-preview'  # $2.5-10, 2025.08

    gpt_5_image = 'openai/gpt-5-image'  # $10-10, $10/K web search, 2025.10
    gpt_5_image_mini = 'openai/gpt-5-image-mini'  # $2.50-2, $10/K web search, 2025.10
    gpt_codex_mini = 'openai/codex-mini'  # $1.5-6

    gpt_oss_20b = 'openai/gpt-oss-20b'
    gpt_oss_120b = 'openai/gpt-oss-120b'
    gpt_o1 = 'openai/o1'  # $15-60
    gpt_o3 = 'openai/o3'  # $2-8
    gpt_o3_mini = 'openai/o3-mini'  # $1.1-4.4
    gpt_o4_mini = 'openai/o4-mini'  # $1.1-4.4

    gpt_4o_search = 'openai/gpt-4o-search-preview'
    gpt_4o_mini_search = 'openai/gpt-4o-mini-search-preview'

    gpt_o3_deep_research = 'openai/o3-deep-research'  # $10-40, 2025.10
    gpt_o4_mini_deep_research = 'openai/o4-mini-deep-research'  # $2-8, 2025.10

    # embeddings
    openai_emb_3_large = 'openai/text-embedding-3-large'  # $0.13
    openai_emb_3_small = 'openai/text-embedding-3-small'  # $0.02
    openai_emb_2_ada = 'openai/text-embedding-ada-002' # $0.10

    claude_opus_4_1 = 'anthropic/claude-opus-4.1'  # coding, very expensive
    claude_opus_4 = 'anthropic/claude-opus-4'  # coding, very expensive
    claude_sonnet_4 = 'anthropic/claude-sonnet-4'  # image, coding, $3-15, 2025.05
    claude_sonnet_4_5 = 'anthropic/claude-sonnet-4.5'  # coding, $3-15, 2025.09
    claude_sonnet_3_7 = 'anthropic/claude-3.7-sonnet'
    claude_haiku_3_5 = 'anthropic/claude-3.5-haiku'  # fastest model for daily tasks
    claude_haiku_4_5 = 'anthropic/claude-haiku-4.5'  # coding, $1-5, 2025.10

    # google: translation, coding
    gemini_flash_lite_2_5 = 'google/gemini-2.5-flash-lite'  # cheap, 2025.07
    gemini_flash_2_5 = 'google/gemini-2.5-flash'  # text, image; translation, $0.3-2.5
    gemini_pro_2_5 = 'google/gemini-2.5-pro'  # text, image; translation, $1.25-10
    gemini_flash_2 = 'google/gemini-2.0-flash-001'  # translation, $0.1-0.4
    gemini_flash_lite_2 = 'google/gemini-2.0-flash-lite-001'  # $0.075-0.3

    gemini_flash_2_5_image = 'google/gemini-2.5-flash-image'  # $0.3-2.5, Nano Banan, 2025.10

    gemini_emb_1 = 'google/gemini-embedding-001'  # $0.15, 2025.03

    # qwen
    qwen3_max = 'qwen/qwen3-max'  # 1.2-6
    qwen3_plus = 'qwen/qwen-plus'  # 0.4-1.2
    qwen3_vl_235b_thinking = 'qwen/qwen3-vl-235b-a22b-thinking'  # 0.3-1.2
    qwen3_vl_235b_instruct = 'qwen/qwen3-vl-235b-a22b-instruct'  # coding, 0.22-0.88
    qwen3_vl_32b_instruct = 'qwen/qwen3-vl-32b-instruct'

    qwen3_coder_plus = 'qwen/qwen3-coder-plus'  # $1-5, 2025.09
    qwen3_coder = 'qwen/qwen3-coder'  # $0.22-0.95, 2025.07, Qwen3-Coder-480B-A35B-Instruct
    qwen3_coder_flash = 'qwen/qwen3-coder-flash'  # coding, $0.3-15, 2025.09

    qwen3_emb_8b = 'qwen/qwen3-embedding-8b'  # $0.01
    qwen3_emb_4b = 'qwen/qwen3-embedding-4b'  # $0.02
    qwen3_emb_06b = 'qwen/qwen3-embedding-0.6b'  # $0.01

    deepseek_v3_2 = 'deepseek/deepseek-v3.2'  # $0.27-0.4
    deepseek_v3_1 = 'deepseek/deepseek-chat-v3.1'  # 671B, MoE model, $0.2-0.8
    deepseek_v3_0324 = 'deepseek/deepseek-chat-v3-0324'  # 685B, MoE model
    kimi_k2 = 'moonshotai/kimi-k2'  # coding
    glm_4_6 = 'z-ai/glm-4.6'  # coding, $0.4-1.75
    glm_4_5v = 'z-ai/glm-4.5v'  # vision
    minimax_m2 = 'minimax/minimax-m2'  # $0.255-1.02, coding, 2025.10

    grok_4 = 'x-ai/grok-4'  # $3-15
    grok_4_fast = 'x-ai/grok-4-fast'  # $0.2-0.5
    grok_code_fast_1 = 'x-ai/grok-code-fast-1'  # coding, translation, $0.2-1.5

    mistral_small_3_instruct = 'mistralai/mistral-small-24b-instruct-2501'  # 24B, $0.05-0.08
    mistral_small_3_2_instruct = 'mistralai/mistral-small-3.2-24b-instruct'  # 24B, $0.06-0.18
    mistral_nemo = 'mistralai/mistral-nemo'  # 12B, $0.02-0.04


if __name__ == '__main__':
    llm = OpenRouterProvider(OpenRouterModels.gemini_flash_lite_2_5)
    # resp = llm.chat(messages=[{'role': 'user', 'content': 'hello，世界。'}], stream=False)
    # print(resp)
    #
    # resp = llm.chat(messages=[{'role': 'user', 'content': 'hello，世界。'}], stream=True)
    # for chunk in resp:
    #     print(chunk, end='')

    # for m in llm.list_models()[:1000]:
    #     print(m)
    #     print()
