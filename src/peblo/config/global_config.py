# coding=utf-8
import logging
import os
from pathlib import Path
import tomllib
from pydantic import BaseModel


DEFAULT_CONTEXT_LENGTH = 16 * 1024

logger = logging.getLogger(__name__)


class PebloConfig(BaseModel):
    context_length: int = DEFAULT_CONTEXT_LENGTH


def load_config() -> PebloConfig:
    """
    Priority:
    1. Config file (~/.config/peblo/config.toml)
    2. Env vars (PEBLO_CONTEXT_LENGTH)
    3. Defaults
    """
    data = {}

    # 1) load from ~/.config/peblo/config.toml
    cfg_path = Path.home() / '.config/peblo/config.toml'
    if cfg_path.exists():
        try:
            with open(cfg_path, 'rb') as f:
                toml_data = tomllib.load(f)
            data.update(toml_data.get('chat', {}))
        except Exception as e:
            logger.error(f'loading config from file failed: {e}')

    # 2) ENV overrides
    env_context = os.getenv('PEBLO_CONTEXT_LENGTH')
    if env_context:
        data['context_length'] = int(env_context)

    # 3) fill defaults
    return PebloConfig(**data)


if __name__ == '__main__':
    # os.environ['PEBLO_CONTEXT_LENGTH'] = '1000'
    print(load_config())
