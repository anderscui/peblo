# coding=utf-8
import logging
from typing import Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field

from peblo.commons.llm.tokenizers import get_tokenizer, count_tokens

MAX_ROUNDS = 20

logger = logging.getLogger(__name__)


class MessageRoles:
    system = 'system'
    assistant = 'assistant'
    user = 'user'


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: Optional[datetime] = None


class ChatSession(BaseModel):
    session_name: str

    created_at: datetime
    updated_at: Optional[datetime] = None

    # "user-defined", "auto", "ephemeral"
    mode: Literal["user-defined", "auto", "ephemeral"]

    file_hash: Optional[str] = Field(
        None,
        description="Only used when mode = auto to ensure same file content"
    )

    history: list[ChatMessage] = Field(default_factory=list)

    def to_dict_messages(self, system_prompt: str | None = None) -> list[dict]:
        messages = []
        if system_prompt:
            messages.append({'role': MessageRoles.system, 'content': system_prompt})
        messages.extend({'role': msg.role, 'content': msg.content} for msg in self.history)
        print('messages:', [msg['content'] for msg in messages])
        return messages

    def to_provider_messages(self, max_rounds: int = MAX_ROUNDS):
        """Convert to list[dict] for llm providers, applying sliding window truncation."""

        system_msgs = [msg for msg in self.history if msg.role == MessageRoles.system]
        other_msgs = [msg for msg in self.history if msg.role != MessageRoles.system]

        # first, always include all system messages
        messages = system_msgs[:]
        remaining = max_rounds - len(messages)
        if remaining > 0:
            messages.extend(other_msgs[-remaining:])
        return [{'role': msg.role, 'content': msg.content} for msg in messages]

    def to_provider_messages_token_window(
            self,
            model: str,
            max_tokens: int
    ):
        """
        Convert chat history to provider message format (list[dict]),
        truncating by token length with a sliding window.
        """
        tokenizer = get_tokenizer(model)

        system_msgs = [m for m in self.history if m.role == MessageRoles.system]
        other_msgs = [m for m in self.history if m.role != MessageRoles.system]

        # first, always include all system messages
        messages = system_msgs[:]
        used = count_tokens(system_msgs, tokenizer)

        # sliding window for the rest
        reversed_msgs = list(reversed(other_msgs))
        kept = []

        for msg in reversed_msgs:
            msg_tokens = count_tokens([msg], tokenizer)
            if used + msg_tokens > max_tokens:
                break
            kept.append(msg)
            used += msg_tokens

        if not kept:
            # system prompt is too long
            logger.warning(f'system prompt is too long, max tokens allowed: {max_tokens}, system: {used}')

        # restore correct order
        kept.reverse()
        messages.extend(kept)

        # convert to provider format
        converted = [{'role': m.role, 'content': m.content} for m in messages]
        print(converted)
        return converted
