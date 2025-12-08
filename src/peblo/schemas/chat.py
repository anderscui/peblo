# coding=utf-8
from typing import Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field


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
