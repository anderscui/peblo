# coding=utf-8
import json
import hashlib
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional

from peblo.schemas.chat import ChatSession, ChatMessage


def session_path(base_dir: Path, session_name: str) -> Path:
    return base_dir / f"{session_name}.json"


def load_session(base_dir: Path, session_name: str) -> Optional[ChatSession]:
    """Load a session by name if JSON file exists."""
    path = session_path(base_dir, session_name)
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return ChatSession.model_validate(data)


def save_session(base_dir: Path, session: ChatSession) -> None:
    """Save a session to disk."""
    path = session_path(base_dir, session.session_name)
    data = session.model_dump()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def calculate_file_hash(path: Path) -> str:
    """Compute SHA256 hash for a given file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def auto_session_name(file_hash: str) -> str:
    """Generate deterministic session name for a file hash."""
    return f"auto-{file_hash[:12]}"


def create_user_defined_session(base_dir: Path, name: str, reset: bool = False) -> ChatSession:
    existing = load_session(base_dir, name)
    if existing and not reset:
        return existing

    now = datetime.now(UTC)
    session = ChatSession(
        session_name=name,
        created_at=now,
        updated_at=now,
        mode="user-defined",
        file_hash=None,
        history=[],
    )
    save_session(base_dir, session)
    return session


def create_auto_session(base_dir: Path, file_hash: str, reset: bool = False) -> ChatSession:
    """Return an existing auto session, or create a new one."""
    name = auto_session_name(file_hash)
    existing = load_session(base_dir, name)
    if existing and not reset:
        return existing

    now = datetime.now(UTC)
    session = ChatSession(
        session_name=name,
        created_at=now,
        updated_at=now,
        mode="auto",
        file_hash=file_hash,
        history=[]
    )
    save_session(base_dir, session)
    return session


def create_ephemeral_session() -> ChatSession:
    """Ephemeral: Not saved unless explicitly requested."""
    now = datetime.now(UTC)
    return ChatSession(
        session_name=f"ephemeral-{now.timestamp()}",
        created_at=now,
        updated_at=now,
        mode="ephemeral",
        file_hash=None,
        history=[]
    )


def append_message(
    base_dir: Path,
    session: ChatSession,
    role: str,
    content: str
) -> ChatSession:
    msg = ChatMessage(role=role, content=content, timestamp=datetime.now(UTC))
    session.history.append(msg)
    session.updated_at = datetime.now(UTC)

    if session.mode != "ephemeral":
        save_session(base_dir, session)

    return session
