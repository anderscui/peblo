# coding=utf-8
from pathlib import Path

from peblo.utils.files import is_safe_text_file

MAX_FILE_SIZE = 5 * 1024 * 1024   # 5 MB
MAX_TEXT_CHARS = 10000            # Max chars sent to models


def read_text_input(input_arg: str) -> dict:
    """
    Normalize peek input (text or file).

    Returns:
        {
            "input_type": "file" | "text",
            "origin": str,
            "text": str,
            "truncated": bool,
            "file_size": int | None
        }
    """

    p = Path(input_arg)
    # ---------- Case 1: File ----------
    if p.is_file() and p.exists():
        file_size = p.stat().st_size

        if file_size > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large for peek (>{MAX_FILE_SIZE / 1024 / 1024:.1f} MB)"
            )

        if not is_safe_text_file(str(p)):
            raise ValueError("File is not a safe text file for peek")

        # read file contents
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        truncated = False
        if len(content) > MAX_TEXT_CHARS:
            content = content[:MAX_TEXT_CHARS]
            truncated = True

        return {
            "input_type": "file",
            "origin": str(p),
            "text": content.strip(),
            "truncated": truncated,
            "file_size": file_size,
        }

    # ---------- Case 2: Plain Text ----------
    text = input_arg.strip()
    truncated = False

    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
        truncated = True

    return {
        "input_type": "text",
        "origin": "inline",
        "text": text,
        "truncated": truncated,
        "file_size": None,
    }
