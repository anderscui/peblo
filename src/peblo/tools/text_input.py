# coding=utf-8
import re
from pathlib import Path

from peblo.tools.summarize import simple_summarizer
from peblo.utils.files import is_safe_text_file
from peblo.utils.io.pdfs import read_pdf_text

MAX_FILE_SIZE = 5 * 1024 * 1024   # 5 MB
MAX_TEXT_CHARS = 5 * 1024 * 1024  # Max chars sent to models


_CONTROL_CHARS = ''.join(
    chr(c) for c in range(32)
    if c not in (9, 10, 13)  # keep \t, \n, \r
)
_CONTROL_CHAR_RE = re.compile(f"[{re.escape(_CONTROL_CHARS)}]")


def clean_text(text: str) -> str:
    """Remove invisible control chars and normalize line endings."""
    text = text.replace("\ufeff", "")  # remove BOM
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHAR_RE.sub("", text)
    return text.strip()


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


def read_plain_text(path: str | Path, max_chars: int | None = None) -> str:
    """
    Read plain text from a file with multi-encoding fallback,
    and optionally limit by character count.
    """
    def _read_text_with_encoding(path: Path, encoding: str):
        """Internal: stream-read file with given encoding and apply character limit."""
        if max_chars is None:
            return clean_text(path.read_text(encoding=encoding))

        chars_read = 0
        chunks = []

        with path.open("r", encoding=encoding, errors="strict") as f:
            for line in f:
                line_len = len(line)

                if chars_read + line_len > max_chars:
                    remain = max_chars - chars_read
                    if remain > 0:
                        chunks.append(line[:remain])
                    break

                chunks.append(line)
                chars_read += line_len

        return clean_text("".join(chunks))


    path = Path(path)

    encodings = ["utf-8", "utf-8-sig", "gb18030", "latin-1"]
    for enc in encodings:
        try:
            return _read_text_with_encoding(path, enc)
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Cannot decode {path}")


def read_file_text(path: str | Path, max_chars: int | None = None):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if is_safe_text_file(path):
        return read_plain_text(path, max_chars=max_chars)
    elif suffix == '.pdf':
        return read_pdf_text(path, max_chars=max_chars)
    # elif suffix == '.docx':
    #     return read_docx_text(path, max_chars)
    # elif suffix == '.epub':
    #     return read_epub_text(path, max_chars)
    # elif suffix == '.pptx':
    #     return read_pptx_text(path, max_chars)

    raise ValueError(f'Unsupported file type: {suffix}')


def load_context_file(path: str | Path,
                      max_file_size=MAX_FILE_SIZE,
                      max_text_size=MAX_TEXT_CHARS,
                      summarizer=None) -> str:
    """Return a context text (maybe summarized)."""

    # Step 1: File size limit
    path = Path(path)
    size = path.stat().st_size
    if size > max_file_size:
        raise ValueError(
            f"File too large: {size/1024/1024:.1f}MB > {max_file_size/1024/1024:.1f}MB"
        )

    # Step 2: Read file as text
    try:
        text = read_file_text(path)  # your own function
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}")

    # Step 3: Limit final text size
    if len(text) <= max_text_size:
        return text

    # Step 4: Summarize if too large
    if summarizer is None:
        raise ValueError(
            f"Text too large ({len(text)} chars) and no summarizer provided."
        )

    summary = summarizer(text, max_text_size)
    return (
        "The original document was too large; this is an auto-generated summary:\n\n"
        + summary
    )


if __name__ == '__main__':
    file = './qa.py'
    # print(read_file_text(file, max_chars=101))

    # print(load_context_file(file))
    print(load_context_file(file, max_text_size=101, summarizer=simple_summarizer))
