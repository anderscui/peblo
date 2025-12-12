# coding=utf-8
import json
import mimetypes
from pathlib import Path

mimetypes.add_type("application/toml", ".toml")


def is_safe_text_file(file_path: str | Path, sample_size: int = 1024) -> bool:
    path = Path(file_path)
    if not path.is_file():
        return False

    # 1. mimetypes
    print(str(path))
    mime, _ = mimetypes.guess_type(str(path))
    print(f'mime: {mime}')
    if mime is None:
        return False

    more_text_mimetypes = {
        "application/json",
        "application/xml",
        "application/x-yaml",
        "application/javascript",
        "text/javascript",
        "application/x-sh",
        "application/x-python",
        'application/toml',
        'application/rls-services+xml',
    }

    if not (mime.startswith("text/") or mime in more_text_mimetypes):
        return False

    # 2. detect by contents (by first `sample_size` bytes)
    try:
        with open(path, "rb") as f:
            sample = f.read(sample_size)

        # empty file -> True
        if not sample:
            return True

        if b'\x00' in sample:
            return False

        # BOM detection (UTF-16/32 etc.)
        bom_prefixes = [
            b'\xff\xfe', b'\xfe\xff',  # UTF-16
            b'\xff\xfe\x00\x00', b'\x00\x00\xfe\xff'  # UTF-32
        ]
        if any(sample.startswith(bom) for bom in bom_prefixes):
            return True

        known_encodings = ('utf-8', 'utf-16', 'gbk', 'latin-1')
        for enc in known_encodings:
            try:
                sample.decode(enc)
                return True
            except UnicodeDecodeError:
                pass

        # if encoding is unknown, check printable chars.
        printable_count = sum(1 for b in sample if 32 <= b <= 126 or b in (9, 10, 13))
        return (printable_count / len(sample)) > 0.8
    except (OSError, IOError):
        return False


def json_load(file):
    file = Path(file)
    with file.open('r', encoding='utf-8') as fin:
        return json.load(fin)


def json_dump(obj, file, ensure_ascii=False, indent=None):
    with open(file, 'w', encoding='utf-8') as fout:
        json.dump(obj, fout, ensure_ascii=ensure_ascii, indent=indent)


if __name__ == '__main__':
    assert is_safe_text_file('../text.py')
