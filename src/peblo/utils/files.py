# coding=utf-8
import mimetypes


def is_safe_text_file(file_path: str, sample_size: int = 1024) -> bool:
    # 1. mimetypes
    mime, _ = mimetypes.guess_type(file_path)
    if mime is None:
        return False

    more_text_mimetypes = {
        "application/json",
        "application/xml",
        "application/x-yaml",
        "application/javascript",
        "text/javascript"
    }

    if not (mime.startswith("text/") or mime in more_text_mimetypes):
        return False

    # 2. detect by contents (by first `sample_size` bytes)
    try:
        with open(file_path, "rb") as f:
            sample = f.read(sample_size)

        # empty file -> True
        if not sample:
            return True

        if b'\x00' in sample:
            return False

        known_encodings = ("utf-8", "utf-16", "gbk")
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


if __name__ == '__main__':
    assert is_safe_text_file('texts.py')
