# coding=utf-8
from pathlib import Path

import fitz
import logging

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 50000


def pdf_to_text(pdf_path: str, max_chars: int = MAX_TEXT_CHARS) -> dict:
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f'Failed to open PDF {pdf_path}: {e}')
        raise

    full_text = []
    truncated = False

    for page in doc:
        text = page.get_text()
        if not text.strip():
            logger.debug(f'Page {page.number+1} is empty or scanned')
        full_text.append(text)

    combined_text = '\n'.join(full_text).strip()

    if len(combined_text) > max_chars:
        combined_text = combined_text[:max_chars]
        truncated = True

    result = {
        'origin': pdf_path,
        'page_count': doc.page_count,
        'text': combined_text,
        'truncated': truncated,
        'used_ocr': False,
    }

    logger.info(f'Extracted text from PDF `{pdf_path}` (pages={doc.page_count}, truncated={truncated})')
    return result


def get_pdf_meta(pdf_path: str) -> dict:
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f'Failed to open PDF {pdf_path}: {e}')
        raise

    has_text_layer = any(page.get_text().strip() for page in doc)
    is_scanned = not has_text_layer

    meta = doc.metadata or {}
    title = meta.get('title')
    author = meta.get('author')
    creation_time = meta.get('creationDate')

    result = {
        'origin': pdf_path,
        'page_count': doc.page_count,
        'has_text_layer': has_text_layer,
        'is_scanned': is_scanned,
        'title': title,
        'author': author,
        'creation_time': creation_time,
    }

    logger.info(f'PDF meta for `{pdf_path}`: pages={doc.page_count}, scanned={is_scanned}')
    return result


def read_pdf_text(path: str | Path, *, start_page: int = 0, end_page: int = None, max_chars: int = None):
    """
    Read plain text from a PDF file.

    Args:
        path (str | Path): PDF file path.
        start_page: start page index (0-based).
        end_page: end page index (exclusive).
        max_chars: maximum characters to read (None = unlimited).

    Returns:
        str: Extracted text.
    """
    # , max_chars: int
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'PDF file not found: {path}')

    text_parts = []
    chars_read = 0

    with fitz.open(path) as doc:
        total_pages = len(doc)
        if end_page is None:
            end_page = total_pages
        end_page = min(end_page, total_pages)
        start_page = max(0, start_page)

        for i in range(start_page, end_page):
            page = doc[i]
            text = page.get_text('text')

            if not text:
                continue

            if max_chars is not None:
                remaining = max_chars - chars_read
                if remaining <= 0:
                    break
                if len(text) > remaining:
                    text_parts.append(text[:remaining])
                    break
                else:
                    text_parts.append(text)
                    chars_read += len(text)
            else:
                text_parts.append(text)
                chars_read += len(text)

    return '\n'.join(text_parts).strip()


if __name__ == '__main__':
    file = '/Users/andersc/Downloads/cool nlp papers/RAGAS - Automated Evaluation of Retrieval Augmented Generation （2023）.pdf'
    print(read_pdf_text(file, max_chars=1000))
