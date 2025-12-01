# coding=utf-8
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
