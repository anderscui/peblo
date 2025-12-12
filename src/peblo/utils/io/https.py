# coding=utf-8
import logging

import requests
import trafilatura
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def is_url(s: str) -> bool:
    return s is not None and s.lower().startswith(('http://', 'https://'))


def load_from_url(url: str, timeout=10.0, max_chars: int | None = None) -> str | None:
    """
    Load text contents from a URL, by BeautifulSoup.
    :param url:
    :param timeout:
    :param max_chars:
    :return:
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:145.0) Gecko/20100101 Firefox/145.0',
    }

    try:
        resp = requests.get(url, headers=headers, timeout=(5, timeout))
        resp.raise_for_status()

        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        # remove irrelevant tags
        for tag in soup.find_all(['script', 'style', 'noscript', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()

        # extract readable text
        text = soup.get_text(separator='\n', strip=True)
        text = '\n'.join([line.strip() for line in text.splitlines() if line.strip()])

        if max_chars:
            text = text[:max_chars]

        return text

    except requests.exceptions.RequestException as e:
        logger.warning(f"[URL Load] Request error: {e}")
        return None
    except Exception as e:
        logger.warning(f"[URL Load] Parse error: {e}")
        return None


def load_from_url_trafilatura(url: str, timeout: float = 10.0, max_chars: int | None = None) -> str | None:
    """
    Load text contents from a URL, by Trafilatura.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:145.0) Gecko/20100101 Firefox/145.0',
    }

    try:
        # Use requests for full control.
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()

        html = resp.text
        if not html:
            logger.warning(f'Empty response: {url}')
            return None

        # Trafilatura extract
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            include_formatting=True,
            output_format='txt'
        )

        if not text:
            logger.warning(f'Empty extraction: {url}')
            return None

        if max_chars:
            text = text[:max_chars]

        return text

    except Exception as e:
        logger.error(f'Loading URL failed: {e}')
        return None


if __name__ == '__main__':
    # url = 'https://ai.youdao.com/DOCSIRMA/html/trans/api/wbfy/index.html'
    # url = 'https://www.leoniemonigatti.com/blog.html'
    url = 'https://mp.weixin.qq.com/s/pWnGzXIybZ0ikQ-_Wz8zNA'
    print(load_from_url_trafilatura(url))
