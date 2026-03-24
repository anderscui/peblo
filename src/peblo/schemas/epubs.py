# coding=utf-8
import os
from pathlib import Path

from ebooklib import epub, ITEM_DOCUMENT
from lxml import html, etree
from typing import Iterable
from pydantic import BaseModel, Field

BLOCK_TAGS = {
    'p', 'div', 'section', 'article',
    'li', 'blockquote', 'pre'
}

HEADING_TAGS = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}


def _iter_block_elements(root: etree._Element) -> Iterable[etree._Element]:
    """
    按文档顺序遍历 block-level 元素
    """
    for elem in root.iter():
        tag = elem.tag.lower() if isinstance(elem.tag, str) else ''
        if tag in BLOCK_TAGS or tag in HEADING_TAGS:
            yield elem


def _extract_text(elem: etree._Element) -> str:
    text = elem.text_content()
    if not text:
        return ''
    return text.strip()


def _resolve_anchor(tree: etree._Element, anchor: str | None) -> etree._Element | None:
    if not anchor:
        return None

    # 优先用 id
    node = tree.xpath(f'//*[@id="{anchor}"]')
    if node:
        return node[0]

    # 有些 epub 用 name
    node = tree.xpath(f'//*[@name="{anchor}"]')
    if node:
        return node[0]

    return None


def _collect_from_anchor(start_elem: etree._Element) -> list[etree._Element]:
    """
    从 anchor 开始，收集后续同级/后续节点
    （简化策略：线性收集，不严格截断到下一 section）
    """
    result = []
    collecting = False

    root = start_elem.getroottree().getroot()

    for elem in root.iter():
        if elem is start_elem:
            collecting = True

        if collecting:
            result.append(elem)

    return result


class TocNode(BaseModel):
    title: str
    href: str | None = None
    file: str | None = None
    anchor: str | None = None

    children: list['TocNode'] = Field(default_factory=list)


class EpubDocument(BaseModel):
    file: str  # file name, see also TocNode.file
    media_type: str | None = None  # content type
    content: bytes  # text of xhtml/html


class EpubImageRef(BaseModel):
    src: str
    alt: str | None = None


class EpubDocumentText(BaseModel):
    file: str
    title: str | None = None
    text: str = ''
    images: list[EpubImageRef] = Field(default_factory=list)


def split_href(href: str | None) -> tuple[str | None, str | None]:
    if not href:
        return None, None

    if '#' in href:
        file, anchor = href.split('#', 1)
        return file, anchor

    return href, None


def _toc_item_to_node(item, level: int, max_level: int) -> TocNode | None:
    if level > max_level:
        return None

    if isinstance(item, epub.Link):
        file, anchor = split_href(item.href)
        return TocNode(
            title=item.title,
            href=item.href,
            file=file,
            anchor=anchor,
        )

    if isinstance(item, epub.Section):
        href = getattr(item, 'href', None)
        file, anchor = split_href(href)
        return TocNode(
            title=item.title,
            href=href,
            file=file,
            anchor=anchor,
        )

    if isinstance(item, tuple) and len(item) == 2:
        parent, children = item

        href = getattr(parent, 'href', None)
        file, anchor = split_href(href)

        # title=getattr(parent, 'title', ''),
        node = TocNode(
            title=parent.title,
            href=href,
            file=file,
            anchor=anchor,
        )

        if level < max_level:
            for child in children:
                child_node = _toc_item_to_node(
                    child,
                    level=level + 1,
                    max_level=max_level,
                )
                if child_node is not None:
                    node.children.append(child_node)

        return node

    return None


def extract_epub_toc(epub_path: str | Path, level: int = 3) -> list[TocNode]:
    """
    Extract TOC from an EPUB file.

    Args:
        epub_path: Path to the EPUB file.
        level: Maximum TOC depth to extract.
            - 1 means only top-level nodes
            - 2 means include children of top-level nodes
            - 3 means include grandchildren
            ...

    Returns:
        A list of TocNode objects.
    """
    if level < 1:
        return []

    book = epub.read_epub(str(epub_path))
    nodes: list[TocNode] = []

    for item in book.toc:
        node = _toc_item_to_node(item, level=1, max_level=level)
        if node is not None:
            nodes.append(node)

    return nodes


def traverse_epub_toc(toc: list[TocNode], level=1, max_level=3):
    if level > max_level:
        return

    indent = '  ' * (level - 1)
    for item in toc:
        title = item.title or 'EMPTY TITLE'
        href = item.href or 'NO HREF'

        print(indent + f'level-{level}: {title}')
        print(indent + f'{href}')
        print()

        if item.children:
            traverse_epub_toc(item.children, level+1, max_level)


def flatten_epub_toc(toc: list[TocNode]) -> list[TocNode]:
    result: list[TocNode] = []

    for node in toc:
        result.append(node)
        if node.children:
            result.extend(flatten_epub_toc(node.children))

    return result


def extract_epub_documents(epub_path: str | Path) -> list[EpubDocument]:
    """
    Extract all document items from an EPUB file.

    Args:
        epub_path: Path to the EPUB file.

    Returns:
        A list of EpubDocument objects.
    """
    book = epub.read_epub(str(epub_path))
    documents: list[EpubDocument] = []

    for item in book.get_items_of_type(ITEM_DOCUMENT):
        content_bytes = item.get_content()

        # try:
        #     content = content_bytes.decode('utf-8')
        # except UnicodeDecodeError:
        #     content = content_bytes.decode('utf-8', errors='replace')

        documents.append(
            EpubDocument(
                file=item.get_name(),
                media_type=getattr(item, 'media_type', None),
                content=content_bytes,
            )
        )

    return documents


def normalize_epub_path(path: str | None) -> str | None:
    if not path:
        return None

    normalized = path.strip()
    normalized = normalized.removeprefix('./')
    return normalized


def build_epub_document_map(documents: list[EpubDocument]) -> dict[str, EpubDocument]:
    result: dict[str, EpubDocument] = {}

    for doc in documents:
        key = normalize_epub_path(doc.file)
        if key:
            result[key] = doc

    return result


def resolve_toc_node_document(node: TocNode, doc_map: dict[str, EpubDocument]) -> EpubDocument | None:
    key = normalize_epub_path(node.file)
    if not key:
        return None
    return doc_map.get(key)


def extract_text_from_epub_document(
    doc: EpubDocument,
    anchor: str | None = None,
) -> EpubDocumentText:
    tree = html.fromstring(doc.content)

    # ---------- title ----------
    title = None
    title_nodes = tree.xpath('//title')
    if title_nodes:
        raw = title_nodes[0].text_content().strip()
        title = raw or None

    # ---------- anchor ----------
    if anchor:
        start_elem = _resolve_anchor(tree, anchor)
        if start_elem is not None:
            elements = _collect_from_anchor(start_elem)
        else:
            elements = list(tree.iter())
    else:
        body_nodes = tree.xpath('//body')
        root = body_nodes[0] if body_nodes else tree
        elements = list(root.iter())

    # ---------- text extraction ----------
    lines: list[str] = []
    seen = set()

    for elem in elements:
        tag = elem.tag.lower() if isinstance(elem.tag, str) else ''

        if tag not in BLOCK_TAGS and tag not in HEADING_TAGS:
            continue

        text = _extract_text(elem)
        if not text:
            continue

        # 去重（避免嵌套重复）
        key = (tag, text)
        if key in seen:
            continue
        seen.add(key)

        if tag in HEADING_TAGS:
            lines.append(f'\n{text}\n')
        else:
            lines.append(text)

    # ---------- normalize ----------
    text = '\n'.join(lines)

    # 压缩多余空行
    text = '\n'.join(
        line.strip()
        for line in text.splitlines()
        if line.strip()
    )

    # ---------- images ----------
    images: list[EpubImageRef] = []
    for img in tree.xpath('//img'):
        src = img.get('src')
        if not src:
            continue

        normalized_src = os.path.normpath(os.path.join(os.path.dirname(doc.file), src)).replace("\\", "/")

        images.append(
            EpubImageRef(
                src=normalized_src,
                alt=img.get('alt'),
            )
        )

    return EpubDocumentText(
        file=doc.file,
        title=title,
        text=text,
        images=images,
    )


if __name__ == '__main__':
    file = ''
    toc_nodes = extract_epub_toc(file, level=3)
    flat_nodes = flatten_epub_toc(toc_nodes)

    docs = extract_epub_documents(file)
    doc_map = build_epub_document_map(docs)

    for node in flat_nodes[:20]:
        print(f'ToC: {node.title}, {node.file}, {node.anchor}')

        node_doc = resolve_toc_node_document(node, doc_map)
        if node_doc is None:
            print(f'  -> No doc found \n')
            continue

        doc_text = extract_text_from_epub_document(node_doc, node.anchor)
        print(f'  -> doc title: {doc_text.title}')
        print(f'  -> text preview: {doc_text.text[:200]}')
        if doc_text.images:
            print(f'  -> images:', [img.src for img in doc_text.images[:3]])
        print()
