"""Stable, page-local chunks for keyword retrieval."""

from .models import ParsedDocument, RetrievalChunk


INDEX_TEMPLATE_VERSION = "2"


def retrieval_chunks(document: ParsedDocument, fingerprint: str, source_path: str,
                     max_chars: int = 2_000, overlap: int = 200) -> list[RetrievalChunk]:
    if max_chars < 1 or overlap < 0 or overlap >= max_chars:
        raise ValueError("retrieval chunk sizes require 0 <= overlap < max_chars")
    chunks: list[RetrievalChunk] = []
    figures = {}
    for figure in document.figures:
        if figure.caption.strip():
            figures.setdefault(figure.page, []).append(figure.caption.strip())

    for page in document.pages:
        sources = [(block.kind, block.text.strip()) for block in page.blocks
                   if block.text.strip()]
        if not sources and page.text.strip():
            sources = [("text", page.text.strip())]
        sources.extend(("figure", caption) for caption in figures.get(page.number, []))
        page_index = 0
        for kind, text in sources:
            start = 0
            while start < len(text):
                end = min(start + max_chars, len(text))
                part = text[start:end].strip()
                if part:
                    page_index += 1
                    chunks.append(RetrievalChunk(
                        id=f"{fingerprint}-p{page.number}-c{page_index}",
                        fingerprint=fingerprint,
                        source_path=source_path,
                        page=page.number,
                        headings=page.headings,
                        kind=kind,
                        text=part,
                    ))
                if end == len(text):
                    break
                start = max(start + 1, end - overlap)
    return chunks
