"""Page-preserving chunks for evidence extraction."""

from .models import Chunk, ParsedDocument


def chunk_document(document: ParsedDocument, max_chars: int = 12_000,
                   overlap: int = 500) -> list[Chunk]:
    if max_chars < 1 or overlap < 0 or overlap >= max_chars:
        raise ValueError("chunk sizes require 0 <= overlap < max_chars")
    chunks: list[Chunk] = []
    pending_pages: list[int] = []
    pending_parts: list[str] = []

    def flush():
        if pending_parts:
            page_label = (f"p{pending_pages[0]}" if len(pending_pages) == 1
                          else f"p{pending_pages[0]}-{pending_pages[-1]}")
            chunks.append(Chunk(
                id=f"{page_label}-c{len(chunks) + 1}",
                pages=list(pending_pages), text="\n\n".join(pending_parts),
            ))
            pending_pages.clear()
            pending_parts.clear()

    for page in document.pages:
        text = page.text.strip()
        if not text:
            continue
        labelled = f"[PDF page {page.number}]\n{text}"
        pending_length = sum(len(part) for part in pending_parts) + 2 * len(pending_parts)
        if len(labelled) <= max_chars:
            if pending_parts and pending_length + 2 + len(labelled) > max_chars:
                flush()
            pending_pages.append(page.number)
            pending_parts.append(labelled)
            continue
        flush()
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            part = text[start:end]
            chunks.append(Chunk(
                id=f"p{page.number}-c{len(chunks) + 1}",
                pages=[page.number],
                text=part,
            ))
            if end == len(text):
                break
            start = max(start + 1, end - overlap)
    flush()
    return chunks
