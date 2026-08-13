"""PDF parser boundary and Marker implementation."""

import json
import html
import re
from pathlib import Path
from typing import Protocol

from .errors import ParserError
from .models import DocumentBlock, Figure, Page, ParsedDocument


class PdfParser(Protocol):
    version: str

    def parse(self, pdf_path: Path, cache_dir: Path) -> ParsedDocument:
        ...


class MarkerPdfParser:
    """Use Marker JSON output while keeping Marker out of the domain layer."""

    version = "marker-2"

    def __init__(self):
        self._converter = None

    def _get_converter(self):
        if self._converter is not None:
            return self._converter
        try:
            from marker.config.parser import ConfigParser
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
        except ImportError as exc:
            raise ParserError(
                "marker-pdf is not installed; install P-Helper with Python 3.10+"
            ) from exc
        config = ConfigParser({"output_format": "json"})
        self._converter = PdfConverter(
            config=config.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config.get_processors(),
            renderer=config.get_renderer(),
        )
        return self._converter

    def parse(self, pdf_path: Path, cache_dir: Path) -> ParsedDocument:
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            rendered = self._get_converter()(str(pdf_path))
            payload = _rendered_payload(rendered)
            document = _normalise_marker(payload, cache_dir)
        except ParserError:
            raise
        except Exception as exc:
            raise ParserError(f"Marker failed to parse {pdf_path.name}: {exc}") from exc
        (cache_dir / "parsed.json").write_text(
            document.model_dump_json(indent=2), encoding="utf-8"
        )
        return document


def _rendered_payload(rendered):
    if isinstance(rendered, dict):
        return rendered
    if isinstance(rendered, str):
        return json.loads(rendered)
    for attr in ("model_dump", "dict"):
        method = getattr(rendered, attr, None)
        if method:
            return method()
    return rendered


def _children(block):
    if isinstance(block, dict):
        return block.get("children") or block.get("blocks") or []
    return []


def _walk(block):
    yield block
    for child in _children(block):
        yield from _walk(child)


def _block_text(block):
    if not isinstance(block, dict):
        return ""
    value = str(block.get("html") or block.get("markdown") or block.get("text") or "")
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", html.unescape(value)).strip()
    return re.sub(r"\s+([.,;:!?%])", r"\1", value)


def _normalise_marker(payload, cache_dir: Path) -> ParsedDocument:
    if not isinstance(payload, dict):
        payload = _rendered_payload(payload)
    root = payload.get("children") or payload.get("pages") or []
    pages: list[Page] = []
    figures: list[Figure] = []
    image_dir = cache_dir / "images"
    image_dir.mkdir(exist_ok=True)

    for index, raw_page in enumerate(root):
        page_number = _page_number(raw_page, index)
        text_parts: list[str] = []
        headings: list[str] = []
        blocks: list[DocumentBlock] = []
        for block in _walk(raw_page):
            block_type = str(block.get("block_type") or block.get("type") or "")
            text = _block_text(block).strip()
            if text and not _children(block):
                text_parts.append(text)
                blocks.append(DocumentBlock(
                    id=str(block.get("id") or f"block-{len(blocks) + 1}"),
                    kind=_block_kind(block_type), page=page_number, text=text,
                ))
            if "SectionHeader" in block_type and text:
                headings.append(text)
            if any(kind in block_type for kind in ("Figure", "Picture", "Diagram")):
                figure_id = str(block.get("id") or f"figure-{len(figures) + 1}")
                block_images = block.get("images") or {}
                image = next(iter(block_images.values()), None)
                target = _save_marker_image(image, image_dir, figure_id)
                if target:
                    figures.append(Figure(
                        id=figure_id,
                        page=page_number,
                        caption=_figure_caption(block),
                        cache_path=str(target),
                    ))
        pages.append(Page(number=page_number, text="\n\n".join(text_parts),
                          headings=headings, blocks=blocks))
    if not pages:
        raise ParserError("Marker returned no pages")
    return ParsedDocument(pages=pages, figures=figures, parser_version="marker-2")


def _page_number(raw_page, fallback_index: int) -> int:
    if "page_id" in raw_page:
        return int(raw_page["page_id"]) + 1
    match = re.search(r"/page/(\d+)/", str(raw_page.get("id", "")), re.I)
    return int(match.group(1)) + 1 if match else fallback_index + 1


def _block_kind(block_type: str) -> str:
    if "SectionHeader" in block_type:
        return "heading"
    if "Table" in block_type:
        return "table"
    return "text"


def _figure_caption(block) -> str:
    explicit = str(block.get("caption") or "").strip()
    if explicit:
        return explicit
    for child in _walk(block):
        block_type = str(child.get("block_type") or child.get("type") or "")
        if "Caption" in block_type:
            caption = _block_text(child)
            if caption:
                return caption
    return _block_text(block)


def _save_marker_image(image, directory: Path, figure_id: str):
    if image is None:
        return None
    safe_id = "".join(c if c.isalnum() or c in "-_" else "-" for c in figure_id)
    target = directory / f"{safe_id}.png"
    if hasattr(image, "save"):
        image.save(target, format="PNG")
        return target
    return None
