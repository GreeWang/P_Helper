from frame.chunking import chunk_document
from frame.models import Page, ParsedDocument
import pytest


def test_chunks_never_cross_pdf_pages():
    document = ParsedDocument(
        pages=[Page(number=1, text="a" * 20), Page(number=2, text="b" * 5)],
        parser_version="test",
    )
    chunks = chunk_document(document, max_chars=10, overlap=2)
    assert [chunk.pages for chunk in chunks] == [[1], [1], [1], [2]]
    assert chunks[1].text.startswith("aa")


def test_empty_pages_are_skipped():
    document = ParsedDocument(pages=[Page(number=3, text="   ")], parser_version="test")
    assert chunk_document(document) == []


def test_invalid_chunk_overlap_is_rejected():
    document = ParsedDocument(pages=[Page(number=1, text="x")], parser_version="test")
    with pytest.raises(ValueError, match="overlap"):
        chunk_document(document, max_chars=10, overlap=10)


def test_short_complete_pages_are_combined_with_page_labels():
    document = ParsedDocument(
        pages=[Page(number=2, text="alpha"), Page(number=3, text="beta")],
        parser_version="test",
    )
    chunks = chunk_document(document, max_chars=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].pages == [2, 3]
    assert "[PDF page 2]\nalpha" in chunks[0].text
    assert "[PDF page 3]\nbeta" in chunks[0].text
