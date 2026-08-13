import pytest

from frame.models import DocumentBlock, Figure, Page, ParsedDocument
from frame.retrieval import retrieval_chunks


def test_retrieval_chunks_stay_on_one_page_and_have_stable_ids():
    document = ParsedDocument(
        pages=[
            Page(number=1, text="ignored", headings=["Method"], blocks=[
                DocumentBlock(id="b1", kind="text", page=1, text="a" * 12),
            ]),
            Page(number=2, text="second page"),
        ],
        figures=[Figure(id="fig", page=2, caption="System overview", cache_path="x")],
        parser_version="test",
    )
    first = retrieval_chunks(document, "f" * 64, "paper.pdf", max_chars=10, overlap=2)
    second = retrieval_chunks(document, "f" * 64, "paper.pdf", max_chars=10, overlap=2)
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert all(chunk.id.startswith(("f" * 64) + "-p") for chunk in first)
    assert all(chunk.page in {1, 2} for chunk in first)
    assert any(chunk.kind == "figure" and chunk.page == 2 for chunk in first)
    assert first[0].headings == ["Method"]


def test_retrieval_chunk_sizes_are_validated():
    document = ParsedDocument(pages=[], parser_version="test")
    with pytest.raises(ValueError, match="overlap"):
        retrieval_chunks(document, "f" * 64, "paper.pdf", max_chars=10, overlap=10)


def test_evidence_ids_do_not_collide_for_same_sha_prefix():
    document = ParsedDocument(pages=[Page(number=1, text="text")], parser_version="test")
    left = retrieval_chunks(document, "a" * 64, "left.pdf")[0]
    right = retrieval_chunks(document, "a" * 12 + "b" * 52, "right.pdf")[0]
    assert left.id != right.id
