from frame.models import DocumentBlock, Page, ParsedDocument
from frame.qa_index import QAIndex


def _document(texts, version="parser-1"):
    return ParsedDocument(
        pages=[Page(number=index, text=text, blocks=[
            DocumentBlock(id=f"b{index}", kind="text", page=index, text=text),
        ]) for index, text in enumerate(texts, 1)],
        parser_version=version,
    )


def test_index_searches_english_chinese_and_numbers(tmp_path):
    with QAIndex(tmp_path / "index.sqlite3") as index:
        index.index_document(
            _document(["Vision language action model.", "视觉语言动作模型。", "Accuracy was 91.3%."]),
            "a" * 64, "paper.pdf", ["paper.pdf"], "signature",
        )
        fingerprints = ["a" * 64]
        assert index.search(["vision"], fingerprints)[0].page == 1
        assert index.search(["语言动作"], fingerprints)[0].page == 2
        assert index.search(["91.3"], fingerprints)[0].page == 3


def test_index_rebuild_replaces_chunks_without_duplicates(tmp_path):
    with QAIndex(tmp_path / "index.sqlite3") as index:
        index.index_document(_document(["old term"]), "a" * 64, "old.pdf", ["old.pdf"], "s1")
        index.index_document(_document(["new term"]), "a" * 64, "new.pdf", ["new.pdf"], "s2")
        assert not index.search(["old"], ["a" * 64])
        results = index.search(["new"], ["a" * 64])
        assert len(results) == 1 and results[0].source_path == "new.pdf"
        assert index.has_signature("a" * 64, "s2")


def test_search_limits_each_paper_to_three_chunks(tmp_path):
    with QAIndex(tmp_path / "index.sqlite3") as index:
        index.index_document(_document([f"shared keyword page {i}" for i in range(6)]),
                             "a" * 64, "a.pdf", ["a.pdf"], "a")
        index.index_document(_document([f"shared keyword other {i}" for i in range(2)]),
                             "b" * 64, "b.pdf", ["b.pdf"], "b")
        results = index.search(["shared keyword"], ["a" * 64, "b" * 64], top_k=8)
        counts = {}
        for result in results:
            counts[result.fingerprint] = counts.get(result.fingerprint, 0) + 1
        assert counts == {"a" * 64: 3, "b" * 64: 2}


def test_paper_selectors_support_source_alias_and_unique_prefix(tmp_path):
    with QAIndex(tmp_path / "index.sqlite3") as index:
        index.index_document(_document(["alpha"]), "abc111" + "0" * 58,
                             "one/a.pdf", ["one/a.pdf", "copy.pdf"], "a")
        index.index_document(_document(["beta"]), "abc222" + "0" * 58,
                             "two/b.pdf", ["two/b.pdf"], "b")
        assert index.resolve_papers(["copy.pdf"]) == ["abc111" + "0" * 58]
        assert index.resolve_papers(["abc222"]) == ["abc222" + "0" * 58]
        try:
            index.resolve_papers(["abc"])
        except ValueError as exc:
            assert "ambiguous" in str(exc)
        else:
            raise AssertionError("ambiguous selector was accepted")


def test_metadata_changes_do_not_require_reindexing_text(tmp_path):
    with QAIndex(tmp_path / "index.sqlite3") as index:
        index.index_document(_document(["alpha term"]), "a" * 64,
                             "old.pdf", ["old.pdf"], "signature")
        index.update_metadata("a" * 64, "new.pdf", ["new.pdf", "copy.pdf"])
        assert index.resolve_papers(["copy.pdf"]) == ["a" * 64]
        result = index.search(["alpha"], ["a" * 64])[0]
        assert result.source_path == "new.pdf"


def test_removing_a_paper_makes_stale_chunks_unqueryable(tmp_path):
    with QAIndex(tmp_path / "index.sqlite3") as index:
        index.index_document(_document(["stale term"]), "a" * 64,
                             "paper.pdf", ["paper.pdf"], "signature")
        assert index.search(["stale"], ["a" * 64])
        index.remove_paper("a" * 64)
        assert not index.search(["stale"], ["a" * 64])
        assert index.resolve_papers([]) == []
