from pathlib import Path

from frame.indexing import build_index
from frame.manifest import Manifest, load_manifest, save_manifest
from frame.pipeline import run_batch

from conftest import FakeParser, fact_response, install_support_model, summary_response


def _model(monkeypatch):
    def structured(config, system, user, schema):
        return schema.model_validate_json(
            fact_response(user) if schema.__name__ == "FactBatch" else summary_response()
        )
    monkeypatch.setattr("frame.summarize.structured_chat", structured)
    install_support_model(monkeypatch)


def test_explicit_index_reuses_parse_cache_and_force_rebuilds(tmp_path, config, monkeypatch):
    _model(monkeypatch)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"paper")
    parser = FakeParser()
    assert run_batch(config, pdf, parser) == 0
    output = Path(config.output_dir)
    entry = next(iter(load_manifest(output / "manifest.json").papers.values()))
    entry.index_status = "failed"
    manifest = load_manifest(output / "manifest.json")
    manifest.papers[entry.fingerprint].index_status = "failed"
    from frame.manifest import save_manifest
    save_manifest(output / "manifest.json", manifest)

    fresh = FakeParser()
    assert build_index(pdf, output, parser=fresh) == 0
    assert fresh.calls == []
    assert load_manifest(output / "manifest.json").papers[entry.fingerprint].index_status == "success"
    assert build_index(pdf, output, force=True, parser=fresh) == 0
    assert fresh.calls == []


def test_explicit_index_failure_is_retryable(tmp_path, config, monkeypatch):
    _model(monkeypatch)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"paper")
    assert run_batch(config, pdf, FakeParser()) == 0
    output = Path(config.output_dir)
    original = __import__("frame.indexing", fromlist=["QAIndex"]).QAIndex.index_document
    monkeypatch.setattr("frame.indexing.QAIndex.index_document",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("broken index")))
    assert build_index(pdf, output, force=True, parser=FakeParser()) == 1
    entry = next(iter(load_manifest(output / "manifest.json").papers.values()))
    assert entry.status == "success" and entry.index_status == "failed"
    monkeypatch.setattr("frame.indexing.QAIndex.index_document", original)
    assert build_index(pdf, output, parser=FakeParser()) == 0
    entry = next(iter(load_manifest(output / "manifest.json").papers.values()))
    assert entry.index_status == "success"


def test_explicit_index_preserves_output_language(tmp_path, config, monkeypatch):
    _model(monkeypatch)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"paper")
    assert run_batch(config, pdf, FakeParser()) == 0
    output = Path(config.output_dir)
    manifest = load_manifest(output / "manifest.json")
    for entry in manifest.papers.values():
        entry.language = "en"
    from frame.manifest import save_manifest
    save_manifest(output / "manifest.json", manifest)
    assert build_index(pdf, output, force=True, parser=FakeParser()) == 0
    assert "QA index" in (output / "README.md").read_text()


def test_explicit_index_syncs_new_aliases_to_manifest(tmp_path, config, monkeypatch):
    _model(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    (source / "paper.pdf").write_bytes(b"paper")
    assert run_batch(config, source, FakeParser()) == 0
    (source / "copy.pdf").write_bytes(b"paper")
    output = Path(config.output_dir)
    assert build_index(source, output, parser=FakeParser()) == 0
    entry = next(iter(load_manifest(output / "manifest.json").papers.values()))
    assert entry.aliases == ["copy.pdf", "paper.pdf"]
    from frame.qa_index import QAIndex
    with QAIndex(output / ".phelper" / "qa" / "index.sqlite3") as index:
        assert index.resolve_papers(["copy.pdf"]) == [entry.fingerprint]


def test_failed_rebuild_removes_stale_index_from_queries(tmp_path, config, monkeypatch):
    _model(monkeypatch)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"paper")
    assert run_batch(config, pdf, FakeParser()) == 0
    output = Path(config.output_dir)
    entry = next(iter(load_manifest(output / "manifest.json").papers.values()))
    monkeypatch.setattr("frame.indexing.QAIndex.index_document",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("broken index")))
    assert build_index(pdf, output, force=True, parser=FakeParser()) == 1
    failed = load_manifest(output / "manifest.json").papers[entry.fingerprint]
    assert failed.index_status == "failed"
    assert failed.index_signature is None and failed.indexed_at is None
    from frame.qa_index import QAIndex
    with QAIndex(output / ".phelper" / "qa" / "index.sqlite3") as index:
        assert entry.fingerprint not in index.resolve_papers([])


def test_explicit_index_database_open_failure_is_recorded(tmp_path, config, monkeypatch):
    _model(monkeypatch)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"paper")
    assert run_batch(config, pdf, FakeParser()) == 0
    output = Path(config.output_dir)
    monkeypatch.setattr("frame.indexing.QAIndex",
                        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cannot open index")))
    assert build_index(pdf, output, force=True, parser=FakeParser()) == 1
    entry = next(iter(load_manifest(output / "manifest.json").papers.values()))
    assert entry.status == "success"
    assert entry.index_status == "failed"
    assert entry.index_signature is None
    assert "cannot open index" in entry.index_error


def test_index_database_open_failure_with_unknown_paper_returns_failure(
        tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"paper")
    output = tmp_path / "summaries"
    output.mkdir()
    save_manifest(output / "manifest.json", Manifest())
    monkeypatch.setattr(
        "frame.indexing.QAIndex",
        lambda *_: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )

    assert build_index(pdf, output, parser=FakeParser()) == 1
