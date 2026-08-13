import json
from pathlib import Path

from frame.manifest import load_manifest
from frame.pipeline import run_batch

from conftest import (FakeParser, fact_response, install_support_model,
                      summary_response, support_response)


def _model(monkeypatch):
    def structured(config, system, user, schema):
        if schema.__name__ == "SupportBatch":
            return schema.model_validate_json(support_response(user))
        return schema.model_validate_json(
            fact_response(user) if schema.__name__ == "FactBatch" else summary_response()
        )
    monkeypatch.setattr("frame.summarize.structured_chat", structured)
    install_support_model(monkeypatch)


def test_batch_deduplicates_content_and_writes_fixed_outputs(tmp_path, config, monkeypatch):
    _model(monkeypatch)
    source = tmp_path / "input"
    (source / "nested").mkdir(parents=True)
    (source / "paper.pdf").write_bytes(b"same")
    (source / "nested" / "copy.pdf").write_bytes(b"same")
    parser = FakeParser()
    assert run_batch(config, source, parser) == 0
    manifest = load_manifest(Path(config.output_dir) / "manifest.json")
    assert len(manifest.papers) == 1
    entry = next(iter(manifest.papers.values()))
    assert entry.aliases == ["nested/copy.pdf", "paper.pdf"]
    assert entry.status == "success"
    assert entry.index_status == "success"
    assert len(parser.calls) == 1
    note = Path(config.output_dir) / entry.output_path
    assert "## 基本信息" in note.read_text()
    assert "PDF 第 1 页" in note.read_text()


def test_unchanged_success_is_skipped_and_force_reprocesses(tmp_path, config, monkeypatch):
    _model(monkeypatch)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"paper")
    parser = FakeParser()
    assert run_batch(config, pdf, parser) == 0
    assert run_batch(config, pdf, parser) == 0
    assert parser.calls == ["paper.pdf"]
    forced = type(config)(**{**config.__dict__, "force": True})
    assert run_batch(forced, pdf, parser) == 0
    assert parser.calls == ["paper.pdf", "paper.pdf"]


def test_unchanged_summary_and_index_skip_without_parse_cache(tmp_path, config, monkeypatch):
    _model(monkeypatch)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"paper")
    parser = FakeParser()
    assert run_batch(config, pdf, parser) == 0
    fingerprint = next(iter(load_manifest(Path(config.output_dir) / "manifest.json").papers))
    import shutil
    shutil.rmtree(Path(config.output_dir) / ".phelper" / "cache" / fingerprint)
    assert run_batch(config, pdf, parser) == 0
    assert parser.calls == ["paper.pdf"]


def test_partial_failure_keeps_success_and_returns_one(tmp_path, config, monkeypatch):
    _model(monkeypatch)
    source = tmp_path / "input"
    source.mkdir()
    (source / "good.pdf").write_bytes(b"good")
    (source / "bad.pdf").write_bytes(b"bad")
    assert run_batch(config, source, FakeParser({"bad.pdf"})) == 1
    manifest = load_manifest(Path(config.output_dir) / "manifest.json")
    statuses = {entry.source_path: entry.status for entry in manifest.papers.values()}
    assert statuses == {"bad.pdf": "failed", "good.pdf": "success"}
    readme = (Path(config.output_dir) / "README.md").read_text()
    assert "失败" in readme and "成功" in readme


def test_failed_paper_is_retried_on_next_run(tmp_path, config, monkeypatch):
    _model(monkeypatch)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"paper")
    assert run_batch(config, pdf, FakeParser({"paper.pdf"})) == 1
    parser = FakeParser()
    assert run_batch(config, pdf, parser) == 0
    assert parser.calls == ["paper.pdf"]


def test_model_change_reuses_parse_cache_but_regenerates_summary(tmp_path, config, monkeypatch):
    calls = []
    def structured(current, system, user, schema):
        calls.append((current.model, schema.__name__))
        if schema.__name__ == "SupportBatch":
            return schema.model_validate_json(support_response(user))
        return schema.model_validate_json(
            fact_response(user) if schema.__name__ == "FactBatch" else summary_response())
    monkeypatch.setattr("frame.summarize.structured_chat", structured)
    monkeypatch.setattr("frame.support.structured_chat", structured)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"paper")
    parser = FakeParser()
    assert run_batch(config, pdf, parser) == 0
    changed = type(config)(**{**config.__dict__, "model": "new-model"})
    assert run_batch(changed, pdf, parser) == 0
    assert parser.calls == ["paper.pdf"]
    assert any(model == "new-model" for model, _ in calls)


def test_model_change_does_not_rebuild_unchanged_index(tmp_path, config, monkeypatch):
    _model(monkeypatch)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"paper")
    parser = FakeParser()
    assert run_batch(config, pdf, parser) == 0
    monkeypatch.setattr("frame.pipeline.QAIndex.index_document",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("unchanged index was rebuilt")))
    changed = type(config)(**{**config.__dict__, "model": "new-model"})
    assert run_batch(changed, pdf, parser) == 0


def test_index_failure_keeps_summary_and_returns_one(tmp_path, config, monkeypatch):
    _model(monkeypatch)
    monkeypatch.setattr("frame.pipeline.index_document",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("index failed")))
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"paper")
    assert run_batch(config, pdf, FakeParser()) == 1
    entry = next(iter(load_manifest(Path(config.output_dir) / "manifest.json").papers.values()))
    assert entry.status == "success"
    assert entry.index_status == "failed"
    assert "index failed" in entry.index_error


def test_index_database_open_failure_does_not_block_summary(tmp_path, config, monkeypatch):
    _model(monkeypatch)
    monkeypatch.setattr("frame.pipeline.QAIndex",
                        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cannot open index")))
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"paper")
    assert run_batch(config, pdf, FakeParser()) == 1
    entry = next(iter(load_manifest(Path(config.output_dir) / "manifest.json").papers.values()))
    assert entry.status == "success"
    assert entry.index_status == "failed"
    assert "cannot open index" in entry.index_error


def test_automatic_index_checkpoints_processing_before_work(tmp_path, config, monkeypatch):
    _model(monkeypatch)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"paper")

    def inspect(index, document, entry, fingerprint, source, aliases, force=False,
                checkpoint=None):
        entry.index_status = "processing"
        checkpoint()
        saved = load_manifest(Path(config.output_dir) / "manifest.json").papers[fingerprint]
        assert saved.index_status == "processing"
        raise KeyboardInterrupt

    monkeypatch.setattr("frame.pipeline.index_document", inspect)
    try:
        run_batch(config, pdf, FakeParser())
    except KeyboardInterrupt:
        pass
    saved = next(iter(load_manifest(Path(config.output_dir) / "manifest.json").papers.values()))
    assert saved.index_status == "processing"


def test_model_error_cannot_write_api_key_to_manifest(tmp_path, config, monkeypatch):
    key = config.api_key
    monkeypatch.setattr("frame.summarize.summarize_document",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(key)))
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"paper")
    assert run_batch(config, pdf, FakeParser()) == 1
    manifest_text = (Path(config.output_dir) / "manifest.json").read_text()
    assert key not in manifest_text


def test_interrupted_manifest_entry_is_retried(tmp_path, config, monkeypatch):
    _model(monkeypatch)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"paper")
    parser = FakeParser()
    assert run_batch(config, pdf, parser) == 0
    manifest_path = Path(config.output_dir) / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    entry = next(iter(payload["papers"].values()))
    entry["status"] = "processing"
    entry["stage"] = "summarizing"
    manifest_path.write_text(json.dumps(payload))
    assert run_batch(config, pdf, parser) == 0
    assert load_manifest(manifest_path).papers[entry["fingerprint"]].status == "success"
