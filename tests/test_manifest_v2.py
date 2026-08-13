import json

from frame.manifest import load_manifest, recover_interrupted


def test_v1_manifest_loads_with_default_index_fields(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"version": 1, "papers": {"f": {
        "fingerprint": "f", "source_path": "paper.pdf", "aliases": ["paper.pdf"],
        "status": "success", "stage": "complete", "signature": "s", "model": "m",
        "language": "zh", "parser_version": "p", "template_version": "1",
        "last_used_at": "now",
    }}}))
    manifest = load_manifest(path)
    assert manifest.version == 2
    assert manifest.papers["f"].index_status == "not_run"


def test_interrupted_index_is_retried_as_failed(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"version": 2, "papers": {"f": {
        "fingerprint": "f", "source_path": "paper.pdf", "aliases": [],
        "status": "success", "stage": "complete", "signature": "s", "model": "m",
        "language": "zh", "parser_version": "p", "template_version": "1",
        "last_used_at": "now", "index_status": "processing",
    }}}))
    manifest = load_manifest(path)
    recover_interrupted(manifest)
    assert manifest.papers["f"].index_status == "failed"
    assert "interrupted" in manifest.papers["f"].index_error
