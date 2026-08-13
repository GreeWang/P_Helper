import time
from pathlib import Path

from fastapi.testclient import TestClient

from frame.manifest import Manifest, ManifestEntry, save_manifest
from frame.models import Page, ParsedDocument, RetrievalChunk
from frame.qa_index import QAIndex
from frame.sessions import SessionStore
from frame.web import create_app


def _library(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = "a" * 64
    paper = output / "papers" / "paper.md"
    paper.parent.mkdir()
    paper.write_text(
        "# 论文标题\n\n## 研究问题\n\n问题内容（PDF 第 1 页）\n\n"
        "## 代表图片\n\n![方法总览](../images/figure.png)\n"
    )
    image = output / "images" / "figure.png"
    image.parent.mkdir()
    image.write_bytes(b"image")
    with QAIndex(output / ".phelper" / "qa" / "index.sqlite3") as index:
        index.index_document(
            ParsedDocument(pages=[Page(number=1, text="Accuracy was 91%")],
                           parser_version="p"),
            fingerprint, "paper.pdf", ["paper.pdf"], "index-signature",
        )
    save_manifest(output / "manifest.json", Manifest(papers={
        fingerprint: ManifestEntry(
            fingerprint=fingerprint, source_path="paper.pdf", aliases=["paper.pdf"],
            status="success", stage="complete", signature="summary-signature",
            model="m", language="zh", parser_version="p", template_version="1",
            output_path="papers/paper.md", last_used_at="now",
            index_status="success", index_signature="index-signature", indexed_at="now",
        )
    }))
    return fingerprint


def _configured(monkeypatch):
    monkeypatch.setenv("P_HELPER_API_KEY", "secret")
    monkeypatch.setenv("P_HELPER_API_URL", "http://model.test/v1/chat/completions")
    monkeypatch.setenv("P_HELPER_MODEL", "model")


def test_web_bootstrap_library_and_summary(tmp_path, monkeypatch):
    _configured(monkeypatch)
    fingerprint = _library(tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        bootstrap = client.get("/api/bootstrap")
        assert bootstrap.status_code == 200
        assert bootstrap.json()["configured"] == {
            "api_key": True, "api_url": True, "model": True,
        }
        assert "secret" not in bootstrap.text
        papers = client.get("/api/library").json()["papers"]
        assert papers[0]["fingerprint"] == fingerprint
        summary = client.get(f"/api/papers/{fingerprint}/summary")
        assert "PDF 第 1 页" in summary.json()["markdown"]
        assert client.get("/artifacts/images/figure.png").content == b"image"
        page = client.get("/")
        assert "https://unpkg.com" not in page.text
        assert 'src="/lucide.min.js"' in page.text
        assert "https://unpkg.com" not in page.headers["content-security-policy"]
        icons = client.get("/lucide.min.js")
        assert icons.status_code == 200 and len(icons.content) > 300_000
        javascript = client.get("/app.js").text
        assert "function markdownImage" in javascript
        assert "parts[0] !== \"images\"" in javascript


def test_web_mutations_require_page_token_and_artifacts_are_isolated(tmp_path, monkeypatch):
    _configured(monkeypatch)
    _library(tmp_path)
    internal = tmp_path / ".phelper" / "qa" / "sessions.sqlite3"
    with SessionStore(internal):
        pass
    with TestClient(create_app(tmp_path)) as client:
        response = client.post("/api/jobs/path", json={"input": str(tmp_path)})
        assert response.status_code == 403
        assert client.get("/artifacts/.phelper/qa/sessions.sqlite3").status_code == 404
        assert client.get("/artifacts/../manifest.json").status_code == 404


def test_web_path_job_runs_existing_pipeline(tmp_path, monkeypatch):
    _configured(monkeypatch)
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"pdf")
    output = tmp_path / "summaries"
    seen = {}
    monkeypatch.setattr(
        "frame.web.run_batch",
        lambda config, path: seen.update(output=config.output_dir, path=path) or 0,
    )
    with TestClient(create_app(output)) as client:
        token = client.get("/api/bootstrap").json()["csrf_token"]
        response = client.post(
            "/api/jobs/path", json={"input": str(source)},
            headers={"X-P-Helper-Token": token},
        )
        assert response.status_code == 202
        for _ in range(50):
            job = client.get("/api/library").json()["jobs"][0]
            if job["status"] not in {"queued", "running"}:
                break
            time.sleep(0.01)
        assert job["status"] == "success"
        assert seen == {"output": str(output.resolve()), "path": source.resolve()}


def test_web_ask_persists_session_and_reuses_evidence_scope(tmp_path, monkeypatch):
    _configured(monkeypatch)
    fingerprint = _library(tmp_path)
    calls = []

    def answer(config, question, history, index, fingerprints, top_k):
        calls.append((question, list(history), fingerprints, top_k))
        evidence = [RetrievalChunk(
            id=f"{fingerprint}-p1-c1", fingerprint=fingerprint,
            source_path="paper.pdf", page=1, kind="text", text="Accuracy was 91%",
        )]
        return "- 准确率为 91%。（paper.pdf，PDF 第 1 页）", evidence

    monkeypatch.setattr("frame.web.answer_question", answer)
    with TestClient(create_app(tmp_path)) as client:
        token = client.get("/api/bootstrap").json()["csrf_token"]
        headers = {"X-P-Helper-Token": token}
        first = client.post("/api/ask", headers=headers, json={
            "question": "准确率？", "papers": [fingerprint], "top_k": 5,
        })
        assert first.status_code == 200
        session_id = first.json()["session_id"]
        assert first.json()["evidence"][0]["page"] == 1
        second = client.post("/api/ask", headers=headers, json={
            "question": "它可靠吗？", "session_id": session_id,
        })
        assert second.status_code == 200
        assert calls[0][2:] == ([fingerprint], 5)
        assert calls[1][1] == [("准确率？", "- 准确率为 91%。（paper.pdf，PDF 第 1 页）")]
        turns = client.get(f"/api/sessions/{session_id}/turns").json()["turns"]
        assert len(turns) == 2


def test_failed_first_web_ask_does_not_leave_empty_session(tmp_path, monkeypatch):
    _configured(monkeypatch)
    _library(tmp_path)
    monkeypatch.setattr(
        "frame.web.answer_question",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("model failed")),
    )
    with TestClient(create_app(tmp_path)) as client:
        token = client.get("/api/bootstrap").json()["csrf_token"]
        response = client.post(
            "/api/ask", json={"question": "问题"},
            headers={"X-P-Helper-Token": token},
        )
        assert response.status_code == 500
        assert client.get("/api/sessions").json()["sessions"] == []
