import json
import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from frame.pipeline import run_batch
from frame.qa import answer_question
from frame.qa_index import QAIndex
from frame.indexing import qa_root
from frame.models import QAAnswer, QAClaim, QueryExpansion
from frame.support import QueryReview, SupportBatch, SupportVerdict

from conftest import FakeParser, fact_response, summary_response


def _support_response(*claim_ids):
    return SupportBatch(verdicts=[SupportVerdict(
        claim_id=claim_id, supported=True, reason="Supported by fixture evidence",
    ) for claim_id in claim_ids]).model_dump_json()


SUMMARY_CLAIM_IDS = (
    "one_sentence", "research_problem", "core_method", "contribution_1",
    "experiment_metrics", "main_result_1", "limitation_1", "keyword_1",
)


class ModelHandler(BaseHTTPRequestHandler):
    responses = []
    requests = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        self.__class__.requests.append(json.loads(self.rfile.read(length)))
        content = self.__class__.responses.pop(0)
        payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


def test_pipeline_uses_openai_compatible_http_contract(tmp_path, config):
    ModelHandler.responses = [
        fact_response(""), summary_response(), _support_response(*SUMMARY_CLAIM_IDS),
    ]
    ModelHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), ModelHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"paper")
        live = type(config)(**{
            **config.__dict__,
            "api_url": f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
        })
        assert run_batch(live, pdf, FakeParser()) == 0
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert len(ModelHandler.requests) == 3
    assert all(request["model"] == "test-model" for request in ModelHandler.requests)
    assert all(request["response_format"] == {"type": "json_object"}
               for request in ModelHandler.requests)


def test_summary_index_and_cross_paper_qa_http_flow(tmp_path, config):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"paper")
    fingerprint = hashlib.sha256(b"paper").hexdigest()
    evidence_id = f"{fingerprint}-p1-c1"
    expansion = QueryExpansion(
        standalone_question="What accuracy was reported?",
        keywords=["accuracy", "精度", "reported accuracy"],
    ).model_dump_json()
    answer = QAAnswer(sufficient=True, claims=[
        QAClaim(text="论文报告的准确率为 91%。", evidence_ids=[evidence_id]),
    ]).model_dump_json()
    ModelHandler.responses = [
        fact_response(""), summary_response(), _support_response(*SUMMARY_CLAIM_IDS), expansion,
        QueryReview(faithful=True, reason="Faithful rewrite").model_dump_json(), answer,
        _support_response("claim_1"),
    ]
    ModelHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), ModelHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        live = type(config)(**{
            **config.__dict__,
            "api_url": f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
        })
        assert run_batch(live, pdf, FakeParser()) == 0
        with QAIndex(qa_root(Path(live.output_dir)) / "index.sqlite3") as index:
            rendered, evidence = answer_question(
                live, "准确率是多少？", [], index, [fingerprint], 8,
            )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert "paper.pdf，PDF 第 1 页" in rendered
    assert len(evidence) == 1
    assert len(ModelHandler.requests) == 7
    assert "What accuracy was reported?" in ModelHandler.requests[-2]["messages"][1]["content"]
    assert ModelHandler.requests[-1]["messages"][0]["content"].startswith(
        "You verify whether evidence directly supports"
    )
