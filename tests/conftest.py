import json
from pathlib import Path

import pytest

from frame.config import Config
from frame.models import ParsedDocument, Page
from frame.support import SupportBatch, SupportVerdict


@pytest.fixture
def config(tmp_path):
    return Config(
        api_key="test-key", api_url="http://model.test/v1/chat/completions",
        model="test-model", output_dir=str(tmp_path / "summaries"),
        retry_backoff=0,
    )


class FakeParser:
    version = "fake-1"

    def __init__(self, failures=None):
        self.calls = []
        self.failures = set(failures or [])

    def parse(self, pdf_path: Path, cache_dir: Path):
        self.calls.append(pdf_path.name)
        if pdf_path.name in self.failures:
            raise RuntimeError("fixture parse failed")
        cache_dir.mkdir(parents=True, exist_ok=True)
        document = ParsedDocument(
            pages=[Page(number=1, text=f"Content of {pdf_path.name}. Accuracy was 91%. Limitations include latency.")],
            parser_version=self.version,
        )
        (cache_dir / "parsed.json").write_text(document.model_dump_json(), encoding="utf-8")
        return document


def fact_response(text):
    return json.dumps({"facts": [
        {"id": "ignored", "category": "result", "text": "Accuracy was 91%.",
         "pages": [1], "quote": "Accuracy was 91%.", "figure_ids": []},
        {"id": "ignored", "category": "limitation", "text": "Limitations include latency.",
         "pages": [1], "quote": "Limitations include latency.", "figure_ids": []},
    ]})


def summary_response(fact_prefix="p1-c1"):
    result_id = f"{fact_prefix}-f1"
    limit_id = f"{fact_prefix}-f2"
    missing = "原文未明确说明"
    claim = lambda text, ids: {"text": text, "evidence_ids": ids}
    return json.dumps({
        "title": claim(missing, []), "authors": claim(missing, []),
        "publication_date": claim(missing, []), "venue": claim(missing, []),
        "one_sentence": claim("Accuracy was 91%.", [result_id]),
        "research_problem": claim("Accuracy was 91%.", [result_id]),
        "core_method": claim("Accuracy was 91%.", [result_id]),
        "contributions": [claim("Accuracy was 91%.", [result_id])],
        "experiment_design": {
            "datasets": claim(missing, []), "baselines": claim(missing, []),
            "metrics": claim("Accuracy was 91%.", [result_id]),
            "setup": claim(missing, []),
        },
        "main_results": [claim("Accuracy was 91%.", [result_id])],
        "limitations": [claim("Limitations include latency.", [limit_id])],
        "keywords": [claim("Accuracy", [result_id])],
    })


def support_response(user):
    payload = json.loads(user)
    claims = payload.get("claims", [])
    return SupportBatch(verdicts=[SupportVerdict(
        claim_id=item["claim_id"], supported=True, reason="Fixture-supported claim",
    ) for item in claims]).model_dump_json()


def install_support_model(monkeypatch):
    def structured(config, system, user, schema):
        return schema.model_validate_json(support_response(user))

    monkeypatch.setattr("frame.support.structured_chat", structured)
