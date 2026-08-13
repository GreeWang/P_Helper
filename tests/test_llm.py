import json

import pytest

from frame.config import Config
from frame.errors import LLMError, safe_error
from frame.llm import chat, structured_chat
from frame.models import FactBatch


class Response:
    def __init__(self, status=200, content="{}", headers=None):
        self.status_code = status
        self.headers = headers or {}
        self.text = content
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _config(**changes):
    values = dict(api_key="secret", api_url="http://test", model="m",
                  retry_backoff=0, max_retries=3, json_repair_attempts=2)
    values.update(changes)
    return Config(**values)


def test_chat_sends_model_key_timeout_and_json_format(monkeypatch):
    seen = {}
    def post(url, **kwargs):
        seen.update(url=url, **kwargs)
        return Response(content='{"facts": []}')
    monkeypatch.setattr("frame.llm.requests.post", post)
    assert chat(_config(), "system", "user") == '{"facts": []}'
    assert seen["json"]["model"] == "m"
    assert seen["json"]["response_format"] == {"type": "json_object"}
    assert seen["headers"]["Authorization"] == "Bearer secret"


def test_retryable_status_is_retried(monkeypatch):
    responses = iter([Response(429), Response(content="ok")])
    monkeypatch.setattr("frame.llm.requests.post", lambda *a, **k: next(responses))
    monkeypatch.setattr("frame.llm.time.sleep", lambda _: None)
    assert chat(_config(), "s", "u") == "ok"


def test_non_retryable_status_fails(monkeypatch):
    monkeypatch.setattr("frame.llm.requests.post",
                        lambda *a, **k: Response(401, "denied secret"))
    with pytest.raises(LLMError, match="401"):
        chat(_config(), "s", "u")


def test_http_error_never_includes_response_body_or_api_key(monkeypatch):
    key = "super-secret-api-key"
    monkeypatch.setattr("frame.llm.requests.post",
                        lambda *a, **k: Response(400, f"Authorization: Bearer {key}"))
    with pytest.raises(LLMError) as exc:
        chat(_config(api_key=key), "s", "u")
    assert key not in str(exc.value)
    assert "Authorization" not in str(exc.value)


def test_request_exception_only_reports_exception_type(monkeypatch):
    key = "super-secret-api-key"
    def fail(*args, **kwargs):
        raise __import__("requests").RequestException(f"failed with {key}")
    monkeypatch.setattr("frame.llm.requests.post", fail)
    with pytest.raises(LLMError) as exc:
        chat(_config(api_key=key, max_retries=1), "s", "u")
    assert key not in str(exc.value)
    assert "RequestException" in str(exc.value)


def test_structured_chat_repairs_invalid_json(monkeypatch):
    prompts = []
    values = iter(["not-json", json.dumps({"facts": []})])
    def fake_chat(config, system, user):
        prompts.append(user)
        return next(values)
    monkeypatch.setattr("frame.llm.chat", fake_chat)
    assert structured_chat(_config(), "s", "u", FactBatch).facts == []
    repair = json.loads(prompts[1])
    assert repair["original_request"] == "u"
    assert repair["previous_invalid_response_untrusted"] == "not-json"


def test_structured_models_reject_unknown_fields():
    with pytest.raises(Exception, match="extra"):
        FactBatch.model_validate({"facts": [], "ignore_previous": True})


def test_structured_chat_stops_after_repair_limit(monkeypatch):
    monkeypatch.setattr("frame.llm.chat", lambda *a, **k: "not-json")
    with pytest.raises(LLMError, match="valid FactBatch"):
        structured_chat(_config(json_repair_attempts=1), "s", "u", FactBatch)


def test_safe_error_redacts_secrets_and_control_characters():
    result = safe_error(RuntimeError("before\nsecret\x1b[31mafter"), "secret")
    assert result == "RuntimeError: before [REDACTED] [31mafter"
