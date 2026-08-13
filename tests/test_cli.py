import pytest

from frame.main import build_parser, main, parse_size
from frame.manifest import Manifest, ManifestEntry, load_manifest, save_manifest
from frame.models import Page, ParsedDocument
from frame.qa_index import QAIndex
from frame.sessions import SessionStore


def test_size_parser():
    assert parse_size("5GB") == 5 * 1024 ** 3
    assert parse_size("500 mb") == 500 * 1024 ** 2


def test_cli_requires_all_model_configuration(monkeypatch, tmp_path):
    monkeypatch.delenv("P_HELPER_API_KEY", raising=False)
    monkeypatch.delenv("P_HELPER_API_URL", raising=False)
    monkeypatch.delenv("P_HELPER_MODEL", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda: False)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"pdf")
    with pytest.raises(SystemExit) as exc:
        main([str(pdf)])
    assert exc.value.code == 2


def test_cli_returns_pipeline_exit_code(monkeypatch, tmp_path):
    monkeypatch.setenv("P_HELPER_API_KEY", "key")
    monkeypatch.setattr("frame.main.run_batch", lambda config, path: 1)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"pdf")
    assert main([str(pdf), "--api-url", "http://test", "--model", "m"]) == 1


def test_cli_converts_keyboard_interrupt_to_130(monkeypatch, tmp_path):
    monkeypatch.setenv("P_HELPER_API_KEY", "key")
    def interrupt(config, path):
        raise KeyboardInterrupt
    monkeypatch.setattr("frame.main.run_batch", interrupt)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"pdf")
    assert main([str(pdf), "--api-url", "http://test", "--model", "m"]) == 130


def _qa_index(output):
    path = output / ".phelper" / "qa" / "index.sqlite3"
    with QAIndex(path) as index:
        index.index_document(
            ParsedDocument(pages=[Page(number=1, text="method")], parser_version="p"),
            "a" * 64, "paper.pdf", ["paper.pdf"], "signature",
        )
    save_manifest(output / "manifest.json", Manifest(papers={"a" * 64: ManifestEntry(
        fingerprint="a" * 64, source_path="paper.pdf", aliases=["paper.pdf"],
        status="success", stage="complete", signature="summary", model="m",
        language="zh", parser_version="p", template_version="1", last_used_at="now",
        index_status="success", index_signature="signature", indexed_at="now",
    )}))
    return path


def test_index_command_does_not_require_model_configuration(monkeypatch, tmp_path):
    monkeypatch.delenv("P_HELPER_API_KEY", raising=False)
    monkeypatch.delenv("P_HELPER_API_URL", raising=False)
    monkeypatch.delenv("P_HELPER_MODEL", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda: False)
    seen = {}
    monkeypatch.setattr("frame.main.build_index", lambda source, output, **kwargs:
                        seen.update(source=source, output=output, **kwargs) or 0)
    assert main(["index", str(tmp_path), "-o", str(tmp_path / "out"), "--force"]) == 0
    assert seen["force"] is True


def test_one_shot_ask_creates_persistent_session(tmp_path, monkeypatch):
    output = tmp_path / "summaries"
    _qa_index(output)
    monkeypatch.setenv("P_HELPER_API_KEY", "key")
    monkeypatch.setattr("frame.main.answer_question",
                        lambda *args, **kwargs: ("回答（paper.pdf，PDF 第 1 页）", []))
    assert main(["ask", "问题", "-o", str(output),
                 "--api-url", "http://test", "--model", "m"]) == 0
    with SessionStore(output / ".phelper" / "qa" / "sessions.sqlite3") as sessions:
        saved = sessions.list()
        assert len(saved) == 1
        assert sessions.recent_turns(saved[0].id) == [("问题", "回答（paper.pdf，PDF 第 1 页）")]


def test_saved_session_rejects_conflicting_scope(tmp_path, monkeypatch):
    output = tmp_path / "summaries"
    _qa_index(output)
    path = output / ".phelper" / "qa" / "sessions.sqlite3"
    with SessionStore(path) as sessions:
        session = sessions.create("zh", ["a" * 64], 8)
    monkeypatch.setenv("P_HELPER_API_KEY", "key")
    with pytest.raises(SystemExit) as exc:
        main(["ask", "问题", "-o", str(output), "--session", session.id,
              "--language", "en", "--api-url", "http://test", "--model", "m"])
    assert exc.value.code == 2


def test_saved_session_accepts_same_paper_scope_in_different_order(tmp_path, monkeypatch):
    output = tmp_path / "summaries"
    index_path = _qa_index(output)
    with QAIndex(index_path) as index:
        index.index_document(
            ParsedDocument(pages=[Page(number=1, text="other")], parser_version="p"),
            "b" * 64, "other.pdf", ["other.pdf"], "signature-b",
        )
    manifest = load_manifest(output / "manifest.json")
    manifest.papers["b" * 64] = ManifestEntry(
        fingerprint="b" * 64, source_path="other.pdf", aliases=["other.pdf"],
        status="success", stage="complete", signature="summary-b", model="m",
        language="zh", parser_version="p", template_version="1", last_used_at="now",
        index_status="success", index_signature="signature-b", indexed_at="now",
    )
    save_manifest(output / "manifest.json", manifest)
    sessions_path = output / ".phelper" / "qa" / "sessions.sqlite3"
    with SessionStore(sessions_path) as sessions:
        session = sessions.create("zh", ["a" * 64, "b" * 64], 8)
    monkeypatch.setenv("P_HELPER_API_KEY", "key")
    monkeypatch.setattr("frame.main.answer_question", lambda *args, **kwargs: ("回答", []))

    assert main([
        "ask", "问题", "-o", str(output), "--session", session.id,
        "--paper", "other.pdf", "--paper", "paper.pdf",
        "--api-url", "http://test", "--model", "m",
    ]) == 0


def test_sessions_list_and_delete(tmp_path, capsys):
    output = tmp_path / "summaries"
    path = output / ".phelper" / "qa" / "sessions.sqlite3"
    with SessionStore(path) as sessions:
        session = sessions.create("zh", [], 8)
    assert main(["sessions", "list", "-o", str(output)]) == 0
    assert session.id in capsys.readouterr().out
    assert main(["sessions", "delete", session.id, "-o", str(output)]) == 0
    with SessionStore(path) as sessions:
        assert sessions.get(session.id) is None


def test_interactive_ask_reuses_one_session_and_history(tmp_path, monkeypatch):
    output = tmp_path / "summaries"
    _qa_index(output)
    monkeypatch.setenv("P_HELPER_API_KEY", "key")
    questions = iter(["第一问", "追问", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(questions))
    histories = []

    def answer(config, question, history, index, fingerprints, top_k):
        histories.append(list(history))
        return f"回答:{question}", []

    monkeypatch.setattr("frame.main.answer_question", answer)
    assert main(["ask", "-o", str(output),
                 "--api-url", "http://test", "--model", "m"]) == 0
    assert histories == [[], [("第一问", "回答:第一问")]]
    with SessionStore(output / ".phelper" / "qa" / "sessions.sqlite3") as sessions:
        saved = sessions.list()
        assert len(saved) == 1
        assert len(sessions.recent_turns(saved[0].id)) == 2


def test_saved_session_rejects_papers_missing_from_current_index(tmp_path, monkeypatch):
    output = tmp_path / "summaries"
    index_path = _qa_index(output)
    sessions_path = output / ".phelper" / "qa" / "sessions.sqlite3"
    with SessionStore(sessions_path) as sessions:
        session = sessions.create("zh", ["a" * 64], 8)
    with QAIndex(index_path) as index:
        index.remove_paper("a" * 64)
        index.index_document(
            ParsedDocument(pages=[Page(number=1, text="other")], parser_version="p"),
            "b" * 64, "other.pdf", ["other.pdf"], "signature-b",
        )
    monkeypatch.setenv("P_HELPER_API_KEY", "key")
    with pytest.raises(SystemExit) as exc:
        main(["ask", "问题", "-o", str(output), "--session", session.id,
              "--api-url", "http://test", "--model", "m"])
    assert exc.value.code == 2


def test_ask_excludes_stale_sqlite_paper_marked_failed_in_manifest(tmp_path, monkeypatch):
    output = tmp_path / "summaries"
    _qa_index(output)
    manifest = load_manifest(output / "manifest.json")
    entry = manifest.papers["a" * 64]
    entry.index_status = "failed"
    entry.index_signature = None
    save_manifest(output / "manifest.json", manifest)
    monkeypatch.setenv("P_HELPER_API_KEY", "key")
    with pytest.raises(SystemExit) as exc:
        main(["ask", "问题", "-o", str(output),
              "--api-url", "http://test", "--model", "m"])
    assert exc.value.code == 2


def test_failed_first_ask_does_not_leave_empty_session(tmp_path, monkeypatch):
    output = tmp_path / "summaries"
    _qa_index(output)
    monkeypatch.setenv("P_HELPER_API_KEY", "key")
    monkeypatch.setattr("frame.main.answer_question",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("model failed")))
    assert main(["ask", "问题", "-o", str(output),
                 "--api-url", "http://test", "--model", "m"]) == 1
    with SessionStore(output / ".phelper" / "qa" / "sessions.sqlite3") as sessions:
        assert sessions.list() == []


def test_failed_resumed_ask_preserves_existing_session(tmp_path, monkeypatch):
    output = tmp_path / "summaries"
    _qa_index(output)
    path = output / ".phelper" / "qa" / "sessions.sqlite3"
    with SessionStore(path) as sessions:
        session = sessions.create("zh", ["a" * 64], 8)
        sessions.add_turn(session.id, "旧问题", "旧回答")
    monkeypatch.setenv("P_HELPER_API_KEY", "key")
    monkeypatch.setattr("frame.main.answer_question",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("model failed")))
    assert main(["ask", "追问", "-o", str(output), "--session", session.id,
                 "--api-url", "http://test", "--model", "m"]) == 1
    with SessionStore(path) as sessions:
        assert sessions.recent_turns(session.id) == [("旧问题", "旧回答")]


def test_empty_interactive_session_is_removed_on_exit(tmp_path, monkeypatch):
    output = tmp_path / "summaries"
    _qa_index(output)
    monkeypatch.setenv("P_HELPER_API_KEY", "key")
    monkeypatch.setattr("builtins.input", lambda _: "exit")
    assert main(["ask", "-o", str(output),
                 "--api-url", "http://test", "--model", "m"]) == 0
    with SessionStore(output / ".phelper" / "qa" / "sessions.sqlite3") as sessions:
        assert sessions.list() == []


def test_later_interactive_failure_keeps_completed_turn(tmp_path, monkeypatch):
    output = tmp_path / "summaries"
    _qa_index(output)
    monkeypatch.setenv("P_HELPER_API_KEY", "key")
    questions = iter(["第一问", "第二问"])
    monkeypatch.setattr("builtins.input", lambda _: next(questions))
    calls = 0

    def answer(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second failed")
        return "第一答", []

    monkeypatch.setattr("frame.main.answer_question", answer)
    assert main(["ask", "-o", str(output),
                 "--api-url", "http://test", "--model", "m"]) == 1
    with SessionStore(output / ".phelper" / "qa" / "sessions.sqlite3") as sessions:
        saved = sessions.list()
        assert len(saved) == 1
        assert sessions.recent_turns(saved[0].id) == [("第一问", "第一答")]


def test_interrupted_first_ask_removes_empty_session(tmp_path, monkeypatch):
    output = tmp_path / "summaries"
    _qa_index(output)
    monkeypatch.setenv("P_HELPER_API_KEY", "key")
    monkeypatch.setattr("frame.main.answer_question",
                        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt))
    assert main(["ask", "问题", "-o", str(output),
                 "--api-url", "http://test", "--model", "m"]) == 130
    with SessionStore(output / ".phelper" / "qa" / "sessions.sqlite3") as sessions:
        assert sessions.list() == []


def test_ask_error_log_redacts_api_key(tmp_path, monkeypatch, caplog):
    output = tmp_path / "summaries"
    _qa_index(output)
    key = "ask-super-secret"
    monkeypatch.setenv("P_HELPER_API_KEY", key)
    monkeypatch.setattr("frame.main.answer_question",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(key)))
    assert main(["ask", "问题", "-o", str(output),
                 "--api-url", "http://test", "--model", "m"]) == 1
    assert key not in caplog.text
    assert "[REDACTED]" in caplog.text
