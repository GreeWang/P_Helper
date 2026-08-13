import os

from frame.sessions import SessionStore


def test_sessions_persist_turns_list_and_delete(tmp_path):
    path = tmp_path / "sessions.sqlite3"
    with SessionStore(path) as store:
        session = store.create("zh", ["a" * 64], 8)
        store.add_turn(session.id, "问题", "回答")
    with SessionStore(path) as store:
        restored = store.get(session.id)
        assert restored.language == "zh" and restored.fingerprints == ["a" * 64]
        assert store.recent_turns(session.id) == [("问题", "回答")]
        assert store.list()[0].id == session.id
        assert store.delete(session.id)
        assert store.get(session.id) is None


def test_session_count_lru_preserves_current_session(tmp_path):
    with SessionStore(tmp_path / "sessions.sqlite3", max_sessions=2) as store:
        first = store.create("zh", [], 8)
        second = store.create("zh", [], 8)
        third = store.create("zh", [], 8)
        assert store.get(first.id) is None
        assert {item.id for item in store.list()} == {second.id, third.id}


def test_session_size_limit_removes_old_sessions(tmp_path):
    path = tmp_path / "sessions.sqlite3"
    with SessionStore(path, max_sessions=100, max_size=1024 * 1024) as store:
        first = store.create("zh", [], 8)
        store.add_turn(first.id, "q", "a" * (2 * 1024 * 1024))
        assert store.get(first.id) is None
        second = store.create("zh", [], 8)
        assert store.get(second.id) is not None
