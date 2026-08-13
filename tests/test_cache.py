import os
from pathlib import Path

from frame.cache import enforce_cache_limits


def _cache(root, name, size, mtime):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "payload").write_bytes(b"x" * size)
    os.utime(directory, (mtime, mtime))
    return directory


def test_cache_removes_oldest_paper_over_count_limit(tmp_path):
    cache = tmp_path / ".phelper" / "cache"
    old = _cache(cache, "old", 10, 1)
    new = _cache(cache, "new", 10, 2)
    enforce_cache_limits(cache, max_papers=1, max_size=100)
    assert not old.exists()
    assert new.exists()


def test_cache_removes_oldest_paper_over_size_limit(tmp_path):
    cache = tmp_path / ".phelper" / "cache"
    old = _cache(cache, "old", 60, 1)
    new = _cache(cache, "new", 60, 2)
    final = tmp_path / "papers" / "paper.md"
    final.parent.mkdir()
    final.write_text("keep")
    enforce_cache_limits(cache, max_papers=50, max_size=100)
    assert not old.exists()
    assert new.exists()
    assert final.read_text() == "keep"
