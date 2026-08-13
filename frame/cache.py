"""Regenerable cache loading and LRU enforcement."""

import os
import shutil
from pathlib import Path

from .models import ParsedDocument


def load_parsed(cache_dir: Path) -> ParsedDocument | None:
    path = cache_dir / "parsed.json"
    if not path.exists():
        return None
    try:
        document = ParsedDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    _touch(cache_dir)
    return document


def enforce_cache_limits(cache_root: Path, max_papers: int, max_size: int):
    if not cache_root.exists():
        return
    entries = []
    for directory in cache_root.iterdir():
        if not directory.is_dir():
            continue
        size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
        entries.append([directory.stat().st_mtime, size, directory])
    entries.sort(key=lambda item: item[0])
    total = sum(item[1] for item in entries)
    while entries and (len(entries) > max_papers or total > max_size):
        _, size, directory = entries.pop(0)
        shutil.rmtree(directory)
        total -= size


def _touch(directory: Path):
    os.utime(directory, None)
