"""Atomic durable state for resumable batch processing."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


MANIFEST_VERSION = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ManifestEntry(BaseModel):
    fingerprint: str
    source_path: str
    aliases: list[str] = Field(default_factory=list)
    status: Literal["processing", "success", "failed"]
    stage: str
    signature: str
    model: str
    language: str
    parser_version: str
    template_version: str
    output_path: str | None = None
    processed_at: str | None = None
    last_used_at: str
    error: str | None = None
    index_status: Literal["not_run", "processing", "success", "failed"] = "not_run"
    index_signature: str | None = None
    index_error: str | None = None
    indexed_at: str | None = None


class Manifest(BaseModel):
    version: int = MANIFEST_VERSION
    papers: dict[str, ManifestEntry] = Field(default_factory=dict)


def load_manifest(path: Path) -> Manifest:
    if not path.exists():
        return Manifest()
    manifest = Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    manifest.version = MANIFEST_VERSION
    return manifest


def recover_interrupted(manifest: Manifest):
    for entry in manifest.papers.values():
        if entry.status == "processing":
            entry.status = "failed"
            entry.stage = "interrupted"
            entry.error = "Previous run was interrupted"
        if entry.index_status == "processing":
            entry.index_status = "failed"
            entry.index_error = "Previous indexing run was interrupted"


def atomic_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_manifest(path: Path, manifest: Manifest):
    atomic_text(path, manifest.model_dump_json(indent=2) + "\n")
