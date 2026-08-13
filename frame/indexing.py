"""Automatic and explicit PDF keyword indexing."""

import logging
import os
from pathlib import Path

from .cache import load_parsed
from .discovery import discover_pdfs, sha256_file
from .errors import safe_error
from .index import render_index
from .manifest import atomic_text, load_manifest, save_manifest, utc_now
from .locking import output_lock
from .parser import MarkerPdfParser, PdfParser
from .qa_index import QAIndex, index_signature


logger = logging.getLogger(__name__)


def qa_root(output_dir: Path) -> Path:
    return output_dir / ".phelper" / "qa"


def build_index(input_path: Path, output_dir: Path, language: str | None = None,
                force: bool = False, parser: PdfParser | None = None) -> int:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_lock(output_dir):
        return _build_index_locked(input_path, output_dir, language, force, parser)


def _build_index_locked(input_path: Path, output_dir: Path, language: str | None,
                        force: bool, parser: PdfParser | None) -> int:
    parser = parser or MarkerPdfParser()
    root, pdfs = discover_pdfs(input_path)
    if not pdfs:
        raise ValueError(f"No PDF files found in {input_path}")
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Manifest does not exist: {manifest_path}")
    manifest = load_manifest(manifest_path)
    if language is None:
        language = next((entry.language for entry in manifest.papers.values()), "zh")
    grouped = {}
    for pdf in pdfs:
        grouped.setdefault(sha256_file(pdf), []).append(pdf)
    failures = 0
    try:
        index_context = QAIndex(qa_root(output_dir) / "index.sqlite3")
    except Exception as exc:
        for fingerprint in grouped:
            entry = manifest.papers.get(fingerprint)
            if entry is not None:
                _index_failure(entry, exc)
            failures += 1
        save_manifest(manifest_path, manifest)
        atomic_text(output_dir / "README.md", render_index(manifest, language))
        return 1 if failures else 0
    with index_context as index:
        for fingerprint, paths in grouped.items():
            aliases = [path.relative_to(root).as_posix() for path in paths]
            entry = manifest.papers.get(fingerprint)
            if entry is None:
                logger.error("Paper has no manifest entry: %s", aliases[0])
                failures += 1
                continue
            entry.source_path = aliases[0]
            entry.aliases = aliases
            signature = index_signature(fingerprint, parser.version)
            if (not force and entry.index_status == "success" and
                    entry.index_signature == signature and
                    index.has_signature(fingerprint, signature)):
                index.update_metadata(fingerprint, entry.source_path, entry.aliases)
                save_manifest(manifest_path, manifest)
                logger.info("Skipping unchanged index: %s", entry.source_path)
                continue
            entry.index_status = "processing"
            entry.index_error = None
            save_manifest(manifest_path, manifest)
            try:
                cache_dir = output_dir / ".phelper" / "cache" / fingerprint
                document = load_parsed(cache_dir)
                if document is None or document.parser_version != parser.version:
                    document = parser.parse(paths[0], cache_dir)
                index.index_document(document, fingerprint, entry.source_path,
                                     entry.aliases or aliases, signature)
                _index_success(entry, signature)
            except Exception as exc:
                failures += 1
                _index_failure(entry, exc)
                index.remove_paper(fingerprint)
            save_manifest(manifest_path, manifest)
    atomic_text(output_dir / "README.md", render_index(manifest, language))
    return 1 if failures else 0


def index_document(index: QAIndex, document, entry, fingerprint: str,
                   source_path: str, aliases: list[str], force: bool = False,
                   checkpoint=None) -> bool:
    signature = index_signature(fingerprint, document.parser_version)
    if (not force and entry.index_status == "success" and
            entry.index_signature == signature and
            index.has_signature(fingerprint, signature)):
        index.update_metadata(fingerprint, source_path, aliases)
        return False
    entry.index_status = "processing"
    entry.index_error = None
    if checkpoint is not None:
        checkpoint()
    index.index_document(document, fingerprint, source_path, aliases, signature)
    _index_success(entry, signature)
    return True


def _index_success(entry, signature: str):
    entry.index_status = "success"
    entry.index_signature = signature
    entry.index_error = None
    entry.indexed_at = utc_now()


def _index_failure(entry, exc: Exception, secret: str = ""):
    entry.index_status = "failed"
    entry.index_signature = None
    entry.index_error = safe_error(exc, secret or os.environ.get("P_HELPER_API_KEY", ""))
    entry.indexed_at = None
