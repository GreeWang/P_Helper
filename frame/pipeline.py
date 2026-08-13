"""Resumable end-to-end batch orchestration."""

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .cache import enforce_cache_limits, load_parsed
from .config import Config
from .discovery import discover_pdfs, note_filename, sha256_file
from .figures import select_figures
from .errors import safe_error
from .index import render_index
from .indexing import _index_failure, index_document, qa_root
from .manifest import (ManifestEntry, atomic_text, load_manifest, recover_interrupted,
                       save_manifest, utc_now)
from .locking import output_lock
from .models import Fact, PaperSummary, ParsedDocument
from .parser import MarkerPdfParser, PdfParser
from .qa_index import QAIndex
from .qa_index import index_signature
from .render import render_paper
from .summarize import summarize_document, summary_template_version


logger = logging.getLogger(__name__)
class _UnavailableQAIndex:
    def __init__(self, error: Exception):
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def has_signature(self, *_):
        return False

    def index_document(self, *_args, **_kwargs):
        raise self.error

    def remove_paper(self, *_):
        return None

    def update_metadata(self, *_):
        return None


@dataclass
class WorkItem:
    fingerprint: str
    pdf_path: Path
    source_path: str
    aliases: list[str]
    cache_dir: Path
    document: ParsedDocument
    signature: str


def run_batch(config: Config, input_path: Path, parser: PdfParser | None = None) -> int:
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_lock(output_dir):
        return _run_batch_locked(config, input_path, parser, output_dir)


def _run_batch_locked(config: Config, input_path: Path, parser: PdfParser | None,
                      output_dir: Path) -> int:
    parser = parser or MarkerPdfParser()
    root, pdfs = discover_pdfs(input_path)
    if not pdfs:
        raise ValueError(f"No PDF files found in {input_path}")
    manifest_path = output_dir / "manifest.json"
    cache_root = output_dir / ".phelper" / "cache"
    manifest = load_manifest(manifest_path)
    recover_interrupted(manifest)
    save_manifest(manifest_path, manifest)

    grouped: dict[str, list[Path]] = {}
    for pdf in pdfs:
        grouped.setdefault(sha256_file(pdf), []).append(pdf)

    ready: list[WorkItem] = []
    failures = 0
    try:
        qa_index_context = QAIndex(qa_root(output_dir) / "index.sqlite3")
    except Exception as exc:
        qa_index_context = _UnavailableQAIndex(exc)
    with qa_index_context as qa_index:
        for fingerprint, paths in grouped.items():
            aliases = [path.relative_to(root).as_posix() for path in paths]
            source = aliases[0]
            signature = _signature(fingerprint, parser.version, config)
            existing = manifest.papers.get(fingerprint)
            skip_summary = (_can_skip(existing, signature, output_dir) and not config.force)
            if skip_summary:
                entry = existing
                entry.aliases = aliases
                entry.source_path = source
                entry.last_used_at = utc_now()
            else:
                entry = ManifestEntry(
                    fingerprint=fingerprint, source_path=source, aliases=aliases,
                    status="processing", stage="parsing", signature=signature,
                    model=config.model, language=config.language,
                    parser_version=parser.version, template_version=summary_template_version(),
                    last_used_at=utc_now(),
                    index_status=existing.index_status if existing else "not_run",
                    index_signature=existing.index_signature if existing else None,
                    index_error=existing.index_error if existing else None,
                    indexed_at=existing.indexed_at if existing else None,
                )
                manifest.papers[fingerprint] = entry
            _checkpoint(output_dir, manifest_path, manifest, config.language)
            cache_dir = cache_root / fingerprint
            current_index_signature = index_signature(fingerprint, parser.version)
            if (skip_summary and entry.index_status == "success" and
                    entry.index_signature == current_index_signature and
                    qa_index.has_signature(fingerprint, current_index_signature)):
                qa_index.update_metadata(fingerprint, source, aliases)
                logger.info("Skipping unchanged paper: %s", source)
                _checkpoint(output_dir, manifest_path, manifest, config.language)
                continue
            try:
                document = None if config.force else load_parsed(cache_dir)
                if document is not None and document.parser_version != parser.version:
                    document = None
                if document is None:
                    document = parser.parse(paths[0], cache_dir)
                entry.parser_version = document.parser_version
            except Exception as exc:
                failures += 1
                if skip_summary:
                    _index_failure(entry, exc, config.api_key)
                else:
                    _fail(entry, "parsing", exc, config.api_key)
                _checkpoint(output_dir, manifest_path, manifest, config.language)
                continue
            try:
                index_document(qa_index, document, entry, fingerprint, source, aliases,
                               force=config.force,
                               checkpoint=lambda: save_manifest(manifest_path, manifest))
            except Exception as exc:
                failures += 1
                _index_failure(entry, exc, config.api_key)
                qa_index.remove_paper(fingerprint)
            if skip_summary:
                logger.info("Skipping unchanged paper: %s", source)
                _checkpoint(output_dir, manifest_path, manifest, config.language)
                continue
            entry.stage = "summarizing"
            save_manifest(manifest_path, manifest)
            ready.append(WorkItem(
                fingerprint, paths[0], source, aliases, cache_dir, document, signature
            ))

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = {executor.submit(summarize_document, config, item.document, item.cache_dir): item
                   for item in ready}
        for future in as_completed(futures):
            item = futures[future]
            entry = manifest.papers[item.fingerprint]
            try:
                summary, facts = future.result()
                _finish_paper(config, output_dir, item, entry, summary, facts)
            except Exception as exc:
                failures += 1
                _fail(entry, "summarizing", exc, config.api_key)
            _checkpoint(output_dir, manifest_path, manifest, config.language)

    enforce_cache_limits(cache_root, config.cache_max_papers, config.cache_max_size)
    _checkpoint(output_dir, manifest_path, manifest, config.language)
    return 1 if failures else 0


def _finish_paper(config: Config, output_dir: Path, item: WorkItem,
                  entry: ManifestEntry, summary: PaperSummary, facts: list[Fact]):
    entry.stage = "rendering"
    figures = select_figures(item.document.figures, output_dir, item.fingerprint)
    relative = Path("papers") / note_filename(item.pdf_path, item.fingerprint)
    markdown = render_paper(
        summary, facts, figures, item.source_path, config.model, config.language
    )
    atomic_text(output_dir / relative, markdown)
    entry.status = "success"
    entry.stage = "complete"
    entry.output_path = relative.as_posix()
    entry.processed_at = utc_now()
    entry.last_used_at = entry.processed_at
    entry.error = None


def _signature(fingerprint: str, parser_version: str, config: Config) -> str:
    payload = {
        "fingerprint": fingerprint,
        "parser": parser_version,
        "template": summary_template_version(),
        "model": config.model,
        "api_url": config.api_url,
        "language": config.language,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _can_skip(entry, signature: str, output_dir: Path) -> bool:
    return bool(entry and entry.status == "success" and entry.signature == signature
                and entry.output_path and (output_dir / entry.output_path).is_file())


def _fail(entry: ManifestEntry, stage: str, exc: Exception, secret: str = ""):
    entry.status = "failed"
    entry.stage = stage
    entry.error = safe_error(exc, secret)
    entry.processed_at = utc_now()
    entry.last_used_at = entry.processed_at


def _checkpoint(output_dir: Path, manifest_path: Path, manifest, language: str):
    save_manifest(manifest_path, manifest)
    atomic_text(output_dir / "README.md", render_index(manifest, language))
