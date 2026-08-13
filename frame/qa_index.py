"""SQLite FTS5 storage and deterministic hybrid keyword retrieval."""

import json
import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .manifest import utc_now
from .models import ParsedDocument, RetrievalChunk
from .retrieval import retrieval_chunks
from .retrieval import INDEX_TEMPLATE_VERSION


RRF_K = 60
PER_INDEX_LIMIT = 30
PER_PAPER_LIMIT = 3


def index_signature(fingerprint: str, parser_version: str) -> str:
    value = f"{fingerprint}\0{parser_version}\0{INDEX_TEMPLATE_VERSION}"
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class IndexedPaper:
    fingerprint: str
    source_path: str
    aliases: list[str]
    parser_version: str
    signature: str


class QAIndex:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _create_schema(self):
        self.connection.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS papers (
                fingerprint TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                aliases TEXT NOT NULL,
                parser_version TEXT NOT NULL,
                signature TEXT NOT NULL,
                indexed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL REFERENCES papers(fingerprint) ON DELETE CASCADE,
                source_path TEXT NOT NULL,
                page INTEGER NOT NULL,
                headings TEXT NOT NULL,
                kind TEXT NOT NULL,
                text TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_unicode USING fts5(
                chunk_id UNINDEXED, text, headings, source_path,
                tokenize='unicode61'
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_trigram USING fts5(
                chunk_id UNINDEXED, text, headings, source_path,
                tokenize='trigram'
            );
        """)

    def has_signature(self, fingerprint: str, signature: str) -> bool:
        row = self.connection.execute(
            "SELECT signature FROM papers WHERE fingerprint = ?", (fingerprint,)
        ).fetchone()
        return bool(row and row["signature"] == signature)

    def update_metadata(self, fingerprint: str, source_path: str, aliases: list[str]):
        with self.connection:
            self.connection.execute(
                "UPDATE papers SET source_path = ?, aliases = ? WHERE fingerprint = ?",
                (source_path, json.dumps(aliases, ensure_ascii=False), fingerprint),
            )
            self.connection.execute(
                "UPDATE chunks SET source_path = ? WHERE fingerprint = ?",
                (source_path, fingerprint),
            )
            chunk_ids = [row[0] for row in self.connection.execute(
                "SELECT id FROM chunks WHERE fingerprint = ?", (fingerprint,)
            )]
            for table in ("chunks_unicode", "chunks_trigram"):
                for chunk_id in chunk_ids:
                    row = self.connection.execute(
                        f"SELECT text, headings FROM {table} WHERE chunk_id = ?", (chunk_id,)
                    ).fetchone()
                    self.connection.execute(f"DELETE FROM {table} WHERE chunk_id = ?", (chunk_id,))
                    self.connection.execute(
                        f"INSERT INTO {table} VALUES (?, ?, ?, ?)",
                        (chunk_id, row["text"], row["headings"], source_path),
                    )

    def remove_paper(self, fingerprint: str):
        with self.connection:
            chunk_ids = [row[0] for row in self.connection.execute(
                "SELECT id FROM chunks WHERE fingerprint = ?", (fingerprint,)
            )]
            self._delete_fts(chunk_ids)
            self.connection.execute("DELETE FROM chunks WHERE fingerprint = ?", (fingerprint,))
            self.connection.execute("DELETE FROM papers WHERE fingerprint = ?", (fingerprint,))

    def index_document(self, document: ParsedDocument, fingerprint: str,
                       source_path: str, aliases: list[str], signature: str):
        chunks = retrieval_chunks(document, fingerprint, source_path)
        with self.connection:
            old_ids = [row[0] for row in self.connection.execute(
                "SELECT id FROM chunks WHERE fingerprint = ?", (fingerprint,)
            )]
            self._delete_fts(old_ids)
            self.connection.execute("DELETE FROM chunks WHERE fingerprint = ?", (fingerprint,))
            self.connection.execute("""
                INSERT INTO papers(fingerprint, source_path, aliases, parser_version,
                                   signature, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(fingerprint) DO UPDATE SET
                    source_path=excluded.source_path, aliases=excluded.aliases,
                    parser_version=excluded.parser_version, signature=excluded.signature,
                    indexed_at=excluded.indexed_at
            """, (fingerprint, source_path, json.dumps(aliases, ensure_ascii=False),
                  document.parser_version, signature, utc_now()))
            for chunk in chunks:
                headings = "\n".join(chunk.headings)
                self.connection.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (chunk.id, chunk.fingerprint, chunk.source_path, chunk.page,
                     headings, chunk.kind, chunk.text),
                )
                values = (chunk.id, chunk.text, headings, chunk.source_path)
                self.connection.execute("INSERT INTO chunks_unicode VALUES (?, ?, ?, ?)", values)
                self.connection.execute("INSERT INTO chunks_trigram VALUES (?, ?, ?, ?)", values)

    def _delete_fts(self, chunk_ids: list[str]):
        for chunk_id in chunk_ids:
            self.connection.execute("DELETE FROM chunks_unicode WHERE chunk_id = ?", (chunk_id,))
            self.connection.execute("DELETE FROM chunks_trigram WHERE chunk_id = ?", (chunk_id,))

    def list_papers(self, allowed: set[str] | None = None) -> list[IndexedPaper]:
        return [IndexedPaper(
            fingerprint=row["fingerprint"], source_path=row["source_path"],
            aliases=json.loads(row["aliases"]), parser_version=row["parser_version"],
            signature=row["signature"],
        ) for row in self.connection.execute("SELECT * FROM papers ORDER BY source_path")
            if allowed is None or row["fingerprint"] in allowed]

    def resolve_papers(self, selectors: list[str], allowed: set[str] | None = None) -> list[str]:
        if not selectors:
            return [paper.fingerprint for paper in self.list_papers(allowed)]
        resolved = []
        papers = self.list_papers(allowed)
        for selector in selectors:
            matches = [paper for paper in papers if
                       paper.source_path == selector or selector in paper.aliases or
                       paper.fingerprint.startswith(selector)]
            if not matches:
                raise ValueError(f"No indexed paper matches: {selector}")
            if len(matches) > 1:
                raise ValueError(f"Paper selector is ambiguous: {selector}")
            if matches[0].fingerprint not in resolved:
                resolved.append(matches[0].fingerprint)
        return resolved

    def search(self, keywords: list[str], fingerprints: list[str],
               top_k: int = 8) -> list[RetrievalChunk]:
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        terms = _clean_terms(keywords)
        if not terms or not fingerprints:
            return []
        scores: dict[str, float] = {}
        for table in ("chunks_unicode", "chunks_trigram"):
            rows = self._fts_search(table, terms, fingerprints)
            for rank, row in enumerate(rows, 1):
                scores[row["chunk_id"]] = scores.get(row["chunk_id"], 0) + 1 / (RRF_K + rank)
        if not scores:
            return []
        placeholders = ",".join("?" for _ in scores)
        rows = self.connection.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})", tuple(scores)
        ).fetchall()
        by_id = {row["id"]: row for row in rows}
        counts: dict[str, int] = {}
        selected = []
        for chunk_id in sorted(scores, key=lambda item: (-scores[item], item)):
            row = by_id[chunk_id]
            if counts.get(row["fingerprint"], 0) >= PER_PAPER_LIMIT:
                continue
            counts[row["fingerprint"]] = counts.get(row["fingerprint"], 0) + 1
            selected.append(_row_chunk(row))
            if len(selected) == top_k:
                break
        return selected

    def _fts_search(self, table: str, terms: list[str], fingerprints: list[str]):
        query = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms
                            if table != "chunks_trigram" or len(term) >= 3)
        if not query:
            return []
        placeholders = ",".join("?" for _ in fingerprints)
        return self.connection.execute(f"""
            SELECT f.chunk_id, bm25({table}) AS score
            FROM {table} AS f JOIN chunks AS c ON c.id = f.chunk_id
            WHERE {table} MATCH ? AND c.fingerprint IN ({placeholders})
            ORDER BY score, f.chunk_id LIMIT ?
        """, (query, *fingerprints, PER_INDEX_LIMIT)).fetchall()


def _clean_terms(keywords: list[str]) -> list[str]:
    seen = set()
    terms = []
    for value in keywords:
        term = " ".join(value.strip().split())
        if term and term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


def _row_chunk(row) -> RetrievalChunk:
    return RetrievalChunk(
        id=row["id"], fingerprint=row["fingerprint"], source_path=row["source_path"],
        page=row["page"], headings=row["headings"].splitlines(),
        kind=row["kind"], text=row["text"],
    )
