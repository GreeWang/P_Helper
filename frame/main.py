"""Command-line entry point for PDF batch summarisation."""

import argparse
import logging
import os
import re
import sys
from pathlib import Path

from .config import (Config, DEFAULT_CACHE_MAX_PAPERS, DEFAULT_CACHE_MAX_SIZE,
                     DEFAULT_OUTPUT_DIR)
from .errors import safe_error
from .indexing import build_index, qa_root
from .manifest import load_manifest
from .pipeline import run_batch
from .qa import answer_question
from .qa_index import QAIndex
from .sessions import SessionStore


def parse_size(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?B)?\s*", value, re.I)
    if not match:
        raise argparse.ArgumentTypeError("size must look like 500MB or 5GB")
    number = float(match.group(1))
    unit = (match.group(2) or "B").upper()
    multiplier = {"B": 1, "KB": 1024, "MB": 1024 ** 2,
                  "GB": 1024 ** 3, "TB": 1024 ** 4}[unit]
    return int(number * multiplier)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="p-helper",
        description="Generate evidence-backed Markdown summaries from PDF papers.",
    )
    parser.add_argument("input", help="A PDF file or a directory recursively containing PDFs.")
    parser.add_argument("--output", "-o", default=os.environ.get("P_HELPER_OUTPUT", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    parser.add_argument("--api-url", default=os.environ.get("P_HELPER_API_URL"))
    parser.add_argument("--model", default=os.environ.get("P_HELPER_MODEL"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cache-max-papers", type=int, default=DEFAULT_CACHE_MAX_PAPERS)
    parser.add_argument("--cache-max-size", type=parse_size, default=DEFAULT_CACHE_MAX_SIZE)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def build_index_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="p-helper index",
                                     description="Build the local paper QA index.")
    parser.add_argument("input")
    parser.add_argument("--output", "-o", default=os.environ.get("P_HELPER_OUTPUT", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def build_ask_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="p-helper ask",
                                     description="Ask evidence-backed questions about indexed papers.")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--output", "-o", default=os.environ.get("P_HELPER_OUTPUT", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--session")
    parser.add_argument("--paper", action="append", default=[])
    parser.add_argument("--language", choices=("zh", "en"))
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--api-url", default=os.environ.get("P_HELPER_API_URL"))
    parser.add_argument("--model", default=os.environ.get("P_HELPER_MODEL"))
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def build_sessions_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="p-helper sessions",
                                     description="List or delete local QA sessions.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    listing = subparsers.add_parser("list")
    listing.add_argument("--output", "-o", default=os.environ.get("P_HELPER_OUTPUT", DEFAULT_OUTPUT_DIR))
    delete = subparsers.add_parser("delete")
    delete.add_argument("session_id")
    delete.add_argument("--output", "-o", default=os.environ.get("P_HELPER_OUTPUT", DEFAULT_OUTPUT_DIR))
    return parser


def build_web_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="p-helper web",
                                     description="Run the local P-Helper browser interface.")
    parser.add_argument("--output", "-o", default=os.environ.get("P_HELPER_OUTPUT", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv=None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    values = list(argv if argv is not None else sys.argv[1:])
    command = values[0] if values else None
    if command == "index":
        return _index_main(values[1:])
    if command == "ask":
        return _ask_main(values[1:])
    if command == "sessions":
        return _sessions_main(values[1:])
    if command == "web":
        return _web_main(values[1:])
    parser = build_parser()
    args = parser.parse_args(values)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    api_key = os.environ.get("P_HELPER_API_KEY")
    missing = [name for name, value in (
        ("P_HELPER_API_KEY", api_key), ("--api-url/P_HELPER_API_URL", args.api_url),
        ("--model/P_HELPER_MODEL", args.model),
    ) if not value]
    if missing:
        parser.error("Missing required model configuration: " + ", ".join(missing))
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.cache_max_papers < 0 or args.cache_max_size < 0:
        parser.error("cache limits cannot be negative")
    config = Config(
        api_key=api_key, api_url=args.api_url, model=args.model,
        output_dir=args.output, language=args.language, workers=args.workers,
        force=args.force, cache_max_papers=args.cache_max_papers,
        cache_max_size=args.cache_max_size,
    )
    try:
        return run_batch(config, Path(args.input))
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Interrupted; completed papers were preserved.")
        return 130
    except ValueError as exc:
        parser.error(str(exc))
    return 2


def _configure_logging(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _index_main(argv) -> int:
    parser = build_index_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return build_index(Path(args.input), Path(args.output), force=args.force)
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Interrupted; completed indexes were preserved.")
        return 130
    except ValueError as exc:
        parser.error(str(exc))
    return 2


def _ask_main(argv) -> int:
    parser = build_ask_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    api_key = os.environ.get("P_HELPER_API_KEY")
    missing = [name for name, value in (
        ("P_HELPER_API_KEY", api_key), ("--api-url/P_HELPER_API_URL", args.api_url),
        ("--model/P_HELPER_MODEL", args.model),
    ) if not value]
    if missing:
        parser.error("Missing required model configuration: " + ", ".join(missing))
    if args.top_k is not None and not 1 <= args.top_k <= 20:
        parser.error("--top-k must be between 1 and 20")
    output_dir = Path(args.output).expanduser().resolve()
    qa_dir = qa_root(output_dir)
    index_path = qa_dir / "index.sqlite3"
    if not index_path.exists():
        parser.error(f"QA index does not exist: {index_path}")
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        parser.error(f"Manifest does not exist: {manifest_path}")
    config = Config(
        api_key=api_key, api_url=args.api_url, model=args.model,
        output_dir=str(output_dir), language=args.language or "zh",
    )
    try:
        with QAIndex(index_path) as index, SessionStore(qa_dir / "sessions.sqlite3") as sessions:
            manifest = load_manifest(manifest_path)
            indexed = {paper.fingerprint: paper for paper in index.list_papers()}
            allowed = {fingerprint for fingerprint, entry in manifest.papers.items()
                       if entry.index_status == "success" and entry.index_signature and
                       fingerprint in indexed and
                       indexed[fingerprint].signature == entry.index_signature}
            if not allowed:
                parser.error("QA index contains no papers")
            session, created = _resolve_session(parser, args, index, sessions, allowed)
            config = type(config)(**{**config.__dict__, "language": session.language})
            print(f"Session: {session.id}")
            if args.question is not None:
                return _ask_turn(config, session, args.question, index, sessions)
            while True:
                try:
                    question = input("Question> ").strip()
                except EOFError:
                    print()
                    _delete_empty_new_session(sessions, session, created)
                    return 0
                if question.lower() in {"exit", "quit"}:
                    _delete_empty_new_session(sessions, session, created)
                    return 0
                if not question:
                    continue
                result = _ask_turn(config, session, question, index, sessions)
                if result:
                    return result
    except KeyboardInterrupt:
        if "created" in locals() and created:
            with SessionStore(qa_dir / "sessions.sqlite3") as cleanup:
                _delete_empty_new_session(cleanup, session, created)
        print()
        return 130
    except ValueError as exc:
        parser.error(str(exc))
    except Exception as exc:
        if "created" in locals() and created:
            with SessionStore(qa_dir / "sessions.sqlite3") as cleanup:
                _delete_empty_new_session(cleanup, session, created)
        logging.getLogger(__name__).error(
            "Question answering failed: %s", safe_error(exc, api_key)
        )
        return 1


def _resolve_session(parser, args, index: QAIndex, sessions: SessionStore,
                     allowed: set[str]):
    selected = index.resolve_papers(args.paper, allowed) if args.paper else None
    if args.session:
        session = sessions.get(args.session)
        if session is None:
            parser.error(f"Session does not exist: {args.session}")
        available = set(index.resolve_papers([], allowed))
        missing = [item for item in session.fingerprints if item not in available]
        if missing:
            parser.error("Saved session contains papers no longer available in the QA index; "
                         "create a new session")
        if args.language is not None and args.language != session.language:
            parser.error("--language conflicts with the saved session")
        if args.top_k is not None and args.top_k != session.top_k:
            parser.error("--top-k conflicts with the saved session")
        if selected is not None and set(selected) != set(session.fingerprints):
            parser.error("--paper conflicts with the saved session")
        return session, False
    fingerprints = selected if selected is not None else index.resolve_papers([], allowed)
    return sessions.create(args.language or "zh", fingerprints, args.top_k or 8), True


def _ask_turn(config, session, question: str, index: QAIndex,
              sessions: SessionStore) -> int:
    answer, _ = answer_question(
        config, question, sessions.recent_turns(session.id), index,
        session.fingerprints, session.top_k,
    )
    print(answer)
    sessions.add_turn(session.id, question, answer)
    return 0


def _delete_empty_new_session(sessions: SessionStore, session, created: bool):
    if created and sessions.turn_count(session.id) == 0:
        sessions.delete(session.id)


def _sessions_main(argv) -> int:
    parser = build_sessions_parser()
    args = parser.parse_args(argv)
    path = qa_root(Path(args.output).expanduser().resolve()) / "sessions.sqlite3"
    if not path.exists():
        if args.action == "list":
            print("No saved sessions.")
            return 0
        parser.error(f"Session store does not exist: {path}")
    with SessionStore(path) as sessions:
        if args.action == "list":
            for session in sessions.list():
                print(f"{session.id}\t{session.language}\t{len(session.fingerprints)} papers\t{session.last_used_at}")
            return 0
        if not sessions.delete(args.session_id):
            parser.error(f"Session does not exist: {args.session_id}")
    print(f"Deleted session: {args.session_id}")
    return 0


def _web_main(argv) -> int:
    parser = build_web_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    _configure_logging(args.verbose)
    from .web import run_web
    try:
        run_web(Path(args.output), args.port)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
