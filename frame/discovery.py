"""PDF discovery and stable content identity."""

import hashlib
from pathlib import Path


def discover_pdfs(input_path: Path) -> tuple[Path, list[Path]]:
    path = input_path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {path}")
        return path.parent, [path]
    if not path.is_dir():
        raise ValueError(f"Input does not exist: {path}")
    files = [item for item in path.rglob("*")
             if item.is_file() and item.suffix.lower() == ".pdf"]
    return path, sorted(files, key=lambda item: item.relative_to(path).as_posix())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def note_filename(path: Path, fingerprint: str) -> str:
    stem = "".join(char if char.isalnum() or char in "-_" else "-"
                   for char in path.stem).strip("-") or "paper"
    return f"{stem}-{fingerprint[:12]}.md"

