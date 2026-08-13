from pathlib import Path

import pytest

from frame.discovery import discover_pdfs, note_filename, sha256_file


def test_discovers_pdfs_recursively_in_stable_order(tmp_path):
    (tmp_path / "z.pdf").write_bytes(b"z")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "B.PDF").write_bytes(b"b")
    (tmp_path / "ignore.txt").write_text("x")
    root, files = discover_pdfs(tmp_path)
    assert root == tmp_path.resolve()
    assert [item.relative_to(root).as_posix() for item in files] == ["a/B.PDF", "z.pdf"]


def test_accepts_a_single_pdf(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"paper")
    root, files = discover_pdfs(pdf)
    assert root == tmp_path.resolve()
    assert files == [pdf.resolve()]


def test_rejects_a_non_pdf_file(tmp_path):
    path = tmp_path / "paper.txt"
    path.write_text("x")
    with pytest.raises(ValueError, match="Not a PDF"):
        discover_pdfs(path)


def test_fingerprint_and_filename_are_content_stable(tmp_path):
    path = tmp_path / "A paper (draft).pdf"
    path.write_bytes(b"same")
    digest = sha256_file(path)
    assert digest == sha256_file(path)
    assert note_filename(path, digest).startswith("A-paper--draft-")
    assert note_filename(path, digest).endswith(f"{digest[:12]}.md")
