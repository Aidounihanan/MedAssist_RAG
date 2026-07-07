"""Tests pour src/ingestion/pdf_loader.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from src.ingestion.pdf_loader import (
    clean_extracted_text,
    load_all_pdfs,
    merge_short_fragments,
)


def test_clean_extracted_text_removes_page_numbers():
    raw = "Contenu médical important\n- 12 -\nSuite du contenu"
    cleaned = clean_extracted_text(raw)
    assert "- 12 -" not in cleaned
    assert "Contenu médical important" in cleaned


def test_clean_extracted_text_merges_hyphenated_words():
    raw = "Le traite-\nment doit être suivi"
    cleaned = clean_extracted_text(raw)
    assert "traitement" in cleaned


def test_clean_extracted_text_handles_empty_input():
    assert clean_extracted_text("") == ""
    assert clean_extracted_text(None) == ""


def test_merge_short_fragments_combines_short_chunks():
    chunks = [
        Document(page_content="court", metadata={"source": "a.pdf"}),
        Document(
            page_content="un texte suffisamment long pour ne pas être fusionné",
            metadata={"source": "a.pdf"},
        ),
    ]
    merged = merge_short_fragments(chunks, min_length=50)
    assert len(merged) == 1
    assert "court" in merged[0].page_content


def test_merge_short_fragments_keeps_long_chunks_separate():
    long_text_a = "a" * 150
    long_text_b = "b" * 150
    chunks = [
        Document(page_content=long_text_a, metadata={}),
        Document(page_content=long_text_b, metadata={}),
    ]
    merged = merge_short_fragments(chunks, min_length=100)
    assert len(merged) == 2


def test_load_all_pdfs_returns_empty_for_missing_folder(tmp_path):
    chunks = load_all_pdfs(str(tmp_path / "nonexistent"))
    assert chunks == []


def test_load_all_pdfs_respects_max_chunks_per_file(tmp_path):
    # Vérifie que la limite est bien appliquée si un dossier contient
    # des PDFs réels (test d'intégration léger — skip si data/ absent)
    data_dir = Path("data/01_pdf_text")
    if not data_dir.exists():
        return

    chunks = load_all_pdfs(str(data_dir), max_chunks_per_file=10)
    files_seen = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "")
        files_seen[src] = files_seen.get(src, 0) + 1

    for count in files_seen.values():
        assert count <= 10
