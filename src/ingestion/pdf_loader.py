"""
MedAssist RAG — src/ingestion/pdf_loader.py
Charge les PDFs texte MSPS, extrait le contenu, enrichit les métadonnées,
produit des chunks prêts pour l'embedding.

Module de production — aucune logique de test ici.
Tests : tests/test_pdf_loader.py
Orchestration CLI : scripts/index_all.py
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Métadonnées connues par fichier
# ─────────────────────────────────────────────────────────────────
PDF_METADATA_MAP: Dict[str, Dict[str, Any]] = {
    "guide_risque_cardiovasculaire_msps.pdf": {
        "title": "Guide d'Évaluation et de Prise en Charge du Risque Cardiovasculaire",
        "speciality": "cardiologie",
        "doc_type": "guideline_national",
        "source_org": "MSPS Maroc",
        "year": 2019,
        "language": "fr",
        "topics": ["HTA", "diabète", "risque cardiovasculaire", "prévention"],
    },
    "guide_tuberculose_msps.pdf": {
        "title": "Guide de Prise en Charge de la Tuberculose — MSPS",
        "speciality": "infectiologie",
        "doc_type": "guideline_national",
        "source_org": "MSPS Maroc",
        "year": 2020,
        "language": "fr",
        "topics": ["tuberculose", "TB", "traitement", "dépistage", "diagnostic"],
    },
    "MS_Rapport_Final_PNSF_2018.pdf": {
        "title": "Enquête Nationale sur la Population et la Santé Familiale 2018",
        "speciality": "epidemiologie",
        "doc_type": "rapport_national",
        "source_org": "MSPS Maroc",
        "year": 2018,
        "language": "fr",
        "topics": ["épidémiologie", "diabète", "HTA", "maladies chroniques", "Maroc"],
    },
    "Version_Finale_Lignes_Directrices_TEP_PNLAT_2023.pdf": {
        "title": "Lignes Directrices TEP — Programme National de Lutte Antituberculeuse 2023",
        "speciality": "infectiologie",
        "doc_type": "guideline_national",
        "source_org": "MSPS Maroc / PNLAT",
        "year": 2023,
        "language": "fr",
        "topics": ["tuberculose", "TEP", "PNLAT", "protocole", "traitement"],
    },
}


@dataclass
class LoaderResult:
    """Résultat du chargement d'un PDF."""
    file: str
    chunks: List[Document]
    n_pages: int
    n_chunks: int
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.chunks) > 0


def clean_extracted_text(text: str) -> str:
    """
    Nettoie le texte brut extrait d'un PDF médical.
    Supprime numéros de page, caractères parasites, normalise les espaces.
    """
    if not text:
        return ""

    text = re.sub(r'[^\x20-\x7E\x80-\xFF\n]', ' ', text)

    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        if re.fullmatch(r'[-]\s*\d{1,3}\s*[-]?', line):
            continue
        if re.match(r'^[Pp]age\s+\d+', line):
            continue
        cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)
    text = re.sub(r'-\n([a-z\xe0-\xff])', r'\1', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def extract_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """Extrait le texte de chaque page avec métadonnées de position."""
    reader = PdfReader(pdf_path)
    pages = []

    for i, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        clean = clean_extracted_text(raw_text)
        if clean:
            pages.append({
                "page_num": i + 1,
                "text": clean,
                "char_count": len(clean),
            })

    return pages


def merge_short_fragments(
    chunks: List[Document],
    min_length: int = 100,
) -> List[Document]:
    """
    Fusionne les chunks trop courts avec le suivant.
    Corrige les logigrammes dont le texte est fragmenté.
    """
    if not chunks:
        return chunks

    merged = []
    buffer = None

    for chunk in chunks:
        if buffer is None:
            buffer = chunk
            continue
        if len(buffer.page_content) < min_length:
            combined = buffer.page_content.strip() + " " + chunk.page_content.strip()
            buffer = Document(page_content=combined, metadata=buffer.metadata)
        else:
            merged.append(buffer)
            buffer = chunk

    if buffer:
        merged.append(buffer)

    return merged


def chunk_medical_pdf(
    pages: List[Dict],
    file_metadata: Dict[str, Any],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Document]:
    """Chunking adapté aux documents médicaux avec fusion des fragments courts."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n\u2022", "\n-", "\n", ". ", ", ", " "],
        length_function=len,
        is_separator_regex=False,
    )

    all_chunks = []
    chunk_index = 0

    for page_data in pages:
        page_text = page_data["text"]
        page_num = page_data["page_num"]

        for chunk_text in splitter.split_text(page_text):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < 30:
                continue

            metadata = {
                "source":           file_metadata.get("source", ""),
                "chunk_id":         f"{Path(file_metadata['source']).stem}_p{page_num}_c{chunk_index}",
                "page":             page_num,
                "chunk_index":      chunk_index,
                "title":            file_metadata.get("title", ""),
                "doc_type":         file_metadata.get("doc_type", ""),
                "source_org":       file_metadata.get("source_org", "MSPS Maroc"),
                "year":             file_metadata.get("year", 0),
                "speciality":       file_metadata.get("speciality", ""),
                "language":         file_metadata.get("language", "fr"),
                "country":          "MA",
                "topics":           ", ".join(file_metadata.get("topics", [])),
                "chunk_size":       len(chunk_text),
                "chunking_method":  "recursive_character",
            }

            all_chunks.append(Document(page_content=chunk_text, metadata=metadata))
            chunk_index += 1

    return merge_short_fragments(all_chunks, min_length=100)


def load_pdf(
    pdf_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> LoaderResult:
    """
    Charge un PDF médical et retourne les chunks prêts pour l'embedding.

    Args:
        pdf_path: chemin vers le fichier PDF
        chunk_size: taille max d'un chunk en caractères
        chunk_overlap: chevauchement entre chunks consécutifs

    Returns:
        LoaderResult avec chunks enrichis et statistiques
    """
    path = Path(pdf_path)
    filename = path.name
    errors: List[str] = []

    known_meta = PDF_METADATA_MAP.get(filename, {})
    file_metadata = {
        "source": str(path),
        "title": known_meta.get("title", filename),
        **known_meta,
    }

    try:
        pages = extract_pages(pdf_path)
        logger.info("PDF %s: %d pages extraites", filename, len(pages))
    except Exception as e:
        errors.append(f"Extraction échouée: {e}")
        logger.error("Extraction échouée pour %s: %s", filename, e)
        return LoaderResult(file=filename, chunks=[], n_pages=0, n_chunks=0, errors=errors)

    if not pages:
        errors.append("Aucun texte extractible — PDF probablement scanné")
        logger.warning("%s: aucun texte extractible", filename)
        return LoaderResult(file=filename, chunks=[], n_pages=0, n_chunks=0, errors=errors)

    try:
        chunks = chunk_medical_pdf(pages, file_metadata, chunk_size, chunk_overlap)
        logger.info("PDF %s: %d chunks produits", filename, len(chunks))
    except Exception as e:
        errors.append(f"Chunking échoué: {e}")
        logger.error("Chunking échoué pour %s: %s", filename, e)
        return LoaderResult(file=filename, chunks=[], n_pages=len(pages), n_chunks=0, errors=errors)

    return LoaderResult(
        file=filename, chunks=chunks, n_pages=len(pages),
        n_chunks=len(chunks), errors=errors,
    )


def load_all_pdfs(
    folder: str = "data/01_pdf_text",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    max_chunks_per_file: Optional[int] = 300,
) -> List[Document]:
    """
    Charge tous les PDFs d'un dossier avec équilibrage par spécialité.

    Args:
        folder: dossier contenant les PDFs
        chunk_size: taille max d'un chunk
        chunk_overlap: chevauchement entre chunks
        max_chunks_per_file: limite par fichier pour équilibrer les spécialités
                              (None = pas de limite). Évite qu'un document
                              volumineux ne domine l'index au détriment
                              des autres spécialités.

    Returns:
        Liste de tous les chunks combinés
    """
    folder_path = Path(folder)
    pdf_files = sorted(folder_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning("Aucun PDF trouvé dans %s", folder)
        return []

    all_chunks: List[Document] = []

    for pdf_file in pdf_files:
        result = load_pdf(str(pdf_file), chunk_size, chunk_overlap)

        if not result.ok:
            logger.warning("Échec chargement %s: %s", result.file, result.errors)
            continue

        chunks_to_add = result.chunks
        if max_chunks_per_file and len(chunks_to_add) > max_chunks_per_file:
            original = len(chunks_to_add)
            step = max(1, original // max_chunks_per_file)
            chunks_to_add = chunks_to_add[::step][:max_chunks_per_file]
            logger.info(
                "%s: limité à %d chunks (sur %d) pour équilibrage",
                result.file, len(chunks_to_add), original,
            )

        all_chunks.extend(chunks_to_add)

    logger.info("Ingestion PDFs terminée: %d chunks au total", len(all_chunks))
    return all_chunks