"""
MedAssist RAG — src/ingestion/scan_loader.py
Charge les PDFs scannés via OCR et produit des chunks texte
prêts pour l'embedding.

Module de production — aucune logique de test ici.
Tests : tests/test_scan_loader.py
Orchestration CLI : scripts/index_all.py

Prérequis système (Windows) :
  - Tesseract installé : https://github.com/UB-Mannheim/tesseract/wiki
  - Chemin dans .env : TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()
logger = logging.getLogger(__name__)


SCAN_METADATA_MAP: Dict[str, Dict[str, Any]] = {
    "guide_avc_msps.pdf": {
        "title": "Guide AVC — Orientation et Prise en Charge",
        "speciality": "neurologie",
        "doc_type": "guideline_national",
        "source_org": "MSPS Maroc",
        "year": 2023,
        "language": "fr",
        "topics": ["AVC", "accident vasculaire cérébral", "urgence", "neurologie"],
    },
    "guide_sca_msps.pdf": {
        "title": "Guide SCA — Syndrome Coronarien Aigu",
        "speciality": "cardiologie",
        "doc_type": "guideline_national",
        "source_org": "MSPS Maroc",
        "year": 2023,
        "language": "fr",
        "topics": ["SCA", "infarctus", "coronarien", "urgence cardiaque"],
    },
    "guide_urgences_pediatriques_msps.pdf": {
        "title": "Guide des Urgences Pédiatriques",
        "speciality": "pediatrie",
        "doc_type": "guideline_national",
        "source_org": "MSPS Maroc",
        "year": 2022,
        "language": "fr",
        "topics": ["urgences", "pédiatrie", "enfant", "nourrisson", "détresse"],
    },
    "guide_Filières_DE_SOINS_UNCV_VF.pdf": {
        "title": "Filières de Soins et Protocoles SCA et AVC — Urgences Neuro-Cardiovasculaires",
        "speciality": "cardiologie",
        "doc_type": "guideline_national",
        "source_org": "MSPS Maroc",
        "year": 2023,
        "language": "fr",
        "topics": ["filières soins", "SCA", "AVC", "protocoles", "urgences"],
    },
    "calendrier_vaccination_national.pdf": {
        "title": "Calendrier National de Vaccination — MSPS Maroc",
        "speciality": "pediatrie",
        "doc_type": "guideline_national",
        "source_org": "MSPS Maroc",
        "year": 2024,
        "language": "fr",
        "topics": ["vaccination", "vaccins", "calendrier", "enfant", "immunisation"],
    },
}


@dataclass
class ScanLoaderResult:
    """Résultat du chargement OCR d'un PDF scanné."""
    file: str
    chunks: List[Document]
    n_pages: int
    n_chunks: int
    ocr_engine: str
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.chunks) > 0


def configure_tesseract() -> bool:
    """
    Configure le chemin Tesseract depuis .env ou chemins par défaut Windows.

    Returns:
        True si Tesseract est disponible, False sinon
    """
    import pytesseract

    custom_path = os.getenv("TESSERACT_CMD")
    if custom_path and Path(custom_path).exists():
        pytesseract.pytesseract.tesseract_cmd = custom_path
        return True

    default_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in default_paths:
        if Path(path).exists():
            pytesseract.pytesseract.tesseract_cmd = path
            return True

    return False


def clean_ocr_text(text: str) -> str:
    """
    Nettoie le texte brut issu de l'OCR.
    Plus agressif que le nettoyeur PDF texte : l'OCR produit
    davantage d'artefacts (lignes courtes, symboles parasites).
    """
    if not text:
        return ""

    text = re.sub(r'[^\x20-\x7E\x80-\xFF\n]', ' ', text)

    lines = text.split('\n')
    cleaned = []

    for line in lines:
        line = line.strip()
        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        if len(line) < 3:
            continue
        if re.fullmatch(r'[-]?\s*\d{1,3}\s*[-]?', line):
            continue

        non_alpha = sum(1 for c in line if not c.isalnum() and c not in ' .,;:-()/')
        if len(line) > 0 and non_alpha / len(line) > 0.5:
            continue

        cleaned.append(line)

    text = '\n'.join(cleaned)
    text = re.sub(r'-\n([a-z\xe0-\xff])', r'\1', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def ocr_pdf_tesseract(
    pdf_path: str,
    dpi: int = 250,
    lang: str = "fra+eng",
    max_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Extrait le texte d'un PDF scanné via Tesseract OCR.

    Args:
        pdf_path: chemin PDF
        dpi: résolution des images (250 recommandé)
        lang: langues Tesseract
        max_pages: limiter le nombre de pages (None = toutes)

    Returns:
        Liste de {page_num, text, char_count, confidence}
    """
    import pytesseract
    import fitz
    from PIL import Image
    import io

    reader = fitz.open(pdf_path)
    pages_data = []
    n_pages = min(len(reader), max_pages) if max_pages else len(reader)

    for i in range(n_pages):
        page = reader[i]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        try:
            config = "--oem 3 --psm 1"
            raw_text = pytesseract.image_to_string(img, lang=lang, config=config)
            data = pytesseract.image_to_data(
                img, lang=lang, config=config, output_type=pytesseract.Output.DICT
            )
            confidences = [c for c in data['conf'] if c != -1]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
        except Exception as e:
            logger.warning("OCR échoué page %d de %s: %s", i + 1, pdf_path, e)
            continue

        clean = clean_ocr_text(raw_text)
        if clean and len(clean) > 50:
            pages_data.append({
                "page_num": i + 1,
                "text": clean,
                "char_count": len(clean),
                "confidence": round(avg_conf, 1),
            })

    reader.close()
    return pages_data


def chunk_ocr_text(
    pages: List[Dict],
    file_metadata: Dict[str, Any],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Document]:
    """Chunking du texte OCR avec métadonnée de confiance par chunk."""

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
        confidence = page_data.get("confidence", 0)

        for chunk_text in splitter.split_text(page_text):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < 50:
                continue

            metadata = {
                "source":         file_metadata.get("source", ""),
                "chunk_id":       f"{Path(file_metadata['source']).stem}_ocr_p{page_num}_c{chunk_index}",
                "page":           page_num,
                "chunk_index":    chunk_index,
                "title":          file_metadata.get("title", ""),
                "doc_type":       file_metadata.get("doc_type", ""),
                "source_org":     file_metadata.get("source_org", "MSPS Maroc"),
                "year":           file_metadata.get("year", 0),
                "speciality":     file_metadata.get("speciality", ""),
                "language":       file_metadata.get("language", "fr"),
                "country":        "MA",
                "topics":         ", ".join(file_metadata.get("topics", [])),
                "chunk_size":     len(chunk_text),
                "chunking_method":"recursive_character",
                "element_type":   "ocr_text",
                "ocr_engine":     "tesseract",
                "ocr_confidence": confidence,
            }

            all_chunks.append(Document(page_content=chunk_text, metadata=metadata))
            chunk_index += 1

    return all_chunks


def load_scan(
    pdf_path: str,
    dpi: int = 250,
    lang: str = "fra+eng",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    max_pages: Optional[int] = None,
) -> ScanLoaderResult:
    """
    Charge un PDF scanné via OCR et retourne les chunks.

    Args:
        pdf_path: chemin vers le PDF scanné
        dpi: résolution OCR
        lang: langues Tesseract
        chunk_size: taille des chunks
        chunk_overlap: chevauchement
        max_pages: limiter les pages traitées (None = toutes)

    Returns:
        ScanLoaderResult avec chunks et statistiques OCR
    """
    path = Path(pdf_path)
    filename = path.name
    errors: List[str] = []

    if not configure_tesseract():
        msg = (
            "Tesseract non trouvé. Installer depuis "
            "https://github.com/UB-Mannheim/tesseract/wiki et configurer "
            "TESSERACT_CMD dans .env"
        )
        errors.append(msg)
        logger.error(msg)
        return ScanLoaderResult(file=filename, chunks=[], n_pages=0, n_chunks=0, ocr_engine="tesseract", errors=errors)

    known_meta = SCAN_METADATA_MAP.get(filename, {})
    file_metadata = {
        "source": str(path),
        "title": known_meta.get("title", filename),
        **known_meta,
    }

    try:
        pages = ocr_pdf_tesseract(pdf_path, dpi=dpi, lang=lang, max_pages=max_pages)
        logger.info("OCR %s: %d pages avec texte extractible", filename, len(pages))
    except Exception as e:
        errors.append(f"OCR échoué: {e}")
        logger.error("OCR échoué pour %s: %s", filename, e)
        return ScanLoaderResult(file=filename, chunks=[], n_pages=0, n_chunks=0, ocr_engine="tesseract", errors=errors)

    if not pages:
        errors.append("Aucun texte extrait par OCR")
        return ScanLoaderResult(file=filename, chunks=[], n_pages=0, n_chunks=0, ocr_engine="tesseract", errors=errors)

    try:
        chunks = chunk_ocr_text(pages, file_metadata, chunk_size, chunk_overlap)
    except Exception as e:
        errors.append(f"Chunking échoué: {e}")
        logger.error("Chunking échoué pour %s: %s", filename, e)
        return ScanLoaderResult(file=filename, chunks=[], n_pages=len(pages), n_chunks=0, ocr_engine="tesseract", errors=errors)

    return ScanLoaderResult(
        file=filename, chunks=chunks, n_pages=len(pages),
        n_chunks=len(chunks), ocr_engine="tesseract", errors=errors,
    )


def load_all_scans(
    folder: str = "data/03_scans",
    dpi: int = 250,
    lang: str = "fra+eng",
    max_pages_per_file: Optional[int] = None,
    min_confidence: float = 65.0,
) -> List[Document]:
    """
    Charge tous les PDFs scannés d'un dossier via OCR.

    Args:
        folder: dossier contenant les PDFs scannés
        dpi: résolution OCR
        lang: langues Tesseract
        max_pages_per_file: limiter les pages par fichier (None = toutes)
        min_confidence: seuil de confiance OCR minimal pour garder un chunk

    Returns:
        Liste de tous les chunks combinés, filtrés par qualité OCR
    """
    folder_path = Path(folder)
    pdf_files = sorted(folder_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning("Aucun PDF trouvé dans %s", folder)
        return []

    all_chunks: List[Document] = []

    for pdf_file in pdf_files:
        result = load_scan(str(pdf_file), dpi=dpi, lang=lang, max_pages=max_pages_per_file)
        if result.ok:
            all_chunks.extend(result.chunks)
        else:
            logger.warning("Échec OCR %s: %s", result.file, result.errors)

    before = len(all_chunks)
    all_chunks = [c for c in all_chunks if c.metadata.get("ocr_confidence", 0) >= min_confidence]
    if len(all_chunks) < before:
        logger.info("Filtre qualité OCR: %d -> %d chunks (seuil %.0f%%)", before, len(all_chunks), min_confidence)

    logger.info("Ingestion scans terminée: %d chunks au total", len(all_chunks))
    return all_chunks