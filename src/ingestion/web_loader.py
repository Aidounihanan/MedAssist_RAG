"""
MedAssist RAG — src/ingestion/web_loader.py
Charge les pages web scrapées (JSONs dans data/04_web/)
et produit des chunks texte prêts pour l'embedding.

Module de production — aucune logique de test ici.
Tests : tests/test_web_loader.py
Orchestration CLI : scripts/index_all.py
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


SHAREPOINT_NOISE = [
    "Activer le mode plus accessible",
    "Désactiver le mode plus accessible",
    "Ignorer les commandes du ruban",
    "Passer au contenu principal",
    "Désactiver les animations",
    "Administration Centrale",
    "Directions Régionales",
    "Etablissements publics",
    "Listes Nationales",
    "Pharmacovigilance",
    "Actuellement sélectionné",
    "Actuellement sélec",
    "Ouvrir le menu",
    "Fermer le menu",
    "Aller au contenu",
    "Fil d'Ariane",
    "Partager cette page",
    "Imprimer cette page",
    "Version imprimable",
    "Retour en haut",
    "Tous droits réservés",
    "Ministère de la Santé",
    "www.sante.gov.ma",
    "Rechercher",
    "Se connecter",
    "Mon compte",
    "Accueil",
    "Plan du site",
    "Mentions légales",
    "Politique de confidentialité",
    "Contact",
    "Copyright",
]

MIN_LINE_LENGTH = 30


@dataclass
class WebLoaderResult:
    """Résultat du chargement d'une page web scrapée."""

    file: str
    chunks: list[Document]
    n_chunks: int
    word_count_before: int
    word_count_after: int
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.chunks) > 0

    @property
    def noise_ratio(self) -> float:
        if self.word_count_before == 0:
            return 0.0
        return round((1 - self.word_count_after / self.word_count_before) * 100, 1)


def clean_sharepoint_text(text: str) -> str:
    """Nettoie le texte d'une page SharePoint MSPS."""
    if not text:
        return ""

    lines = text.split("\n")
    cleaned = []
    seen_lines = set()

    for line in lines:
        line = line.strip()

        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue

        if any(noise.lower() in line.lower() for noise in SHAREPOINT_NOISE):
            continue

        if len(line) < MIN_LINE_LENGTH:
            continue

        alpha_ratio = sum(1 for c in line if c.isalpha()) / len(line)
        if alpha_ratio < 0.4:
            continue

        line_key = line.lower()[:50]
        if line_key in seen_lines:
            continue
        seen_lines.add(line_key)

        if re.match(r"^https?://", line) and len(line.split()) == 1:
            continue

        cleaned.append(line)

    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_web_json(
    json_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> WebLoaderResult:
    """
    Charge un fichier JSON produit par le scraper web.

    Args:
        json_path: chemin vers le fichier JSON scrapé
        chunk_size: taille des chunks
        chunk_overlap: chevauchement entre chunks

    Returns:
        WebLoaderResult avec chunks et statistiques de nettoyage
    """
    path = Path(json_path)
    filename = path.name
    errors: list[str] = []

    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        errors.append(f"Lecture JSON échouée: {e}")
        logger.error("Lecture échouée pour %s: %s", filename, e)
        return WebLoaderResult(
            file=filename,
            chunks=[],
            n_chunks=0,
            word_count_before=0,
            word_count_after=0,
            errors=errors,
        )

    raw_content = data.get("content", "")
    word_count_before = len(raw_content.split())

    clean_content = clean_sharepoint_text(raw_content)
    word_count_after = len(clean_content.split())

    if word_count_after < 50:
        errors.append(f"Contenu insuffisant après nettoyage ({word_count_after} mots)")
        logger.warning(
            "%s: contenu insuffisant après nettoyage (%d mots)", filename, word_count_after
        )
        return WebLoaderResult(
            file=filename,
            chunks=[],
            n_chunks=0,
            word_count_before=word_count_before,
            word_count_after=word_count_after,
            errors=errors,
        )

    file_metadata = {
        "source": data.get("source_url", str(path)),
        "title": data.get("title", filename),
        "doc_type": "web_page",
        "source_org": "MSPS Maroc",
        "speciality": data.get("speciality", "general"),
        "language": data.get("language", "fr"),
        "country": data.get("country", "MA"),
        "scraped_at": data.get("scraped_at", ""),
        "year": 2024,
        "topics": data.get("description", ""),
    }

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ", ", " "],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = []
    for idx, chunk_text in enumerate(splitter.split_text(clean_content)):
        chunk_text = chunk_text.strip()
        if len(chunk_text) < 50:
            continue

        metadata = {
            "source": file_metadata["source"],
            "chunk_id": f"{path.stem}_chunk_{idx}",
            "page": 0,
            "chunk_index": idx,
            "title": file_metadata["title"],
            "doc_type": file_metadata["doc_type"],
            "source_org": file_metadata["source_org"],
            "year": file_metadata["year"],
            "speciality": file_metadata["speciality"],
            "language": file_metadata["language"],
            "country": file_metadata["country"],
            "topics": file_metadata["topics"],
            "scraped_at": file_metadata["scraped_at"],
            "chunk_size": len(chunk_text),
            "chunking_method": "recursive_character",
            "element_type": "web_content",
        }
        chunks.append(Document(page_content=chunk_text, metadata=metadata))

    logger.info(
        "Web %s: %d chunks produits (%.1f%% bruit supprimé)",
        filename,
        len(chunks),
        round((1 - word_count_after / max(word_count_before, 1)) * 100, 1),
    )

    return WebLoaderResult(
        file=filename,
        chunks=chunks,
        n_chunks=len(chunks),
        word_count_before=word_count_before,
        word_count_after=word_count_after,
        errors=errors,
    )


def load_all_web(
    folder: str = "data/04_web",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """
    Charge tous les fichiers JSON du dossier web.

    Args:
        folder: dossier contenant les pages web scrapées (JSON)
        chunk_size: taille des chunks
        chunk_overlap: chevauchement entre chunks

    Returns:
        Liste de tous les chunks combinés
    """
    folder_path = Path(folder)
    json_files = sorted(folder_path.glob("*.json"))

    if not json_files:
        logger.warning("Aucun fichier JSON trouvé dans %s", folder)
        return []

    all_chunks: list[Document] = []

    for json_file in json_files:
        result = load_web_json(str(json_file), chunk_size, chunk_overlap)
        if result.ok:
            all_chunks.extend(result.chunks)
        else:
            logger.warning("Échec chargement %s: %s", result.file, result.errors)

    logger.info("Ingestion web terminée: %d chunks au total", len(all_chunks))
    return all_chunks
