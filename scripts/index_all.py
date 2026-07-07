"""
MedAssist RAG — scripts/index_all.py
Script CLI d'orchestration : charge tous les types de documents
et les indexe dans le vector store.

Usage:
    python scripts/index_all.py
    python scripts/index_all.py --reset
    python scripts/index_all.py --no-scans
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.vector_store import get_embeddings, get_vector_store
from src.ingestion.pdf_loader import load_all_pdfs
from src.ingestion.scan_loader import load_all_scans
from src.ingestion.table_loader import load_all_tables
from src.ingestion.web_loader import load_all_web

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Indexe tous les documents MedAssist")
    parser.add_argument("--reset", action="store_true", help="Réinitialise l'index avant d'indexer")
    parser.add_argument("--no-pdfs", action="store_true", help="Ignorer les PDFs texte")
    parser.add_argument("--no-tables", action="store_true", help="Ignorer les tableaux")
    parser.add_argument("--no-scans", action="store_true", help="Ignorer les scans OCR")
    parser.add_argument("--no-web", action="store_true", help="Ignorer le contenu web")
    parser.add_argument(
        "--scan-max-pages", type=int, default=None, help="Limiter les pages OCR par fichier"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    all_chunks = []

    if not args.no_pdfs:
        pdf_chunks = load_all_pdfs("data/01_pdf_text", max_chunks_per_file=300)
        all_chunks.extend(pdf_chunks)
        logger.info("PDFs: %d chunks", len(pdf_chunks))

    if not args.no_tables:
        table_chunks = load_all_tables("data/02_tables")
        all_chunks.extend(table_chunks)
        logger.info("Tableaux: %d chunks", len(table_chunks))

    if not args.no_scans:
        scan_chunks = load_all_scans("data/03_scans", max_pages_per_file=args.scan_max_pages)
        all_chunks.extend(scan_chunks)
        logger.info("Scans: %d chunks", len(scan_chunks))

    if not args.no_web:
        web_chunks = load_all_web("data/04_web")
        all_chunks.extend(web_chunks)
        logger.info("Web: %d chunks", len(web_chunks))

    logger.info("Total à indexer: %d chunks", len(all_chunks))

    if not all_chunks:
        logger.warning("Aucun chunk à indexer — arrêt.")
        return

    embeddings = get_embeddings()
    vs = get_vector_store(embeddings)

    existing = vs.count()
    if existing > 0:
        if args.reset:
            vs.reset_collection()
            logger.info("Collection réinitialisée (%d documents supprimés)", existing)
        else:
            logger.info(
                "Index existant: %d documents. Relancer avec --reset pour ré-indexer.", existing
            )
            return

    n_indexed = vs.add_documents(all_chunks)
    elapsed = time.time() - t0

    logger.info(
        "Indexation terminée: %d chunks en %.1fs (%.0f chunks/s)",
        n_indexed,
        elapsed,
        n_indexed / elapsed if elapsed > 0 else 0,
    )
    logger.info("Index total: %d chunks", vs.count())


if __name__ == "__main__":
    main()
