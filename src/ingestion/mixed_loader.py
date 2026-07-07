"""
MedAssist RAG — src/ingestion/mixed_loader.py
Charge les documents mixtes (texte + tableaux + images)
via Unstructured.io et produit des chunks typés.

Cas d'usage : calendrier_vaccination_national.pdf
  - Texte narratif → chunking standard
  - Tableaux       → row-level chunks
  - Titres         → enrichissement métadonnées
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────────────────────────
# Métadonnées connues
# ─────────────────────────────────────────────────────────────────
MIXED_METADATA_MAP: dict[str, dict[str, Any]] = {
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


# ─────────────────────────────────────────────────────────────────
# Dataclass résultat
# ─────────────────────────────────────────────────────────────────
@dataclass
class MixedLoaderResult:
    file: str
    chunks: list[Document]
    n_chunks: int
    element_counts: dict[str, int]
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.chunks) > 0


# ─────────────────────────────────────────────────────────────────
# Nettoyage texte extrait par Unstructured
# ─────────────────────────────────────────────────────────────────
def clean_element_text(text: str) -> str:
    """Nettoie le texte d'un élément Unstructured."""
    if not text:
        return ""
    text = re.sub(r"[^\x20-\x7E\x80-\xFF\n]", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────
# Traitement élément par élément Unstructured
# ─────────────────────────────────────────────────────────────────
def process_elements(
    elements: list,
    file_metadata: dict[str, Any],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> tuple[list[Document], dict[str, int]]:
    """
    Traite chaque élément Unstructured selon son type.

    Types gérés :
      NarrativeText → chunking récursif standard
      Table         → description NL (header + lignes)
      Title         → enrichit le contexte du chunk suivant
      ListItem      → agrégé avec les autres items de la liste
      Header/Footer → ignorés
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = []
    element_counts: dict[str, int] = {}
    chunk_index = 0
    current_section = ""  # Titre de section courant
    list_buffer: list[str] = []  # Buffer pour agréger les ListItems

    def flush_list_buffer():
        """Vide le buffer de liste et crée un chunk."""
        nonlocal chunk_index
        if not list_buffer:
            return
        list_text = "\n".join(f"• {item}" for item in list_buffer)
        if len(list_text) >= 50:
            meta = build_metadata(file_metadata, chunk_index, current_section, "list_items")
            chunks.append(Document(page_content=list_text, metadata=meta))
            chunk_index += 1
        list_buffer.clear()

    def build_metadata(fmeta: dict, idx: int, section: str, etype: str) -> dict[str, Any]:
        return {
            "source": fmeta.get("source", ""),
            "chunk_id": f"{Path(fmeta['source']).stem}_mixed_{idx}",
            "page": 0,
            "chunk_index": idx,
            "title": fmeta.get("title", ""),
            "doc_type": fmeta.get("doc_type", ""),
            "source_org": fmeta.get("source_org", "MSPS Maroc"),
            "year": fmeta.get("year", 0),
            "speciality": fmeta.get("speciality", ""),
            "language": fmeta.get("language", "fr"),
            "country": "MA",
            "topics": ", ".join(fmeta.get("topics", [])),
            "section": section,
            "chunk_size": 0,  # mis à jour après
            "chunking_method": "unstructured",
            "element_type": etype,
        }

    for element in elements:
        etype = type(element).__name__
        element_counts[etype] = element_counts.get(etype, 0) + 1

        raw_text = getattr(element, "text", "") or ""
        text = clean_element_text(raw_text)

        if not text or len(text) < 20:
            continue

        # ── Titre : mise à jour du contexte de section ──────────
        if etype in ("Title", "Header"):
            flush_list_buffer()
            current_section = text[:80]
            continue

        # ── Footer : ignoré ──────────────────────────────────────
        if etype == "Footer":
            continue

        # ── Élément de liste : aggréger ──────────────────────────
        if etype == "ListItem":
            list_buffer.append(text)
            continue

        # ── Texte narratif : chunking récursif ───────────────────
        if etype in ("NarrativeText", "Text", "UncategorizedText"):
            flush_list_buffer()
            sub_chunks = splitter.split_text(text)
            for sub in sub_chunks:
                sub = sub.strip()
                if len(sub) < 50:
                    continue
                meta = build_metadata(file_metadata, chunk_index, current_section, "narrative_text")
                meta["chunk_size"] = len(sub)
                chunks.append(Document(page_content=sub, metadata=meta))
                chunk_index += 1
            continue

        # ── Tableau : description NL ─────────────────────────────
        if etype == "Table":
            flush_list_buffer()
            # Unstructured retourne le tableau en HTML ou texte brut
            table_text = text

            # Construire un chunk descriptif
            table_chunk = (
                f"Tableau {current_section + ' — ' if current_section else ''}"
                f"Vaccins et calendrier :\n{table_text}"
            )
            if len(table_chunk) > 50:
                meta = build_metadata(file_metadata, chunk_index, current_section, "table_content")
                meta["chunk_size"] = len(table_chunk)
                chunks.append(Document(page_content=table_chunk, metadata=meta))
                chunk_index += 1
            continue

        # ── Autres types : traiter comme texte si assez long ─────
        flush_list_buffer()
        if len(text) >= 80:
            meta = build_metadata(file_metadata, chunk_index, current_section, etype.lower())
            meta["chunk_size"] = len(text)
            chunks.append(Document(page_content=text, metadata=meta))
            chunk_index += 1

    # Vider le buffer final
    flush_list_buffer()

    return chunks, element_counts


# ─────────────────────────────────────────────────────────────────
# Charger un document mixte
# ─────────────────────────────────────────────────────────────────
def load_mixed(
    file_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    strategy: str = "fast",
) -> MixedLoaderResult:
    """
    Charge un document mixte via Unstructured.io.

    Args:
        file_path  : chemin vers le fichier
        chunk_size : taille des chunks texte
        strategy   : "fast" (rapide) | "hi_res" (meilleur, nécessite pdf2image)
    """
    path = Path(file_path)
    filename = path.name
    errors = []

    print(f"\n  Chargement mixte : {filename}")

    # Vérifier Unstructured
    try:
        from unstructured.partition.auto import partition
    except ImportError:
        errors.append("unstructured non installé.\n" "Installer : pip install unstructured")
        return MixedLoaderResult(
            file=filename, chunks=[], n_chunks=0, element_counts={}, errors=errors
        )

    # Métadonnées
    known_meta = MIXED_METADATA_MAP.get(filename, {})
    file_metadata = {
        "source": str(path),
        "title": known_meta.get("title", filename),
        **known_meta,
    }

    # Partitionnement avec Unstructured
    try:
        print(f"     -> Analyse Unstructured (strategy={strategy})...")
        elements = partition(
            filename=file_path,
            strategy=strategy,
            languages=["fra", "eng"],
            include_page_breaks=False,
        )
        print(f"     -> {len(elements)} éléments détectés")

        # Compter les types
        type_counts: dict[str, int] = {}
        for el in elements:
            t = type(el).__name__
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"     -> Types : {dict(list(type_counts.items())[:6])}")

    except Exception as e:
        errors.append(f"Unstructured échoué : {e}")
        print(f"     ERREUR Unstructured : {e}")
        print("     -> Fallback : lecture PDF simple via pypdf")

        # Fallback : lire comme PDF texte classique
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from pypdf import PdfReader

            reader = PdfReader(file_path)
            all_text = ""
            for page in reader.pages:
                all_text += (page.extract_text() or "") + "\n\n"

            if all_text.strip():
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ". ", " "],
                    length_function=len,
                    is_separator_regex=False,
                )
                fallback_chunks = []
                for idx, chunk_text in enumerate(splitter.split_text(all_text)):
                    if len(chunk_text.strip()) < 50:
                        continue
                    meta = {
                        "source": str(path),
                        "chunk_id": f"{path.stem}_fallback_{idx}",
                        "page": 0,
                        "chunk_index": idx,
                        "title": file_metadata.get("title", filename),
                        "doc_type": file_metadata.get("doc_type", ""),
                        "source_org": file_metadata.get("source_org", "MSPS Maroc"),
                        "year": file_metadata.get("year", 0),
                        "speciality": file_metadata.get("speciality", ""),
                        "language": file_metadata.get("language", "fr"),
                        "country": "MA",
                        "topics": ", ".join(file_metadata.get("topics", [])),
                        "chunk_size": len(chunk_text.strip()),
                        "chunking_method": "fallback_pypdf",
                        "element_type": "narrative_text",
                    }
                    fallback_chunks.append(Document(page_content=chunk_text.strip(), metadata=meta))

                print(f"     -> Fallback OK : {len(fallback_chunks)} chunks")
                return MixedLoaderResult(
                    file=filename,
                    chunks=fallback_chunks,
                    n_chunks=len(fallback_chunks),
                    element_counts={"fallback_pypdf": len(fallback_chunks)},
                    errors=errors,
                )
        except Exception as e2:
            errors.append(f"Fallback aussi échoué : {e2}")

        return MixedLoaderResult(
            file=filename, chunks=[], n_chunks=0, element_counts={}, errors=errors
        )

    # Traitement des éléments
    chunks, element_counts = process_elements(elements, file_metadata, chunk_size, chunk_overlap)
    print(f"     -> {len(chunks)} chunks produits")

    return MixedLoaderResult(
        file=filename,
        chunks=chunks,
        n_chunks=len(chunks),
        element_counts=element_counts,
        errors=errors,
    )


# ─────────────────────────────────────────────────────────────────
# Charger tous les documents mixtes
# ─────────────────────────────────────────────────────────────────
def load_all_mixed(
    folder: str = "data/05_mixed",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """Charge tous les PDFs du dossier mixte."""

    folder_path = Path(folder)
    pdf_files = sorted(folder_path.glob("*.pdf"))

    if not pdf_files:
        print(f"  Aucun fichier trouvé dans {folder}")
        return []

    print(f"\n{'='*55}")
    print(f"  Chargement de {len(pdf_files)} document(s) mixte(s)")
    print(f"{'='*55}")

    all_chunks: list[Document] = []
    results = []

    for pdf_file in pdf_files:
        result = load_mixed(str(pdf_file), chunk_size, chunk_overlap)
        results.append(result)
        if result.ok:
            all_chunks.extend(result.chunks)
            print("     OK")
        else:
            print(f"     ECHEC : {result.errors[0][:80] if result.errors else '?'}")

    print(f"\n{'='*55}")
    print("  RESUME INGESTION MIXTE")
    print(f"{'='*55}")
    print(f"  Fichiers chargés : {sum(1 for r in results if r.ok)}/{len(results)}")
    print(f"  Total chunks     : {len(all_chunks)}")

    if all_chunks:
        etypes: dict[str, int] = {}
        for c in all_chunks:
            et = c.metadata.get("element_type", "inconnu")
            etypes[et] = etypes.get(et, 0) + 1
        print("  Types de chunks :")
        for et, count in sorted(etypes.items(), key=lambda x: -x[1]):
            print(f"    - {et:<20} : {count}")

    return all_chunks


# ─────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n=== Test MedAssist Mixed Loader ===")

    chunks = load_all_mixed("data/05_mixed")

    if not chunks:
        print("\nAucun chunk — vérifier data/05_mixed/")
        exit(1)

    print(f"\n{'='*55}")
    print("  APERCU — 5 premiers chunks")
    print(f"{'='*55}")
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n  Chunk {i+1}")
        print(f"  Type     : {chunk.metadata.get('element_type')}")
        print(f"  Section  : {chunk.metadata.get('section', '')[:40]}")
        print(f"  Taille   : {len(chunk.page_content)} chars")
        print(f"  Texte    : {chunk.page_content[:200]}...")

    with open("data/mixed_ingestion_report.json", "w", encoding="utf-8") as f:
        json.dump({"total_chunks": len(chunks)}, f, ensure_ascii=False, indent=2)

    print("\n  Rapport : data/mixed_ingestion_report.json")
    print("  Prochaine etape : mettre a jour vector_store.py")
    print("  -> Ajouter load_all_web() + load_all_mixed()")
