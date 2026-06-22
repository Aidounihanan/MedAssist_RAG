"""
MedAssist RAG — src/ingestion/table_loader.py
Charge les fichiers Excel/XLS/CSV MSPS et produit des chunks texte
prêts pour l'embedding.

Module de production — aucune logique de test ici.
Tests : tests/test_table_loader.py
Orchestration CLI : scripts/index_all.py
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


TABLE_METADATA_MAP: Dict[str, Dict[str, Any]] = {
    "etablissements_sante_maroc.xlsx": {
        "title": "Établissements de Santé au Maroc",
        "speciality": "organisation_sanitaire",
        "doc_type": "dataset_officiel",
        "source_org": "MSPS Maroc",
        "year": 2024,
        "language": "fr",
        "topics": ["établissements", "hôpitaux", "centres santé", "Maroc", "régions"],
    },
    "infrastructures-privees-2024.xlsx": {
        "title": "Infrastructures Sanitaires Privées 2024",
        "speciality": "organisation_sanitaire",
        "doc_type": "dataset_officiel",
        "source_org": "MSPS Maroc",
        "year": 2024,
        "language": "fr",
        "topics": ["secteur privé", "cliniques", "cabinets", "infrastructures"],
    },
    "repartition-des-hopitaux-par-region-et-province-2024.xlsx": {
        "title": "Répartition des Hôpitaux par Région et Province 2024",
        "speciality": "organisation_sanitaire",
        "doc_type": "dataset_officiel",
        "source_org": "MSPS Maroc",
        "year": 2024,
        "language": "fr",
        "topics": ["hôpitaux", "régions", "provinces", "répartition", "Maroc"],
    },
    "repartition-des-hopitaux-par-region-et-province-2020.xlsx": {
        "title": "Répartition des Hôpitaux par Région et Province 2020",
        "speciality": "organisation_sanitaire",
        "doc_type": "dataset_officiel",
        "source_org": "MSPS Maroc",
        "year": 2020,
        "language": "fr",
        "topics": ["hôpitaux", "régions", "provinces", "répartition", "Maroc"],
    },
    "agences-cnss-2011.xls": {
        "title": "Agences CNSS 2011",
        "speciality": "organisation_sanitaire",
        "doc_type": "dataset_officiel",
        "source_org": "CNSS Maroc",
        "year": 2011,
        "language": "fr",
        "topics": ["CNSS", "agences", "couverture sociale", "Maroc"],
    },
    "offre-privees-2013.xls": {
        "title": "Offre de Soins Privée 2013",
        "speciality": "organisation_sanitaire",
        "doc_type": "dataset_officiel",
        "source_org": "MSPS Maroc",
        "year": 2013,
        "language": "fr",
        "topics": ["secteur privé", "offre de soins", "médecins", "Maroc"],
    },
}


@dataclass
class TableLoaderResult:
    """Résultat du chargement d'un fichier tabulaire."""
    file: str
    chunks: List[Document]
    n_sheets: int
    n_rows: int
    n_chunks: int
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.chunks) > 0


def clean_cell(value: Any) -> str:
    """Convertit une valeur de cellule en texte propre."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.match(r'^\d+\.0$', text):
        text = text[:-2]
    return text


def detect_headers(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Détecte la vraie ligne de headers dans un DataFrame.
    Certains fichiers MSPS ont des lignes de titre avant les vrais headers.

    Returns:
        Tuple (df_nettoyé, liste_headers)
    """
    unnamed_count = sum(1 for c in df.columns if 'Unnamed' in str(c))

    if unnamed_count <= len(df.columns) // 2:
        headers = [str(c).strip() for c in df.columns]
        return df, headers

    for i in range(min(5, len(df))):
        row = df.iloc[i].tolist()
        non_null = sum(1 for v in row if not pd.isna(v) and str(v).strip())
        if non_null >= len(df.columns) * 0.6:
            new_headers = [clean_cell(v) or f"Col_{j}" for j, v in enumerate(row)]
            df_clean = df.iloc[i + 1:].reset_index(drop=True)
            df_clean.columns = new_headers
            return df_clean, new_headers

    headers = [f"Colonne_{i+1}" for i in range(len(df.columns))]
    return df, headers


def row_level_chunks(
    df: pd.DataFrame,
    headers: List[str],
    file_metadata: Dict[str, Any],
    sheet_name: str = "Sheet1",
    max_rows: int = 500,
) -> List[Document]:
    """
    Chaque ligne du tableau devient un chunk autonome.
    Le header est répété dans chaque chunk pour le contexte.
    """
    chunks = []
    df_limited = df.head(max_rows)

    for row_idx, (_, row) in enumerate(df_limited.iterrows()):
        parts = []
        for header, value in zip(headers, row):
            val = clean_cell(value)
            if val:
                parts.append(f"{header}: {val}")

        if not parts:
            continue

        chunk_text = " | ".join(parts)
        if len(chunk_text) < 20:
            continue

        metadata = {
            "source":          file_metadata.get("source", ""),
            "chunk_id":        f"{Path(file_metadata['source']).stem}_sheet_{sheet_name}_row_{row_idx}",
            "sheet":           sheet_name,
            "row_index":       row_idx,
            "title":           file_metadata.get("title", ""),
            "doc_type":        file_metadata.get("doc_type", ""),
            "source_org":      file_metadata.get("source_org", "MSPS Maroc"),
            "year":            file_metadata.get("year", 0),
            "speciality":      file_metadata.get("speciality", ""),
            "language":        file_metadata.get("language", "fr"),
            "country":         "MA",
            "topics":          ", ".join(file_metadata.get("topics", [])),
            "chunk_size":      len(chunk_text),
            "chunking_method": "row_level",
            "element_type":    "table_row",
        }

        chunks.append(Document(page_content=chunk_text, metadata=metadata))

    return chunks


def table_summary_chunk(
    df: pd.DataFrame,
    headers: List[str],
    file_metadata: Dict[str, Any],
    sheet_name: str = "Sheet1",
) -> Optional[Document]:
    """
    Génère un chunk de résumé statistique du tableau (sans LLM).
    Utile pour les questions globales type "combien d'établissements ?".
    """
    n_rows = len(df)
    n_cols = len(headers)

    col_summaries = []
    for col in df.columns:
        try:
            col_clean = df[col].dropna()
            if len(col_clean) == 0:
                continue
            if pd.api.types.is_numeric_dtype(col_clean):
                col_summaries.append(
                    f"{col}: valeurs numériques de {col_clean.min():.0f} à {col_clean.max():.0f}"
                )
            else:
                unique_vals = col_clean.unique()
                if len(unique_vals) <= 15:
                    vals_str = ", ".join([str(v)[:30] for v in unique_vals[:10]])
                    col_summaries.append(f"{col}: {vals_str}")
                else:
                    col_summaries.append(f"{col}: {len(unique_vals)} valeurs uniques")
        except Exception:
            continue

    summary_text = (
        f"Tableau : {file_metadata.get('title', 'Données MSPS')}\n"
        f"Source : {file_metadata.get('source_org', 'MSPS Maroc')} ({file_metadata.get('year', '')})\n"
        f"Contenu : {n_rows} lignes, {n_cols} colonnes\n"
        f"Colonnes : {', '.join(headers[:10])}\n"
    )
    if col_summaries:
        summary_text += "Données :\n" + "\n".join(f"  - {s}" for s in col_summaries[:10])

    metadata = {
        "source":          file_metadata.get("source", ""),
        "chunk_id":        f"{Path(file_metadata['source']).stem}_sheet_{sheet_name}_summary",
        "sheet":           sheet_name,
        "row_index":       -1,
        "title":           file_metadata.get("title", ""),
        "doc_type":        file_metadata.get("doc_type", ""),
        "source_org":      file_metadata.get("source_org", "MSPS Maroc"),
        "year":            file_metadata.get("year", 0),
        "speciality":      file_metadata.get("speciality", ""),
        "language":        file_metadata.get("language", "fr"),
        "country":         "MA",
        "topics":          ", ".join(file_metadata.get("topics", [])),
        "chunk_size":      len(summary_text),
        "chunking_method": "table_summary",
        "element_type":    "table_summary",
    }

    return Document(page_content=summary_text, metadata=metadata)


def read_table_file(file_path: str) -> Dict[str, pd.DataFrame]:
    """
    Lit un fichier tabulaire et retourne un dict {sheet_name: DataFrame}.
    Gère .xlsx, .xls, .csv.
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    sheets: Dict[str, pd.DataFrame] = {}

    if ext == ".csv":
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                sheets["Sheet1"] = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

    elif ext == ".xls":
        xl = pd.ExcelFile(file_path, engine="xlrd")
        for sheet in xl.sheet_names:
            sheets[str(sheet)] = xl.parse(sheet)

    elif ext in (".xlsx", ".xlsm"):
        xl = pd.ExcelFile(file_path, engine="openpyxl")
        for sheet in xl.sheet_names:
            sheets[str(sheet)] = xl.parse(sheet)

    else:
        raise ValueError(f"Format non supporté: {ext}")

    return sheets


def load_table(
    file_path: str,
    max_rows_per_sheet: int = 500,
) -> TableLoaderResult:
    """
    Charge un fichier Excel/CSV et produit des chunks prêts pour l'embedding.

    Stratégie combinée :
      - 1 chunk de résumé par feuille (vue globale)
      - N chunks row-level (une ligne = un chunk)

    Args:
        file_path: chemin vers le fichier
        max_rows_per_sheet: limite de lignes par feuille

    Returns:
        TableLoaderResult avec chunks et statistiques
    """
    path = Path(file_path)
    filename = path.name
    errors: List[str] = []

    known_meta = TABLE_METADATA_MAP.get(filename, {})
    file_metadata = {
        "source": str(path),
        "title": known_meta.get("title", filename),
        **known_meta,
    }

    try:
        sheets = read_table_file(file_path)
    except Exception as e:
        errors.append(f"Lecture échouée: {e}")
        logger.error("Lecture échouée pour %s: %s", filename, e)
        return TableLoaderResult(file=filename, chunks=[], n_sheets=0, n_rows=0, n_chunks=0, errors=errors)

    all_chunks: List[Document] = []
    total_rows = 0

    for sheet_name, df_raw in sheets.items():
        if df_raw.empty or len(df_raw) < 2:
            continue

        df, headers = detect_headers(df_raw)
        if df.empty:
            continue

        total_rows += len(df)

        summary = table_summary_chunk(df, headers, file_metadata, sheet_name)
        if summary:
            all_chunks.append(summary)

        row_chunks = row_level_chunks(
            df, headers, file_metadata,
            sheet_name=sheet_name, max_rows=max_rows_per_sheet,
        )
        all_chunks.extend(row_chunks)

    logger.info("Table %s: %d chunks produits (%d lignes lues)", filename, len(all_chunks), total_rows)

    return TableLoaderResult(
        file=filename, chunks=all_chunks, n_sheets=len(sheets),
        n_rows=total_rows, n_chunks=len(all_chunks), errors=errors,
    )


def load_all_tables(
    folder: str = "data/02_tables",
    max_rows_per_sheet: int = 500,
) -> List[Document]:
    """
    Charge tous les fichiers Excel/XLS/CSV d'un dossier.

    Args:
        folder: dossier contenant les fichiers tabulaires
        max_rows_per_sheet: limite de lignes par feuille

    Returns:
        Liste de tous les chunks combinés
    """
    folder_path = Path(folder)
    files: List[Path] = []
    for ext in ("*.xlsx", "*.xls", "*.csv"):
        files.extend(sorted(folder_path.glob(ext)))

    if not files:
        logger.warning("Aucun fichier tabulaire trouvé dans %s", folder)
        return []

    all_chunks: List[Document] = []

    for file_path in files:
        result = load_table(str(file_path), max_rows_per_sheet)
        if result.ok:
            all_chunks.extend(result.chunks)
        else:
            logger.warning("Échec chargement %s: %s", result.file, result.errors)

    logger.info("Ingestion tableaux terminée: %d chunks au total", len(all_chunks))
    return all_chunks