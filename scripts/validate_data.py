"""
MedAssist RAG — Validation des données téléchargées
Lance : python scripts/validate_data.py
"""

import os
import sys
from pathlib import Path

# ── Couleurs terminal Windows ─────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):  print(f"  {GREEN}✓{RESET} {msg}")
def err(msg): print(f"  {RED}✗{RESET} {msg}")
def warn(msg):print(f"  {YELLOW}⚠{RESET} {msg}")
def info(msg):print(f"  {BLUE}→{RESET} {msg}")

# ─────────────────────────────────────────────────────────────────
print(f"\n{BOLD}=== Validation données MedAssist RAG ==={RESET}\n")

total_ok = 0
total_warn = 0
total_err = 0

# ══════════════════════════════════════════════════════════════════
# 1. VÉRIFICATION PDFs texte
# ══════════════════════════════════════════════════════════════════
print(f"{BOLD}── 01_pdf_text/ ──────────────────────────────────────{RESET}")

try:
    from pypdf import PdfReader
    pdf_ok = True
except ImportError:
    err("pypdf non installé")
    pdf_ok = False

PDF_FILES = [
    "data/01_pdf_text/guide_avc_msps.pdf",
    "data/01_pdf_text/guide_sca_msps.pdf",
    "data/01_pdf_text/guide_risque_cardiovasculaire_msps.pdf",
    "data/01_pdf_text/guide_tuberculose_msps.pdf",
    "data/01_pdf_text/guide_urgences_pediatriques_msps.pdf",
    "data/01_pdf_text/guide_Filières_DE_SOINS_UNCV_VF.pdf",
]

pdf_results = []

for pdf_path in PDF_FILES:
    path = Path(pdf_path)
    fname = path.name

    if not path.exists():
        err(f"{fname} — FICHIER MANQUANT")
        total_err += 1
        pdf_results.append({"file": fname, "status": "missing"})
        continue

    size_kb = path.stat().st_size / 1024
    if size_kb < 10:
        err(f"{fname} — Trop petit ({size_kb:.0f} KB) — probablement corrompu")
        total_err += 1
        pdf_results.append({"file": fname, "status": "corrupted", "size_kb": size_kb})
        continue

    if not pdf_ok:
        warn(f"{fname} — {size_kb:.0f} KB (pypdf manquant, impossible de lire)")
        total_warn += 1
        continue

    try:
        reader = PdfReader(pdf_path)
        n_pages = len(reader.pages)

        # Lire les 2 premières pages pour vérifier le texte
        sample_text = ""
        for i in range(min(2, n_pages)):
            sample_text += reader.pages[i].extract_text() or ""

        words = len(sample_text.split())

        if words < 50:
            warn(f"{fname} — {n_pages} pages, {size_kb:.0f} KB — PDF SCANNÉ (peu de texte extractible : {words} mots)")
            info(f"→ Sera traité par OCR en Phase 4 — déplacer vers data/03_scans/ si scan pur")
            total_warn += 1
            pdf_results.append({"file": fname, "pages": n_pages, "words_sample": words, "status": "scanned"})
        else:
            ok(f"{fname} — {n_pages} pages, {size_kb:.0f} KB, ~{words} mots sur 2 pages")
            total_ok += 1
            pdf_results.append({"file": fname, "pages": n_pages, "words_sample": words, "status": "ok", "size_kb": size_kb})

        # Afficher un extrait
        if words > 20:
            preview = " ".join(sample_text.split()[:20])
            info(f'Extrait : "{preview}..."')

    except Exception as e:
        err(f"{fname} — Erreur lecture : {e}")
        total_err += 1
        pdf_results.append({"file": fname, "status": "error", "error": str(e)})

# ══════════════════════════════════════════════════════════════════
# 2. VÉRIFICATION TABLEAUX Excel/XLS
# ══════════════════════════════════════════════════════════════════
print(f"\n{BOLD}── 02_tables/ ────────────────────────────────────────{RESET}")

try:
    import pandas as pd
    xl_ok = True
except ImportError:
    err("pandas non installé")
    xl_ok = False

TABLE_FILES = [
    "data/02_tables/etablissements_sante_maroc.xlsx",
    "data/02_tables/infrastructures-privees-2024.xlsx",
    "data/02_tables/repartition-des-hopitaux-par-region-et-province-2024.xlsx",
    "data/02_tables/repartition-des-hopitaux-par-region-et-province-2020.xlsx",
    "data/02_tables/agences-cnss-2011.xls",
    "data/02_tables/offre-privees-2013.xls",
]

for tbl_path in TABLE_FILES:
    path = Path(tbl_path)
    fname = path.name

    if not path.exists():
        err(f"{fname} — FICHIER MANQUANT")
        total_err += 1
        continue

    size_kb = path.stat().st_size / 1024

    if not xl_ok:
        warn(f"{fname} — {size_kb:.0f} KB (pandas manquant)")
        total_warn += 1
        continue

    try:
        engine = "xlrd" if fname.endswith(".xls") else "openpyxl"
        try:
            df = pd.read_excel(tbl_path, engine=engine, nrows=5)
        except Exception:
            # Essai sans engine spécifié
            df = pd.read_excel(tbl_path, nrows=5)

        n_cols = len(df.columns)
        cols_preview = list(df.columns[:5])
        ok(f"{fname} — {size_kb:.0f} KB, {n_cols} colonnes")
        info(f"Colonnes : {cols_preview}")
        total_ok += 1

    except Exception as e:
        err(f"{fname} — Erreur lecture : {e}")
        info(f"→ Fichier potentiellement corrompu ou format non standard")
        total_err += 1

# ══════════════════════════════════════════════════════════════════
# 3. VÉRIFICATION Document mixte
# ══════════════════════════════════════════════════════════════════
print(f"\n{BOLD}── 05_mixed/ ─────────────────────────────────────────{RESET}")

mixed_path = Path("data/05_mixed/calendrier_vaccination_national.pdf")
if mixed_path.exists():
    size_kb = mixed_path.stat().st_size / 1024
    if pdf_ok:
        try:
            reader = PdfReader(str(mixed_path))
            n_pages = len(reader.pages)
            sample = reader.pages[0].extract_text() or ""
            words = len(sample.split())
            if words > 30:
                ok(f"calendrier_vaccination_national.pdf — {n_pages} pages, {size_kb:.0f} KB")
                info(f'Extrait p.1 : "{" ".join(sample.split()[:15])}..."')
                total_ok += 1
            else:
                warn(f"calendrier_vaccination_national.pdf — {n_pages} pages — PDF scanné ou tableaux image")
                info("→ Unstructured.io sera nécessaire pour extraire les tableaux")
                total_warn += 1
        except Exception as e:
            err(f"calendrier_vaccination_national.pdf — {e}")
            total_err += 1
    else:
        ok(f"calendrier_vaccination_national.pdf — {size_kb:.0f} KB (présent)")
        total_ok += 1
else:
    err("calendrier_vaccination_national.pdf — MANQUANT")
    total_err += 1

# ══════════════════════════════════════════════════════════════════
# 4. VÉRIFICATION Dossiers vides
# ══════════════════════════════════════════════════════════════════
print(f"\n{BOLD}── Dossiers à compléter ──────────────────────────────{RESET}")

scan_files = list(Path("data/03_scans").glob("*")) if Path("data/03_scans").exists() else []
web_files  = list(Path("data/04_web").glob("*"))   if Path("data/04_web").exists()   else []

if not scan_files:
    warn("data/03_scans/ — vide")
    info("→ Phase 4 : scanner un des PDFs téléchargés ou utiliser un document scanné")
else:
    ok(f"data/03_scans/ — {len(scan_files)} fichier(s)")

if not web_files:
    warn("data/04_web/ — vide")
    info("→ Lancer : python scripts/scrape_web_pages.py")
else:
    ok(f"data/04_web/ — {len(web_files)} fichier(s)")

# ══════════════════════════════════════════════════════════════════
# 5. GOLDEN DATASET
# ══════════════════════════════════════════════════════════════════
print(f"\n{BOLD}── Golden dataset ────────────────────────────────────{RESET}")

import json
gd_path = Path("data/golden_dataset.json")
if gd_path.exists():
    try:
        with open(gd_path, encoding="utf-8") as f:
            gd = json.load(f)
        ok(f"golden_dataset.json — {len(gd)} questions de référence")

        # Vérifier que les sources attendues existent
        for q in gd:
            src = q.get("expected_source")
            if src:
                possible_paths = [
                    f"data/01_pdf_text/{src}",
                    f"data/02_tables/{src}",
                    f"data/05_mixed/{src}",
                ]
                found = any(Path(p).exists() for p in possible_paths)
                if not found and not q.get("garde_fou"):
                    warn(f"Q{q['id']} attend '{src}' — fichier non trouvé")
        total_ok += 1
    except Exception as e:
        err(f"golden_dataset.json — {e}")
        total_err += 1
else:
    err("golden_dataset.json — MANQUANT")
    total_err += 1

# ══════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ══════════════════════════════════════════════════════════════════
print(f"\n{BOLD}{'='*52}{RESET}")
print(f"{BOLD}RÉSUMÉ VALIDATION{RESET}")
print(f"{'='*52}")
print(f"  {GREEN}✓ OK      : {total_ok}{RESET}")
print(f"  {YELLOW}⚠ Warnings : {total_warn}{RESET}")
print(f"  {RED}✗ Erreurs  : {total_err}{RESET}")
print()

if total_err == 0 and total_warn == 0:
    print(f"{GREEN}{BOLD}Tout est parfait — prêt pour la Phase 2 : ingestion !{RESET}")
elif total_err == 0:
    print(f"{YELLOW}{BOLD}Données exploitables avec quelques points à noter.{RESET}")
    print(f"{YELLOW}Les warnings n'empêchent pas de continuer.{RESET}")
else:
    print(f"{RED}{BOLD}Corriger les erreurs avant de continuer.{RESET}")

print()

# Résumé des PDFs pour le chunking
scanned = [r for r in pdf_results if r.get("status") == "scanned"]
text_ok  = [r for r in pdf_results if r.get("status") == "ok"]

if text_ok:
    print(f"{BOLD}PDFs avec texte extractible (→ chunking direct) :{RESET}")
    for r in text_ok:
        print(f"  • {r['file']} — {r['pages']} pages")

if scanned:
    print(f"\n{BOLD}PDFs scannés (→ OCR requis en Phase 4) :{RESET}")
    for r in scanned:
        print(f"  • {r['file']}")

print(f"\n{BOLD}Prochaine étape :{RESET}")
print("  → python scripts/scrape_web_pages.py   (compléter data/04_web/)")
print("  → Coder src/ingestion/pdf_loader.py    (Phase 2 ingestion)")
print()