"""
MedAssist RAG — Scraper pages web officielles MSPS
Lance : python scripts/scrape_web_pages.py
"""

import json
from datetime import datetime
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

Path("data/04_web").mkdir(parents=True, exist_ok=True)

# ── Pages web officielles MSPS à scraper ─────────────────────────
WEB_PAGES = [
    {
        "url": "https://www.sante.gov.ma/Publications/Pages/Bullten_%C3%89pid%C3%A9miologique.aspx",
        "filename": "data/04_web/bulletin_epidemiologique_msps.json",
        "description": "Bulletin épidémiologique MSPS",
        "speciality": "epidemiologie",
    },
    {
        "url": "https://www.sante.gov.ma/Maladies/Pages/Maladies-Cardiovasculaires.aspx",
        "filename": "data/04_web/maladies_cardiovasculaires_msps.json",
        "description": "Maladies cardiovasculaires — MSPS",
        "speciality": "cardiologie",
    },
    {
        "url": "https://www.sante.gov.ma/Maladies/Pages/Tuberculose.aspx",
        "filename": "data/04_web/tuberculose_msps.json",
        "description": "Tuberculose — MSPS",
        "speciality": "infectiologie",
    },
    {
        "url": "https://www.sante.gov.ma/Maladies/Pages/Diabete.aspx",
        "filename": "data/04_web/diabete_msps.json",
        "description": "Diabète — MSPS",
        "speciality": "endocrinologie",
    },
    {
        "url": "https://www.sante.gov.ma/Maladies/Pages/Hypertension-Art%C3%A9rielle.aspx",
        "filename": "data/04_web/hypertension_msps.json",
        "description": "Hypertension artérielle — MSPS",
        "speciality": "cardiologie",
    },
    {
        "url": "https://www.sante.gov.ma/Maladies/Pages/Cancer.aspx",
        "filename": "data/04_web/cancer_msps.json",
        "description": "Cancer — MSPS",
        "speciality": "oncologie",
    },
]


# ── Fix SSL Windows ───────────────────────────────────────────────
def get_client():
    return httpx.Client(timeout=30, follow_redirects=True, verify=False), "no_ssl"


def clean_text(soup: BeautifulSoup) -> str:
    """Extrait et nettoie le texte d'une page HTML."""
    # Supprimer les éléments parasites
    for tag in soup(["nav", "footer", "script", "style", "header", "aside", "iframe", "noscript"]):
        tag.decompose()

    # Supprimer les menus de navigation répétitifs
    for tag in soup.find_all(class_=["menu", "navigation", "sidebar", "breadcrumb", "ms-nav"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    # Nettoyer lignes vides multiples
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    lines = [line for line in lines if len(line) > 15]

    # Dédupliquer les lignes consécutives identiques (menus répétés)
    deduped = []
    for line in lines:
        if not deduped or line != deduped[-1]:
            deduped.append(line)

    return "\n".join(deduped)


def scrape_page(url: str, filename: str, description: str, speciality: str):
    """Scrape une page et sauvegarde en JSON avec métadonnées."""

    if Path(filename).exists():
        print(f"  Déjà présent : {filename}")
        return True

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "fr-FR,fr;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
    }

    client, ssl_mode = get_client()

    try:
        with client:
            response = client.get(url, headers=headers)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        clean = clean_text(soup)
        word_count = len(clean.split())

        if word_count < 100:
            print(f"  ⚠  {description} — contenu trop court ({word_count} mots)")
            print("     → Page peut nécessiter JavaScript (SPA/SharePoint)")
            # On sauvegarde quand même pour traçabilité

        title = soup.title.string.strip() if soup.title else description

        doc = {
            "source_url": url,
            "title": title,
            "description": description,
            "speciality": speciality,
            "scraped_at": datetime.now().isoformat(),
            "ssl_mode": ssl_mode,
            "content": clean,
            "word_count": word_count,
            "doc_type": "web_page",
            "source": "sante.gov.ma",
            "language": "fr",
            "country": "MA",
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

        status = "OK" if word_count >= 100 else "COURT"
        print(f"  ✓  [{status}] {description}")
        print(f"     {word_count} mots → {filename}")
        return True

    except httpx.HTTPStatusError as e:
        print(f"  ✗  {description} — HTTP {e.response.status_code}")
        print(f"     URL: {url}")
        return False
    except Exception as e:
        print(f"  ✗  {description} — {type(e).__name__}: {e}")
        return False


# ── Lancement ─────────────────────────────────────────────────────
print("=== Scraping pages web MSPS ===\n")

ok_count = 0
fail_count = 0
short_count = 0

for page in WEB_PAGES:
    result = scrape_page(**page)
    if result:
        ok_count += 1
    else:
        fail_count += 1

# ── Résumé ────────────────────────────────────────────────────────
print(f"\n=== Résumé : {ok_count} OK / {fail_count} échecs ===")

if fail_count > 0:
    print("""
Note : sante.gov.ma utilise SharePoint — certaines pages
nécessitent un navigateur pour le rendu JavaScript.

Alternative : télécharger les pages manuellement
  1. Ouvrir la page dans Chrome/Firefox
  2. Ctrl+S → "Page Web, HTML seulement"
  3. Placer dans data/04_web/ avec extension .html
  4. Le loader html sera codé en Phase 5 pour les lire
""")

web_files = list(Path("data/04_web").glob("*.json"))
print(f"Fichiers dans data/04_web/ : {len(web_files)}")
for f in web_files:
    size = f.stat().st_size / 1024
    print(f"  • {f.name} ({size:.0f} KB)")

print("\nProchaine étape : python scripts/validate_data.py")
