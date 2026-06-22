"""
MedAssist RAG — app.py
Interface Streamlit de démonstration pour MedAssist.

Lancer : streamlit run app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, ".")
from src.rag.chain import MedAssistChain


# ─────────────────────────────────────────────────────────────────
# Configuration de la page
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedAssist: Assistant Médical MSPS",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CSS personnalisé
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1B3A6B 0%, #2E75B6 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }
    .source-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2E75B6;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
    }
    .source-title {
        font-weight: 600;
        color: #1B3A6B;
        font-size: 0.9rem;
    }
    .source-meta {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 0.2rem;
    }
    .badge {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 500;
        margin-right: 0.3rem;
    }
    .badge-pdf { background: #DBEAFE; color: #1E40AF; }
    .badge-table_row { background: #FEF3C7; color: #92400E; }
    .badge-table_summary { background: #FDE68A; color: #78350F; }
    .badge-ocr_text { background: #FCE7E7; color: #991B1B; }
    .badge-web_content { background: #D1FAE5; color: #065F46; }
    .warning-box {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        border-radius: 8px;
        padding: 1rem;
        color: #991B1B;
        margin: 1rem 0;
    }
    .disclaimer {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #92400E;
        margin-top: 1rem;
    }
    .stat-box {
        background: #F8FAFC;
        border-radius: 8px;
        padding: 0.6rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🩺 MedAssist: Assistant Médical MSPS</h1>
    <p>Consultation rapide des protocoles officiels du Ministère de la Santé et de la Protection Sociale 
    <img src="https://flagcdn.com/24x18/ma.png" alt="Maroc" style="vertical-align: middle;">
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Initialisation du RAG chain (mise en cache)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_chain():
    return MedAssistChain(k=5, use_filter=True)


with st.spinner("Initialisation de MedAssist — chargement de l'index médical..."):
    try:
        chain = load_chain()
        chain_ready = True
        chain_error = None
    except Exception as e:
        chain_ready = False
        chain_error = str(e)


if not chain_ready:
    st.error(f"Erreur d'initialisation : {chain_error}")
    st.info("Vérifiez votre fichier .env (OPENAI_API_KEY) et que chroma_db/ existe.")
    st.stop()


# ─────────────────────────────────────────────────────────────────
# Sidebar — informations et exemples
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 État du système")

    n_docs = chain.vs.count()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Spécialités", "6")

    st.markdown("---")
    st.markdown("### 📚 Sources couvertes")
    st.markdown("""
    - Guide Tuberculose (PNLAT)
    - Guide Risque Cardiovasculaire
    - Guide AVC / SCA
    - Guide Urgences Pédiatriques
    - Calendrier Vaccination
    - Établissements de santé (Open Data)
    - Bulletin épidémiologique MSPS
    """)

    st.markdown("---")
    st.markdown("### 💡 Questions d'exemple")

    example_questions = [
        "Quels sont les critères de diagnostic de la tuberculose ?",
        "Quelles sont les cibles tensionnelles pour un patient hypertendu ?",
        "Combien d'hôpitaux y a-t-il dans la région de Casablanca-Settat ?",
        "Quelle est la définition d'un cas suspect de tuberculose ?",
        "Quel médicament dois-je prescrire ?",
    ]

    for q in example_questions:
        if st.button(q, key=f"ex_{hash(q)}", use_container_width=True):
            st.session_state["pending_question"] = q

    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <b>Avertissement médical</b><br>
        MedAssist est un outil de consultation documentaire.
        Il ne pose aucun diagnostic et ne prescrit aucun traitement.
        Toute décision clinique doit être validée par un professionnel
        de santé qualifié.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Historique de conversation
# ─────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ─────────────────────────────────────────────────────────────────
# Zone de saisie
# ─────────────────────────────────────────────────────────────────
col_input, col_button = st.columns([5, 1])

with col_input:
    default_value = st.session_state.pending_question or ""
    question = st.text_input(
        "Posez votre question médicale",
        value=default_value,
        placeholder="Ex : Quels sont les critères de prise en charge de l'HTA chez un patient diabétique ?",
        label_visibility="collapsed",
        key="question_input",
    )

with col_button:
    submit = st.button("Rechercher", type="primary", use_container_width=True)

st.session_state.pending_question = None


# ─────────────────────────────────────────────────────────────────
# Traitement de la question
# ─────────────────────────────────────────────────────────────────
if submit and question.strip():
    with st.spinner("Recherche dans les protocoles MSPS..."):
        try:
            response = chain.invoke(question)
            st.session_state.history.insert(0, response)
        except Exception as e:
            st.error(f"Erreur lors du traitement : {e}")


# ─────────────────────────────────────────────────────────────────
# Affichage des résultats
# ─────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")

    for idx, resp in enumerate(st.session_state.history):

        with st.container():
            # Question
            st.markdown(f"#### 🔍 {resp.question}")

            # Badges méta
            badges_html = ""
            if resp.speciality_detected:
                badges_html += f'<span class="badge badge-pdf">{resp.speciality_detected}</span>'
            badges_html += f'<span class="badge badge-table_row">{resp.latency_ms:.0f}ms</span>'
            badges_html += f'<span class="badge badge-web_content">{resp.n_chunks_retrieved} sources analysées</span>'
            st.markdown(badges_html, unsafe_allow_html=True)

            st.markdown("")

            # Réponse
            if resp.is_hors_scope:
                st.markdown(f"""
                <div class="warning-box">
                    {resp.answer}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(resp.answer)

            # Sources
            if resp.sources:
                with st.expander(f"📄 Voir les {len(resp.sources)} sources citées"):
                    for i, src in enumerate(resp.sources, 1):
                        etype = src.get("element_type", "pdf")
                        badge_class = f"badge-{etype}" if etype in [
                            "pdf", "table_row", "table_summary", "ocr_text", "web_content"
                        ] else "badge-pdf"

                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-title">[{i}] {src.get('title', 'N/A')}</div>
                            <div class="source-meta">
                                <span class="badge {badge_class}">{etype}</span>
                                Page {src.get('page', '?')} •
                                Score: {src.get('score', 0):.3f} •
                                Année: {src.get('year', 'N/A')}
                            </div>
                            <div style="margin-top:0.5rem; font-size:0.85rem; color:#475569;">
                                {src.get('excerpt', '')}...
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            st.markdown("---")

    # Bouton pour vider l'historique
    if st.button("🗑️ Effacer l'historique"):
        st.session_state.history = []
        st.rerun()

else:
    # État vide — message d'accueil
    st.info(
        "👋 Bienvenue sur MedAssist. Posez une question médicale ci-dessus ou "
        "choisissez un exemple dans le menu latéral pour commencer."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **🦠 Infectiologie**
        Protocoles tuberculose, dépistage, traitement
        """)
    with col2:
        st.markdown("""
        **❤️ Cardiologie**
        Risque cardiovasculaire, HTA, SCA, AVC
        """)
    with col3:
        st.markdown("""
        **🏥 Organisation sanitaire**
        Établissements, hôpitaux, infrastructures par région
        """)


# ─────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Données : Ministère de la Santé et de la Protection Sociale (MSPS) Maroc | "
)