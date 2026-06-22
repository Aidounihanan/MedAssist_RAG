"""
MedAssist RAG — src/rag/chain.py
La RAG chain : reçoit une question médicale, cherche dans le vector store,
génère une réponse avec sources citées et garde-fous.

Module de production — aucune logique de test ici.
Tests : tests/test_chain.py
Démo interactive : app.py (Streamlit)
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

from src.embeddings.vector_store import get_embeddings, get_vector_store

load_dotenv()
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Tu es MedAssist, un assistant médical spécialisé pour les professionnels
de santé marocains (médecins, infirmiers, pharmaciens).

Tu réponds aux questions médicales en te basant UNIQUEMENT sur les documents
officiels du Ministère de la Santé et de la Protection Sociale (MSPS) du Maroc
fournis dans le contexte.

RÈGLES IMPÉRATIVES :
1. Base-toi UNIQUEMENT sur les documents fournis — jamais sur tes connaissances générales
2. Cite toujours tes sources avec le format [Source: nom_document, Page X]
3. Si l'information n'est PAS dans le contexte, dis clairement :
   "Cette information n'est pas disponible dans mes documents actuels."
4. Ne fais JAMAIS de diagnostic médical
5. Ne prescris JAMAIS de médicaments
6. Recommande toujours de consulter un médecin pour les décisions cliniques
7. Pour les urgences vitales, rappelle d'appeler le 15 (SAMU Maroc)
8. Réponds en français sauf si la question est en arabe

FORMAT DE RÉPONSE :
- Réponse directe et structurée en points si nécessaire
- Sources citées à la fin : [Source: ..., Page X]
- Mise en garde médicale si le sujet le requiert
"""

SPECIALITY_KEYWORDS: Dict[str, List[str]] = {
    "infectiologie": [
        "tuberculose", "tb", "bacille", "koch", "pnlat", "tep",
        "infection", "antibiotique", "bactérie", "viral",
    ],
    "cardiologie": [
        "cardiaque", "cœur", "infarctus", "sca", "avc", "coronarien",
        "hypertension", "hta", "tensionnel", "cardiovasculaire",
        "artérielle", "cardio", "pression artérielle",
    ],
    "neurologie": [
        "avc", "accident vasculaire", "cérébral", "neurologique",
        "neurovascular", "cerveau", "ischémique", "hémorragique",
    ],
    "pediatrie": [
        "enfant", "nourrisson", "pédiatrique", "vaccination", "vaccin",
        "calendrier vaccinal", "bébé", "nouveau-né", "pédiatrie",
    ],
    "epidemiologie": [
        "épidémiologie", "prévalence", "incidence", "statistique",
        "maroc", "population", "enquête", "nationale", "mortalité",
    ],
    "organisation_sanitaire": [
        "hôpital", "établissement", "centre de santé", "région",
        "province", "délégation", "infrastructure", "cnss",
    ],
}

HORS_SCOPE_PATTERNS = [
    "prescris", "prescrit", "ordonnance", "quel médicament",
    "dosage exact", "dose exacte", "diagnostic de",
    "est-ce que j'ai", "ai-je", "suis-je malade",
]


def detect_speciality(question: str) -> Optional[str]:
    """Détecte la spécialité médicale depuis la question, ou None si générale."""
    question_lower = question.lower()
    scores: Dict[str, int] = {}

    for speciality, keywords in SPECIALITY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in question_lower)
        if score > 0:
            scores[speciality] = score

    return max(scores, key=lambda k: scores[k]) if scores else None


def is_hors_scope(question: str) -> bool:
    """Détecte si la question demande un diagnostic ou une prescription."""
    question_lower = question.lower()
    return any(pattern in question_lower for pattern in HORS_SCOPE_PATTERNS)


@dataclass
class RAGResponse:
    """Réponse structurée de la RAG chain, avec sources et métadonnées."""
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    speciality_detected: Optional[str]
    latency_ms: float
    n_chunks_retrieved: int
    is_hors_scope: bool = False
    error: Optional[str] = None


def get_llm(temperature: float = 0.1):
    """Retourne le LLM configuré (GPT-4o-mini via OpenAI)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY manquante dans .env")

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        openai_api_key=api_key,
        max_tokens=1024,
    )


def build_context(docs_with_scores: List[Tuple[Document, float]]) -> str:
    """Construit le contexte textuel numéroté à injecter dans le prompt."""
    context_parts = []

    for i, (doc, score) in enumerate(docs_with_scores, 1):
        title = doc.metadata.get("title", "Document MSPS")
        page = doc.metadata.get("page", "?")
        year = doc.metadata.get("year", "")
        etype = doc.metadata.get("element_type", "texte")

        header = f"[Document {i}] {title}"
        if year:
            header += f" ({year})"
        header += f" — Page {page}"
        if etype == "table_row":
            header += " [Tableau]"
        elif etype == "ocr_text":
            header += " [Document scanné]"
        elif etype == "web_content":
            header += " [Web MSPS]"

        context_parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(context_parts)


class MedAssistChain:
    """
    RAG chain complète de MedAssist.

    Usage:
        chain = MedAssistChain()
        response = chain.invoke("Quels sont les critères de diagnostic de la tuberculose ?")
    """

    def __init__(self, k: int = 5, use_filter: bool = True):
        """
        Args:
            k: nombre de chunks à récupérer
            use_filter: filtrer par spécialité détectée
        """
        self.embeddings = get_embeddings()
        self.vs = get_vector_store(self.embeddings)
        self.llm = get_llm()
        self.k = k
        self.use_filter = use_filter
        logger.info("MedAssistChain initialisée: %d chunks indexés", self.vs.count())

    def _hors_scope_response(self, question: str, t0: float) -> RAGResponse:
        return RAGResponse(
            question=question,
            answer=(
                "Je ne peux pas répondre à cette demande.\n\n"
                "MedAssist est un outil d'aide à la consultation des protocoles "
                "MSPS — il ne pose pas de diagnostic et ne prescrit pas de "
                "médicaments.\n\n"
                "Pour toute décision clinique, veuillez consulter un médecin "
                "ou contacter le SAMU au 15 en cas d'urgence."
            ),
            sources=[],
            speciality_detected=detect_speciality(question),
            latency_ms=(time.time() - t0) * 1000,
            n_chunks_retrieved=0,
            is_hors_scope=True,
        )

    def _no_results_response(self, question: str, speciality: Optional[str], t0: float) -> RAGResponse:
        return RAGResponse(
            question=question,
            answer=(
                "Cette information n'est pas disponible dans mes documents actuels.\n\n"
                "Je vous recommande de consulter directement le site du MSPS : "
                "www.sante.gov.ma ou de contacter votre Direction Régionale de Santé."
            ),
            sources=[],
            speciality_detected=speciality,
            latency_ms=(time.time() - t0) * 1000,
            n_chunks_retrieved=0,
        )

    def invoke(self, question: str) -> RAGResponse:
        """
        Traite une question médicale et retourne une réponse structurée.

        Pipeline : détection spécialité/garde-fou -> retrieval filtré
        -> construction prompt -> appel LLM -> réponse avec sources.

        Args:
            question: question médicale posée par le professionnel de santé

        Returns:
            RAGResponse avec réponse, sources, latence et métadonnées
        """
        t0 = time.time()

        if is_hors_scope(question):
            return self._hors_scope_response(question, t0)

        speciality = detect_speciality(question)
        filter_dict = {"speciality": speciality} if (self.use_filter and speciality) else None

        docs_with_scores = self.vs.similarity_search_with_score(
            query=question, k=self.k, filter_dict=filter_dict,
        )

        if len(docs_with_scores) < 2 and filter_dict:
            docs_with_scores = self.vs.similarity_search_with_score(
                query=question, k=self.k, filter_dict=None,
            )

        if not docs_with_scores:
            return self._no_results_response(question, speciality, t0)

        context = build_context(docs_with_scores)
        user_prompt = f"""DOCUMENTS DE RÉFÉRENCE MSPS :

{context}

---

QUESTION DU PROFESSIONNEL DE SANTÉ :
{question}

Réponds en te basant uniquement sur les documents fournis ci-dessus.
Cite les sources avec le format [Document X]."""

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            llm_response = self.llm.invoke(messages)
            answer = llm_response.content
        except Exception as e:
            logger.error("Erreur génération LLM: %s", e)
            answer = "Une erreur est survenue lors de la génération de la réponse. Merci de réessayer."

        sources = [
            {
                "title":        doc.metadata.get("title", ""),
                "page":         doc.metadata.get("page", "?"),
                "source":       doc.metadata.get("source", ""),
                "speciality":   doc.metadata.get("speciality", ""),
                "element_type": doc.metadata.get("element_type", "pdf"),
                "year":         doc.metadata.get("year", ""),
                "score":        score,
                "excerpt":      doc.page_content[:150],
            }
            for doc, score in docs_with_scores
        ]

        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            speciality_detected=speciality,
            latency_ms=(time.time() - t0) * 1000,
            n_chunks_retrieved=len(docs_with_scores),
        )