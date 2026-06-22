"""
MedAssist RAG — src/embeddings/vector_store.py
Couche d'abstraction vector store : ChromaDB (dev) | Weaviate (prod) | Qdrant (prod)
Contrôlé par VECTOR_STORE_BACKEND dans .env

Module de production — aucune logique de test ici.
Tests : tests/test_vector_store.py
Orchestration CLI : scripts/index_all.py
"""

import os
import logging
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()
logger = logging.getLogger(__name__)


def get_embeddings():
    """
    Retourne le modèle d'embedding selon USE_LOCAL_EMBEDDINGS dans .env.

    true  -> sentence-transformers/all-MiniLM-L6-v2 (local, gratuit, 384 dims)
    false -> OpenAI text-embedding-3-small (API, 1536 dims)
    """
    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"

    if use_local:
        from langchain_huggingface import HuggingFaceEmbeddings
        logger.info("Embeddings: all-MiniLM-L6-v2 (local, 384 dims)")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    from langchain_openai import OpenAIEmbeddings
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY manquante dans .env")
    logger.info("Embeddings: text-embedding-3-small (OpenAI, 1536 dims)")
    return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)


class ChromaVectorStore:
    """Vector store ChromaDB — développement local, persistance sur disque."""

    COLLECTION_NAME = "medassist_docs"

    def __init__(self, embeddings, persist_dir: str = "chroma_db"):
        from langchain_chroma import Chroma
        self.persist_dir = persist_dir
        self.embeddings = embeddings
        self._store = Chroma(
            collection_name=self.COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )

    def add_documents(self, documents: List[Document], batch_size: int = 100) -> int:
        """Ajoute des documents par batch. Retourne le nombre indexé."""
        total = len(documents)
        for i in range(0, total, batch_size):
            self._store.add_documents(documents[i:i + batch_size])
        return total

    def similarity_search(
        self, query: str, k: int = 5, filter_dict: Optional[dict] = None
    ) -> List[Document]:
        return self._store.similarity_search(query=query, k=k, filter=filter_dict)

    def similarity_search_with_score(
        self, query: str, k: int = 5, filter_dict: Optional[dict] = None
    ) -> List[tuple]:
        return self._store.similarity_search_with_score(query=query, k=k, filter=filter_dict)

    def count(self) -> int:
        return self._store._collection.count()

    def reset_collection(self):
        """Vide la collection avant une ré-indexation complète."""
        self._store.delete_collection()
        from langchain_chroma import Chroma
        self._store = Chroma(
            collection_name=self.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )

    def as_retriever(self, k: int = 5, filter_dict: Optional[dict] = None):
        search_kwargs = {"k": k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict
        return self._store.as_retriever(search_kwargs=search_kwargs)


class WeaviateVectorStore:
    """Vector store Weaviate — production. Nécessite weaviate-client installé."""

    def __init__(self, embeddings):
        try:
            import weaviate
            from langchain_weaviate import WeaviateVectorStore as LCWeaviate
        except ImportError as e:
            raise ImportError(
                "weaviate-client non installé. "
                "Décommenter dans requirements.txt et réinstaller."
            ) from e

        url = os.getenv("WEAVIATE_URL")
        api_key = os.getenv("WEAVIATE_API_KEY")
        if not url or not api_key:
            raise ValueError("WEAVIATE_URL et WEAVIATE_API_KEY requis dans .env")

        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url, auth_credentials=weaviate.AuthApiKey(api_key)
        )
        self._store = LCWeaviate(
            client=self.client,
            index_name="MedAssistDoc",
            text_key="content",
            embedding=embeddings,
            attributes=["source", "speciality", "doc_type", "year", "page"],
        )

    def add_documents(self, documents: List[Document], batch_size: int = 100) -> int:
        total = len(documents)
        for i in range(0, total, batch_size):
            self._store.add_documents(documents[i:i + batch_size])
        return total

    def similarity_search(self, query, k=5, filter_dict=None):
        return self._store.similarity_search(query, k=k, filters=filter_dict)

    def similarity_search_with_score(self, query, k=5, filter_dict=None):
        return self._store.similarity_search_with_score(query, k=k)

    def as_retriever(self, k=5, filter_dict=None):
        return self._store.as_retriever(search_kwargs={"k": k})


class QdrantVectorStore:
    """Vector store Qdrant — production. Nécessite qdrant-client installé."""

    def __init__(self, embeddings):
        try:
            from langchain_qdrant import QdrantVectorStore as LCQdrant
            from qdrant_client import QdrantClient
        except ImportError as e:
            raise ImportError(
                "qdrant-client non installé. "
                "Décommenter dans requirements.txt et réinstaller."
            ) from e

        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        self.client = QdrantClient(url=url, api_key=api_key)
        self._store = LCQdrant(
            client=self.client, collection_name="medassist_docs", embeddings=embeddings,
        )

    def add_documents(self, documents, batch_size=100):
        total = len(documents)
        for i in range(0, total, batch_size):
            self._store.add_documents(documents[i:i + batch_size])
        return total

    def similarity_search(self, query, k=5, filter_dict=None):
        return self._store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query, k=5, filter_dict=None):
        return self._store.similarity_search_with_score(query, k=k)

    def as_retriever(self, k=5, filter_dict=None):
        return self._store.as_retriever(search_kwargs={"k": k})


def get_vector_store(embeddings=None):
    """
    Retourne le vector store selon VECTOR_STORE_BACKEND dans .env.

    C'est le seul point d'entrée du module — tout le reste du code
    appelle uniquement cette fonction, jamais les classes directement.

    Returns:
        Instance de ChromaVectorStore, WeaviateVectorStore ou QdrantVectorStore
    """
    if embeddings is None:
        embeddings = get_embeddings()

    backend = os.getenv("VECTOR_STORE_BACKEND", "chroma").lower()
    logger.info("Vector store backend: %s", backend)

    if backend == "chroma":
        return ChromaVectorStore(embeddings)
    if backend == "weaviate":
        return WeaviateVectorStore(embeddings)
    if backend == "qdrant":
        return QdrantVectorStore(embeddings)

    raise ValueError(
        f"Backend inconnu: '{backend}'. Valeurs acceptées: chroma | weaviate | qdrant"
    )