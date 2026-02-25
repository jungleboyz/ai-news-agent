"""Centralized singleton registry for embedding, vector store, and semantic scorer.

All agents import from here instead of defining their own instances.
"""
from typing import Optional

from services.embeddings import EmbeddingService
from services.vector_store import VectorStore
from services.semantic_scorer import SemanticScorer

CHROMADB_DIR = "chromadb_data"

# Service singletons
_embedding_service: Optional[EmbeddingService] = None
_vector_store: Optional[VectorStore] = None
_semantic_scorer: Optional[SemanticScorer] = None

# Quota exceeded flag — prevents repeated API calls after 429 error.
# Shared across all agents in the same process.
_quota_exceeded: bool = False


def get_embedding_service() -> EmbeddingService:
    """Get or create the embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def get_vector_store() -> VectorStore:
    """Get or create the vector store singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(
            persist_dir=CHROMADB_DIR,
            embedding_service=get_embedding_service(),
        )
    return _vector_store


def get_semantic_scorer() -> SemanticScorer:
    """Get or create the semantic scorer singleton."""
    global _semantic_scorer
    if _semantic_scorer is None:
        _semantic_scorer = SemanticScorer(embedding_service=get_embedding_service())
    return _semantic_scorer


def is_quota_exceeded() -> bool:
    return _quota_exceeded


def set_quota_exceeded(value: bool = True) -> None:
    global _quota_exceeded
    _quota_exceeded = value
