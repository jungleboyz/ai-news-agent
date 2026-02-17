"""Background tasks for embedding generation."""

import os
import sys
from typing import Optional

from celery import shared_task
from celery.utils.log import get_task_logger

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.embeddings import EmbeddingService
from services.vector_store import VectorStore
from services.semantic_scorer import SemanticScorer

logger = get_task_logger(__name__)

# Singleton instances (created once per worker)
_embedding_service: Optional[EmbeddingService] = None
_vector_store: Optional[VectorStore] = None
_semantic_scorer: Optional[SemanticScorer] = None


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
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        persist_dir = os.path.join(project_dir, "chromadb_data")
        _vector_store = VectorStore(persist_dir=persist_dir, embedding_service=get_embedding_service())
    return _vector_store


def get_semantic_scorer() -> SemanticScorer:
    """Get or create the semantic scorer singleton."""
    global _semantic_scorer
    if _semantic_scorer is None:
        _semantic_scorer = SemanticScorer(embedding_service=get_embedding_service())
    return _semantic_scorer


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def embed_item(
    self,
    item_id: str,
    text: str,
    item_type: str,
    metadata: Optional[dict] = None,
) -> dict:
    """Generate embedding for a single item and store it.

    Args:
        item_id: Unique identifier for the item.
        text: Text content to embed.
        item_type: One of "news", "podcast", "video".
        metadata: Optional metadata to store.

    Returns:
        Dict with item_id and semantic_score.
    """
    try:
        embedding_service = get_embedding_service()
        vector_store = get_vector_store()
        scorer = get_semantic_scorer()

        # Generate embedding
        embedding = embedding_service.get_embedding(text)

        # Calculate semantic score
        semantic_score = scorer.score_item(embedding)

        # Store in vector store
        meta = metadata or {}
        meta["semantic_score"] = semantic_score
        vector_store.add_item(item_id, text, item_type, meta, embedding)

        logger.info(f"Embedded item {item_id} with score {semantic_score:.3f}")

        return {
            "item_id": item_id,
            "semantic_score": semantic_score,
            "success": True,
        }

    except Exception as exc:
        logger.error(f"Failed to embed item {item_id}: {exc}")
        raise self.retry(exc=exc)


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def embed_new_items(
    self,
    items: list[dict],
    item_type: str,
) -> dict:
    """Batch process items for embedding generation.

    Args:
        items: List of dicts with keys: id, text, metadata (optional).
        item_type: One of "news", "podcast", "video".

    Returns:
        Dict with processed count and scores.
    """
    try:
        if not items:
            return {"processed": 0, "scores": {}}

        embedding_service = get_embedding_service()
        vector_store = get_vector_store()
        scorer = get_semantic_scorer()

        # Extract texts for batch embedding
        texts = [item["text"] for item in items]
        ids = [item["id"] for item in items]

        # Batch generate embeddings (up to 2048 at a time)
        batch_size = 2048
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = embedding_service.batch_embed(batch_texts)
            all_embeddings.extend(batch_embeddings)

        # Score all items
        scores = scorer.score_items_batch(all_embeddings)

        # Prepare items for batch storage
        items_to_store = []
        scores_map = {}

        for i, (item, embedding, score) in enumerate(zip(items, all_embeddings, scores)):
            if embedding:  # Only store if embedding was generated
                meta = item.get("metadata", {})
                meta["semantic_score"] = score
                items_to_store.append({
                    "id": item["id"],
                    "text": item["text"],
                    "metadata": meta,
                    "embedding": embedding,
                })
                scores_map[item["id"]] = score

        # Store in vector store
        stored_ids = vector_store.add_items_batch(items_to_store, item_type)

        logger.info(f"Embedded {len(stored_ids)} {item_type} items")

        return {
            "processed": len(stored_ids),
            "scores": scores_map,
            "success": True,
        }

    except Exception as exc:
        logger.error(f"Failed to batch embed items: {exc}")
        raise self.retry(exc=exc)


@shared_task
def check_duplicates(
    text: str,
    item_type: str,
    threshold: float = 0.95,
) -> dict:
    """Check if similar content already exists.

    Args:
        text: Text content to check.
        item_type: Collection to search in.
        threshold: Minimum similarity to consider duplicate.

    Returns:
        Dict with is_duplicate flag and similar items.
    """
    try:
        embedding_service = get_embedding_service()
        vector_store = get_vector_store()

        embedding = embedding_service.get_embedding(text)
        similar = vector_store.find_similar(embedding, item_type, threshold)

        return {
            "is_duplicate": len(similar) > 0,
            "similar_items": similar,
            "success": True,
        }

    except Exception as exc:
        logger.error(f"Failed to check duplicates: {exc}")
        return {
            "is_duplicate": False,
            "similar_items": [],
            "success": False,
            "error": str(exc),
        }
