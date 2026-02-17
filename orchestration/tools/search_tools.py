"""Semantic search tools for MCP."""

import os
import sys
from typing import Optional, List

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.vector_store import VectorStore
from services.semantic_scorer import SemanticScorer


def get_vector_store() -> VectorStore:
    """Get the vector store instance."""
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    chromadb_dir = os.path.join(project_dir, "chromadb_data")
    return VectorStore(persist_dir=chromadb_dir)


def semantic_search(
    query: str,
    item_type: Optional[str] = None,
    limit: int = 10,
    min_score: float = 0.0,
) -> dict:
    """Search content using semantic similarity.

    Args:
        query: Search query text.
        item_type: Filter by type (news, podcast, video).
        limit: Maximum results to return.
        min_score: Minimum similarity score threshold.

    Returns:
        Dict with search results.
    """
    try:
        vector_store = get_vector_store()
        results = vector_store.search(
            query_text=query,
            item_type=item_type,
            limit=limit,
        )

        # Filter by minimum score
        filtered = [r for r in results if r["similarity"] >= min_score]

        return {
            "success": True,
            "query": query,
            "total_results": len(filtered),
            "results": [
                {
                    "id": r["id"],
                    "title": r.get("metadata", {}).get("title", "Untitled"),
                    "item_type": r.get("metadata", {}).get("item_type", "unknown"),
                    "link": r.get("metadata", {}).get("link", ""),
                    "source": r.get("metadata", {}).get("source", ""),
                    "similarity": round(r["similarity"], 4),
                    "semantic_score": r.get("metadata", {}).get("semantic_score"),
                    "text_preview": r.get("text", "")[:200],
                }
                for r in filtered
            ],
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def find_similar_items(
    item_id: str,
    item_type: str,
    limit: int = 5,
    threshold: float = 0.5,
) -> dict:
    """Find items similar to a given item.

    Args:
        item_id: ID of the source item.
        item_type: Type of the source item.
        limit: Maximum results to return.
        threshold: Minimum similarity threshold.

    Returns:
        Dict with similar items.
    """
    try:
        vector_store = get_vector_store()

        # Get the source item
        item = vector_store.get_item(item_id, item_type)
        if not item:
            return {
                "success": False,
                "error": f"Item {item_id} not found in {item_type} collection",
            }

        embedding = item.get("embedding")
        if not embedding:
            return {
                "success": False,
                "error": f"Item {item_id} has no embedding",
            }

        # Find similar items
        results = vector_store.find_similar(
            embedding=embedding,
            item_type=item_type,
            threshold=threshold,
            exclude_ids=[item_id],
        )[:limit]

        source_title = item.get("metadata", {}).get("title", item_id)

        return {
            "success": True,
            "source_item": {
                "id": item_id,
                "title": source_title,
                "type": item_type,
            },
            "similar_count": len(results),
            "similar_items": [
                {
                    "id": r["id"],
                    "title": r.get("metadata", {}).get("title", "Untitled"),
                    "similarity": round(r["similarity"], 4),
                    "link": r.get("metadata", {}).get("link", ""),
                }
                for r in results
            ],
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def get_vector_stats() -> dict:
    """Get statistics about the vector store.

    Returns:
        Dict with collection counts and totals.
    """
    try:
        vector_store = get_vector_store()

        stats = {}
        for item_type in ["news", "podcast", "video"]:
            try:
                stats[item_type] = vector_store.get_collection_count(item_type)
            except Exception:
                stats[item_type] = 0

        return {
            "success": True,
            "collections": stats,
            "total_embeddings": sum(stats.values()),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def score_text(text: str) -> dict:
    """Score text against user interests using semantic similarity.

    Args:
        text: Text to score.

    Returns:
        Dict with semantic score and relevance assessment.
    """
    try:
        scorer = SemanticScorer()
        score = scorer.score_text(text)
        is_relevant = scorer.is_relevant(score)

        return {
            "success": True,
            "text_preview": text[:200],
            "semantic_score": round(score, 4),
            "is_relevant": is_relevant,
            "relevance_level": (
                "high" if score >= 0.5 else
                "medium" if score >= 0.3 else
                "low"
            ),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
