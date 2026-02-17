"""Semantic search API endpoints."""

import os
import sys
from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

# Add parent directories to path for imports
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_dir)

from services.embeddings import EmbeddingService
from services.vector_store import VectorStore
from services.semantic_scorer import SemanticScorer

router = APIRouter(prefix="/api", tags=["semantic-search"])

# Singleton instances
_embedding_service: Optional[EmbeddingService] = None
_vector_store: Optional[VectorStore] = None
_scorer: Optional[SemanticScorer] = None


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
        persist_dir = os.path.join(project_dir, "chromadb_data")
        _vector_store = VectorStore(persist_dir=persist_dir, embedding_service=get_embedding_service())
    return _vector_store


def get_scorer() -> SemanticScorer:
    """Get or create the semantic scorer singleton."""
    global _scorer
    if _scorer is None:
        _scorer = SemanticScorer(embedding_service=get_embedding_service())
    return _scorer


class SearchResult(BaseModel):
    """A single search result."""
    id: str
    text: str
    title: Optional[str] = None
    link: Optional[str] = None
    source: Optional[str] = None
    item_type: str
    similarity: float
    semantic_score: Optional[float] = None


class SearchResponse(BaseModel):
    """Response from semantic search."""
    query: str
    results: list[SearchResult]
    total: int


@router.get("/semantic-search", response_model=SearchResponse)
async def semantic_search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Maximum results to return"),
    item_type: Optional[str] = Query(None, description="Filter by type: news, podcast, video"),
    min_score: float = Query(0.0, ge=0.0, le=1.0, description="Minimum similarity score"),
) -> SearchResponse:
    """Search for items semantically similar to the query.

    Args:
        q: Search query text.
        limit: Maximum number of results.
        item_type: Optional filter for item type.
        min_score: Minimum similarity threshold.

    Returns:
        SearchResponse with ranked results.
    """
    try:
        vector_store = get_vector_store()

        # Search across collections
        results = vector_store.search(
            query_text=q,
            item_type=item_type,
            limit=limit,
        )

        # Filter by minimum score
        filtered_results = [r for r in results if r["similarity"] >= min_score]

        # Convert to response format
        search_results = []
        for r in filtered_results:
            metadata = r.get("metadata", {})
            search_results.append(SearchResult(
                id=r["id"],
                text=r["text"][:500] if r["text"] else "",  # Truncate for response
                title=metadata.get("title"),
                link=metadata.get("link"),
                source=metadata.get("source"),
                item_type=metadata.get("item_type", "unknown"),
                similarity=round(r["similarity"], 4),
                semantic_score=metadata.get("semantic_score"),
            ))

        return SearchResponse(
            query=q,
            results=search_results,
            total=len(search_results),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/semantic-search/similar/{item_id}")
async def find_similar(
    item_id: str,
    item_type: str = Query(..., description="Item type: news, podcast, video"),
    limit: int = Query(5, ge=1, le=50, description="Maximum results"),
    threshold: float = Query(0.7, ge=0.0, le=1.0, description="Minimum similarity"),
) -> SearchResponse:
    """Find items similar to a specific item.

    Args:
        item_id: ID of the item to find similar content for.
        item_type: Type of the item.
        limit: Maximum number of results.
        threshold: Minimum similarity threshold.

    Returns:
        SearchResponse with similar items.
    """
    try:
        vector_store = get_vector_store()

        # Get the source item
        item = vector_store.get_item(item_id, item_type)
        if not item:
            raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

        embedding = item.get("embedding")
        if not embedding:
            raise HTTPException(status_code=400, detail="Item has no embedding")

        # Find similar items
        results = vector_store.find_similar(
            embedding=embedding,
            item_type=item_type,
            threshold=threshold,
            exclude_ids=[item_id],
        )

        # Limit results
        results = results[:limit]

        # Convert to response format
        search_results = []
        for r in results:
            metadata = r.get("metadata", {})
            search_results.append(SearchResult(
                id=r["id"],
                text=r["text"][:500] if r["text"] else "",
                title=metadata.get("title"),
                link=metadata.get("link"),
                source=metadata.get("source"),
                item_type=metadata.get("item_type", item_type),
                similarity=round(r["similarity"], 4),
                semantic_score=metadata.get("semantic_score"),
            ))

        return SearchResponse(
            query=f"similar to {item_id}",
            results=search_results,
            total=len(search_results),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/semantic-search/stats")
async def get_stats() -> dict:
    """Get statistics about the vector store.

    Returns:
        Dict with collection counts.
    """
    try:
        vector_store = get_vector_store()

        stats = {}
        for item_type in ["news", "podcast", "video"]:
            try:
                stats[item_type] = vector_store.get_collection_count(item_type)
            except Exception:
                stats[item_type] = 0

        stats["total"] = sum(stats.values())

        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
