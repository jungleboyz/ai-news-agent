"""Digest management tools for MCP."""

import os
import sys
from datetime import date, timedelta
from typing import Optional, List

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from web.database import SessionLocal, init_db
from web.models import Digest, Item

# Initialize database
init_db()


def get_digest_by_date(digest_date: Optional[str] = None) -> dict:
    """Get a digest by date.

    Args:
        digest_date: Date in YYYY-MM-DD format. Defaults to today.

    Returns:
        Dict with digest details and items.
    """
    try:
        if digest_date:
            target_date = date.fromisoformat(digest_date)
        else:
            target_date = date.today()

        db = SessionLocal()
        try:
            digest = db.query(Digest).filter(Digest.date == target_date).first()

            if not digest:
                return {
                    "success": False,
                    "error": f"No digest found for {target_date}",
                }

            items = (
                db.query(Item)
                .filter(Item.digest_id == digest.id)
                .order_by(Item.position)
                .all()
            )

            return {
                "success": True,
                "digest": {
                    "id": digest.id,
                    "date": str(digest.date),
                    "created_at": str(digest.created_at) if digest.created_at else None,
                    "news_sources_count": digest.news_sources_count,
                    "podcast_sources_count": digest.podcast_sources_count,
                    "total_items_considered": digest.total_items_considered,
                    "md_path": digest.md_path,
                    "html_path": digest.html_path,
                },
                "item_count": len(items),
                "items": [
                    {
                        "id": item.id,
                        "position": item.position,
                        "type": item.type,
                        "title": item.title,
                        "link": item.link,
                        "source": item.source,
                        "score": item.score,
                        "semantic_score": item.semantic_score,
                        "embedding_id": item.embedding_id,
                        "summary": item.summary,
                        "show_name": item.show_name,
                        "is_match": item.is_match,
                    }
                    for item in items
                ],
            }
        finally:
            db.close()

    except ValueError as e:
        return {
            "success": False,
            "error": f"Invalid date format: {digest_date}. Use YYYY-MM-DD.",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def list_recent_digests(limit: int = 10, include_items: bool = False) -> dict:
    """List recent digests.

    Args:
        limit: Maximum number of digests to return.
        include_items: Include item details in response.

    Returns:
        Dict with list of digests.
    """
    try:
        db = SessionLocal()
        try:
            digests = (
                db.query(Digest)
                .order_by(Digest.date.desc())
                .limit(limit)
                .all()
            )

            result = {
                "success": True,
                "digest_count": len(digests),
                "digests": [],
            }

            for digest in digests:
                item_count = db.query(Item).filter(Item.digest_id == digest.id).count()

                digest_data = {
                    "id": digest.id,
                    "date": str(digest.date),
                    "news_sources_count": digest.news_sources_count,
                    "podcast_sources_count": digest.podcast_sources_count,
                    "total_items_considered": digest.total_items_considered,
                    "item_count": item_count,
                }

                if include_items:
                    items = (
                        db.query(Item)
                        .filter(Item.digest_id == digest.id)
                        .order_by(Item.position)
                        .all()
                    )
                    digest_data["items"] = [
                        {
                            "type": item.type,
                            "title": item.title,
                            "score": item.score,
                            "semantic_score": item.semantic_score,
                        }
                        for item in items
                    ]

                result["digests"].append(digest_data)

            return result

        finally:
            db.close()

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def get_item_details(item_id: int) -> dict:
    """Get details for a specific item.

    Args:
        item_id: Database ID of the item.

    Returns:
        Dict with full item details.
    """
    try:
        db = SessionLocal()
        try:
            item = db.query(Item).filter(Item.id == item_id).first()

            if not item:
                return {
                    "success": False,
                    "error": f"Item {item_id} not found",
                }

            digest = db.query(Digest).filter(Digest.id == item.digest_id).first()

            return {
                "success": True,
                "item": {
                    "id": item.id,
                    "digest_date": str(digest.date) if digest else None,
                    "position": item.position,
                    "type": item.type,
                    "title": item.title,
                    "link": item.link,
                    "source": item.source,
                    "score": item.score,
                    "semantic_score": item.semantic_score,
                    "embedding_id": item.embedding_id,
                    "summary": item.summary,
                    "show_name": item.show_name,
                    "is_match": item.is_match,
                    "item_hash": item.item_hash,
                },
            }

        finally:
            db.close()

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def get_items_by_type(
    item_type: str,
    limit: int = 20,
    days_back: int = 7,
) -> dict:
    """Get recent items filtered by type.

    Args:
        item_type: Type filter (news, podcast, video).
        limit: Maximum items to return.
        days_back: How many days back to search.

    Returns:
        Dict with filtered items.
    """
    try:
        db = SessionLocal()
        try:
            cutoff_date = date.today() - timedelta(days=days_back)

            # Get recent digests
            digests = (
                db.query(Digest)
                .filter(Digest.date >= cutoff_date)
                .all()
            )
            digest_ids = [d.id for d in digests]

            if not digest_ids:
                return {
                    "success": True,
                    "item_count": 0,
                    "items": [],
                }

            # Get items of specified type
            items = (
                db.query(Item)
                .filter(Item.digest_id.in_(digest_ids))
                .filter(Item.type == item_type)
                .order_by(Item.score.desc())
                .limit(limit)
                .all()
            )

            return {
                "success": True,
                "item_type": item_type,
                "item_count": len(items),
                "items": [
                    {
                        "id": item.id,
                        "title": item.title,
                        "link": item.link,
                        "score": item.score,
                        "semantic_score": item.semantic_score,
                        "source": item.source or item.show_name,
                        "summary": item.summary[:200] if item.summary else None,
                    }
                    for item in items
                ],
            }

        finally:
            db.close()

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def search_items(
    query: str,
    item_type: Optional[str] = None,
    limit: int = 20,
) -> dict:
    """Full-text search across items.

    Args:
        query: Search query.
        item_type: Optional type filter.
        limit: Maximum results.

    Returns:
        Dict with search results.
    """
    try:
        from sqlalchemy import text

        db = SessionLocal()
        try:
            # Use FTS5 search
            fts_query = text("""
                SELECT items.id, items.title, items.link, items.type,
                       items.score, items.semantic_score, items.summary,
                       bm25(items_fts) as rank
                FROM items_fts
                JOIN items ON items_fts.rowid = items.id
                WHERE items_fts MATCH :query
                ORDER BY rank
                LIMIT :limit
            """)

            results = db.execute(
                fts_query,
                {"query": query, "limit": limit}
            ).fetchall()

            # Filter by type if specified
            if item_type:
                results = [r for r in results if r.type == item_type]

            return {
                "success": True,
                "query": query,
                "result_count": len(results),
                "results": [
                    {
                        "id": r.id,
                        "title": r.title,
                        "link": r.link,
                        "type": r.type,
                        "score": r.score,
                        "semantic_score": r.semantic_score,
                        "summary": r.summary[:200] if r.summary else None,
                        "rank": r.rank,
                    }
                    for r in results
                ],
            }

        finally:
            db.close()

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
