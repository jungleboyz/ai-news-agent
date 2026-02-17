"""Digest resource handlers for MCP."""

import os
import sys
from datetime import date
from typing import Optional, List

# Add parent directories to path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)

from web.database import SessionLocal, init_db
from web.models import Digest, Item

# Initialize database
init_db()


def list_available_digests(limit: int = 30) -> dict:
    """List all available digests.

    Args:
        limit: Maximum number of digests to return.

    Returns:
        Dict with list of digest dates and metadata.
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

            return {
                "success": True,
                "digest_count": len(digests),
                "digests": [
                    {
                        "date": str(d.date),
                        "id": d.id,
                        "has_markdown": bool(d.md_path and os.path.exists(d.md_path)),
                        "has_html": bool(d.html_path and os.path.exists(d.html_path)),
                        "news_sources": d.news_sources_count,
                        "podcast_sources": d.podcast_sources_count,
                        "items_considered": d.total_items_considered,
                    }
                    for d in digests
                ],
            }
        finally:
            db.close()

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def get_digest_markdown(digest_date: Optional[str] = None) -> dict:
    """Get digest content as markdown.

    Args:
        digest_date: Date in YYYY-MM-DD format. Defaults to today.

    Returns:
        Dict with markdown content or generated markdown.
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

            # Try to read from file first
            if digest.md_path and os.path.exists(digest.md_path):
                with open(digest.md_path, "r") as f:
                    content = f.read()
                return {
                    "success": True,
                    "date": str(target_date),
                    "source": "file",
                    "file_path": digest.md_path,
                    "content": content,
                }

            # Generate markdown from database
            items = (
                db.query(Item)
                .filter(Item.digest_id == digest.id)
                .order_by(Item.position)
                .all()
            )

            content = f"# AI News Digest — {target_date}\n\n"
            content += f"Sources: {digest.news_sources_count} news, "
            content += f"{digest.podcast_sources_count} podcasts\n"
            content += f"Items considered: {digest.total_items_considered}\n\n"
            content += "---\n\n"

            for i, item in enumerate(items, 1):
                icon = {"news": "📰", "podcast": "🎙️", "video": "📺"}.get(item.type, "📄")
                tag = "MATCH" if item.is_match else "FALLBACK"

                content += f"## {i}. {icon} [{item.score}] ({tag}) {item.title}\n\n"

                if item.type == "news":
                    content += f"- **Link:** {item.link}\n"
                    content += f"- **Source:** {item.source}\n"
                elif item.type == "podcast":
                    content += f"- **Show:** {item.show_name}\n"
                    content += f"- **Link:** {item.link}\n"
                else:
                    content += f"- **Channel:** {item.show_name}\n"
                    content += f"- **Link:** {item.link}\n"

                if item.semantic_score:
                    content += f"- **Semantic Score:** {item.semantic_score:.3f}\n"

                if item.summary:
                    label = "Why this matters" if item.type == "news" else "Summary"
                    content += f"\n**{label}:**\n{item.summary}\n"

                content += "\n---\n\n"

            return {
                "success": True,
                "date": str(target_date),
                "source": "database",
                "content": content,
            }

        finally:
            db.close()

    except ValueError:
        return {
            "success": False,
            "error": f"Invalid date format: {digest_date}. Use YYYY-MM-DD.",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def get_digest_html(digest_date: Optional[str] = None) -> dict:
    """Get digest content as HTML.

    Args:
        digest_date: Date in YYYY-MM-DD format. Defaults to today.

    Returns:
        Dict with HTML content.
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

            # Read from file
            if digest.html_path and os.path.exists(digest.html_path):
                with open(digest.html_path, "r") as f:
                    content = f.read()
                return {
                    "success": True,
                    "date": str(target_date),
                    "file_path": digest.html_path,
                    "content": content,
                }

            return {
                "success": False,
                "error": f"No HTML file found for {target_date}",
            }

        finally:
            db.close()

    except ValueError:
        return {
            "success": False,
            "error": f"Invalid date format: {digest_date}. Use YYYY-MM-DD.",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def get_digest_items_by_score(
    digest_date: Optional[str] = None,
    min_score: int = 0,
    item_type: Optional[str] = None,
) -> dict:
    """Get digest items filtered by score.

    Args:
        digest_date: Date in YYYY-MM-DD format. Defaults to today.
        min_score: Minimum score threshold.
        item_type: Optional type filter.

    Returns:
        Dict with filtered items.
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

            query = db.query(Item).filter(Item.digest_id == digest.id)

            if min_score > 0:
                query = query.filter(Item.score >= min_score)

            if item_type:
                query = query.filter(Item.type == item_type)

            items = query.order_by(Item.score.desc()).all()

            return {
                "success": True,
                "date": str(target_date),
                "filters": {
                    "min_score": min_score,
                    "item_type": item_type,
                },
                "item_count": len(items),
                "items": [
                    {
                        "id": item.id,
                        "type": item.type,
                        "title": item.title,
                        "score": item.score,
                        "semantic_score": item.semantic_score,
                        "link": item.link,
                        "source": item.source or item.show_name,
                    }
                    for item in items
                ],
            }

        finally:
            db.close()

    except ValueError:
        return {
            "success": False,
            "error": f"Invalid date format: {digest_date}. Use YYYY-MM-DD.",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
