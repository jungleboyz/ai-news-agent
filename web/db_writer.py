"""Helper functions to write digest data to the database."""
import hashlib
from datetime import date, timedelta
from typing import List, Dict, Optional, Set

from web.database import SessionLocal, init_db
from web.models import Digest, Item


def get_seen_hashes_from_db(days: int = 30, item_type: str = None) -> Set[str]:
    """Load item hashes from the database for deduplication.

    Returns a set of item_hash values from digests in the last N days.
    This ensures items don't repeat across days even if the file-based
    seen cache (out/seen.json) is wiped by a container restart.

    Args:
        days: How many days back to check (default 30).
        item_type: Optional filter by type ("news", "podcast", "video", "web").
    """
    try:
        init_db()
        db = SessionLocal()
        try:
            cutoff = date.today() - timedelta(days=days)
            today = date.today()
            query = (
                db.query(Item.item_hash)
                .join(Digest)
                .filter(Digest.date >= cutoff, Digest.date < today)
            )
            if item_type:
                query = query.filter(Item.type == item_type)
            rows = query.all()
            return {row[0] for row in rows}
        finally:
            db.close()
    except Exception as e:
        print(f"  ⚠ Could not load seen hashes from DB: {e}")
        return set()


def get_seen_links_from_db(days: int = 30) -> Set[str]:
    """Load item links from the database for cross-source deduplication.

    Returns a set of link URLs from digests in the last N days,
    excluding today so that re-runs can regenerate today's digest.
    """
    try:
        init_db()
        db = SessionLocal()
        try:
            cutoff = date.today() - timedelta(days=days)
            today = date.today()
            rows = (
                db.query(Item.link)
                .join(Digest)
                .filter(Digest.date >= cutoff, Digest.date < today)
                .all()
            )
            return {row[0] for row in rows}
        finally:
            db.close()
    except Exception as e:
        print(f"  ⚠ Could not load seen links from DB: {e}")
        return set()


def save_digest_to_db(
    digest_date: date,
    news_sources_count: int,
    podcast_sources_count: int,
    total_items_considered: int,
    items: List[Dict],
    md_path: Optional[str] = None,
    html_path: Optional[str] = None
) -> int:
    """
    Save a digest and its items to the database.

    Args:
        digest_date: The date of the digest
        news_sources_count: Number of RSS news sources processed
        podcast_sources_count: Number of podcast feeds processed
        total_items_considered: Total items before filtering
        items: List of item dicts with keys:
            - type: "news" or "podcast"
            - title: Item title
            - link: Item URL
            - source: RSS feed URL (news) or show name (podcast)
            - score: Keyword match score
            - summary: AI-generated summary
            - show_name: Podcast show name (optional)
        md_path: Path to the markdown file
        html_path: Path to the HTML file

    Returns:
        The ID of the created digest
    """
    # Initialize database if needed
    init_db()

    db = SessionLocal()
    try:
        # Check if digest already exists for this date
        existing = db.query(Digest).filter(Digest.date == digest_date).first()
        if existing:
            # Update existing digest
            existing.news_sources_count = news_sources_count
            existing.podcast_sources_count = podcast_sources_count
            existing.total_items_considered = total_items_considered
            existing.md_path = md_path
            existing.html_path = html_path

            # Delete existing items
            db.query(Item).filter(Item.digest_id == existing.id).delete()
            db.flush()
            digest = existing
        else:
            # Create new digest
            digest = Digest(
                date=digest_date,
                news_sources_count=news_sources_count,
                podcast_sources_count=podcast_sources_count,
                total_items_considered=total_items_considered,
                md_path=md_path,
                html_path=html_path
            )
            db.add(digest)
            db.flush()

        # Add items
        for position, item_data in enumerate(items, 1):
            # Generate hash
            raw = f"{item_data['title']}|{item_data['link']}"
            item_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

            item = Item(
                digest_id=digest.id,
                item_hash=item_hash,
                type=item_data.get("type", "news"),
                title=item_data["title"],
                link=item_data["link"],
                source=item_data.get("source", ""),
                score=item_data.get("score", 0),
                summary=item_data.get("summary", ""),
                show_name=item_data.get("show_name"),
                position=position,
                embedding_id=item_data.get("embedding_id"),
                semantic_score=item_data.get("semantic_score"),
            )
            db.add(item)

        db.commit()

        # Update FTS index for new items
        _update_fts_for_digest(db, digest.id)

        return digest.id

    finally:
        db.close()


def _update_fts_for_digest(db, digest_id: int):
    """Update the FTS index for items in a specific digest."""
    from sqlalchemy import text

    try:
        # Get all items for this digest
        items = db.query(Item).filter(Item.digest_id == digest_id).all()

        for item in items:
            # Delete existing FTS entry if any
            db.execute(
                text("DELETE FROM items_fts WHERE rowid = :id"),
                {"id": item.id}
            )
            # Insert new FTS entry
            db.execute(
                text("INSERT INTO items_fts(rowid, title, summary) VALUES (:id, :title, :summary)"),
                {"id": item.id, "title": item.title, "summary": item.summary or ""}
            )

        db.commit()
    except Exception as e:
        print(f"Warning: Could not update FTS index: {e}")
        db.rollback()
