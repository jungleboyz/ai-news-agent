"""Helper functions to write digest data to the database."""
import hashlib
from datetime import date
from typing import List, Dict, Optional

from web.database import SessionLocal, init_db
from web.models import Digest, Item


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
                position=position
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
