"""Import existing digest markdown files into the database."""
import os
import re
import hashlib
from datetime import datetime
from glob import glob

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.database import SessionLocal, init_db
from web.models import Digest, Item


def parse_digest_header(content: str) -> dict:
    """Parse the header of a digest markdown file."""
    header = {}

    # Extract date from title
    title_match = re.search(r"# AI News Digest — (\d{4}-\d{2}-\d{2})", content)
    if title_match:
        header["date"] = title_match.group(1)

    # Parse stats line
    # Format: Sources: X | New items considered: Y | Podcasts: Z | Total items: W
    # or: Sources: X | New items considered: Y | Total items: W
    stats_match = re.search(
        r"Sources:\s*(\d+)\s*\|\s*New items considered:\s*(\d+)"
        r"(?:\s*\|\s*Podcasts:\s*(\d+))?"
        r"\s*\|\s*Total items:\s*(\d+)",
        content
    )
    if stats_match:
        header["news_sources_count"] = int(stats_match.group(1))
        header["total_items_considered"] = int(stats_match.group(2))
        header["podcast_sources_count"] = int(stats_match.group(3) or 0)
        header["total_items"] = int(stats_match.group(4))

    return header


def parse_items(content: str) -> list:
    """Parse items from a digest markdown file."""
    items = []

    # Pattern to match item blocks - supports both formats:
    # Format 1 (newer): ### N. ICON [SCORE] (TAG) Title
    # Format 2 (older): ## N. [SCORE] (TAG) Title
    item_pattern_new = re.compile(
        r"###?\s*(\d+)\.\s*"
        r"(📰|🎙️)\s*"
        r"\[(\d+)\]\s*"
        r"\((MATCH|FALLBACK)\)\s*"
        r"(.+?)(?=\n)",
        re.MULTILINE
    )

    item_pattern_old = re.compile(
        r"##\s*(\d+)\.\s*"
        r"\[(\d+)\]\s*"
        r"\((MATCH|FALLBACK)\)\s*"
        r"(.+?)(?=\n)",
        re.MULTILINE
    )

    # Split content into item sections (handle both ## and ###)
    sections = re.split(r"(?=##\s*\d+\.)", content)

    for section in sections:
        # Try new format first
        match = item_pattern_new.search(section)
        if match:
            position = int(match.group(1))
            item_type = "news" if match.group(2) == "📰" else "podcast"
            score = int(match.group(3))
            title = match.group(5).strip()
        else:
            # Try old format
            match = item_pattern_old.search(section)
            if not match:
                continue
            position = int(match.group(1))
            item_type = "news"  # Old format was news-only
            score = int(match.group(2))
            title = match.group(4).strip()

        # Extract link
        link_match = re.search(r"- Link:\s*(.+)", section)
        link = link_match.group(1).strip() if link_match else ""

        # Extract source (for news)
        source_match = re.search(r"- Source:\s*(.+)", section)
        source = source_match.group(1).strip() if source_match else ""

        # Extract show name (for podcasts)
        show_match = re.search(r"- Show:\s*(.+)", section)
        show_name = show_match.group(1).strip() if show_match else None

        # Detect podcast from show name if present
        if show_name:
            item_type = "podcast"

        # Extract summary
        summary = ""
        summary_match = re.search(
            r"\*\*(Why this matters|Summary):\*\*\s*\n(.+?)(?=\n---|\Z)",
            section,
            re.DOTALL
        )
        if summary_match:
            summary = summary_match.group(2).strip()

        # Generate hash
        raw = f"{title}|{link}"
        item_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

        items.append({
            "position": position,
            "type": item_type,
            "title": title,
            "link": link,
            "source": source if item_type == "news" else show_name,
            "score": score,
            "summary": summary,
            "show_name": show_name,
            "item_hash": item_hash
        })

    return items


def import_digest(md_path: str, db) -> bool:
    """Import a single digest file into the database."""
    print(f"Processing: {md_path}")

    try:
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"  Error reading file: {e}")
        return False

    # Parse header
    header = parse_digest_header(content)
    if "date" not in header:
        print(f"  Skipping: Could not parse date from file")
        return False

    # Check if digest already exists
    digest_date = datetime.strptime(header["date"], "%Y-%m-%d").date()
    existing = db.query(Digest).filter(Digest.date == digest_date).first()
    if existing:
        print(f"  Skipping: Digest for {header['date']} already exists")
        return False

    # Determine HTML path
    html_path = md_path.replace(".md", ".html")
    if not os.path.exists(html_path):
        html_path = None

    # Create digest
    digest = Digest(
        date=digest_date,
        news_sources_count=header.get("news_sources_count", 0),
        podcast_sources_count=header.get("podcast_sources_count", 0),
        total_items_considered=header.get("total_items_considered", 0),
        md_path=md_path,
        html_path=html_path
    )
    db.add(digest)
    db.flush()  # Get the ID

    # Parse and add items
    items = parse_items(content)
    for item_data in items:
        item = Item(
            digest_id=digest.id,
            item_hash=item_data["item_hash"],
            type=item_data["type"],
            title=item_data["title"],
            link=item_data["link"],
            source=item_data["source"],
            score=item_data["score"],
            summary=item_data["summary"],
            show_name=item_data["show_name"],
            position=item_data["position"]
        )
        db.add(item)

    db.commit()
    print(f"  Imported: {len(items)} items")
    return True


def update_fts_index(db):
    """Rebuild the FTS index from items table."""
    print("\nRebuilding full-text search index...")
    from sqlalchemy import text
    from web.database import engine

    try:
        # Use raw connection to avoid transaction issues
        with engine.connect() as conn:
            # Get all items
            items = conn.execute(text("SELECT id, title, summary FROM items")).fetchall()

            for item in items:
                item_id, title, summary = item
                # Insert into FTS
                conn.execute(
                    text("INSERT OR REPLACE INTO items_fts(rowid, title, summary) VALUES (:id, :title, :summary)"),
                    {"id": item_id, "title": title or "", "summary": summary or ""}
                )
            conn.commit()
        print(f"  FTS index rebuilt successfully ({len(items)} items)")
    except Exception as e:
        print(f"  Warning: Could not rebuild FTS index: {e}")


def main():
    """Main entry point for the migration script."""
    print("=" * 60)
    print("AI News Agent - Import Existing Digests")
    print("=" * 60)

    # Initialize database
    print("\nInitializing database...")
    init_db()

    # Find all digest files
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "out")
    digest_files = sorted(glob(os.path.join(out_dir, "digest-*.md")))

    if not digest_files:
        print(f"\nNo digest files found in {out_dir}")
        return

    print(f"\nFound {len(digest_files)} digest files")
    print("-" * 60)

    # Import each digest
    db = SessionLocal()
    try:
        imported = 0
        for md_path in digest_files:
            if import_digest(md_path, db):
                imported += 1

        # Rebuild FTS index
        update_fts_index(db)

        print("-" * 60)
        print(f"\nImport complete: {imported} digests imported")

    finally:
        db.close()


if __name__ == "__main__":
    main()
