#!/usr/bin/env python3
"""Re-index all DB items into ChromaDB for semantic search.

Usage:
    python scripts/reindex_chromadb.py [--batch-size 50] [--type news]

Reads items from the database, generates embeddings, and stores them
in ChromaDB.  Skips items that are already in the vector store.
"""

import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from web.database import SessionLocal, init_db
from web.models import Item, Digest
from services.vector_store import VectorStore
from services.embeddings import EmbeddingService


def reindex(batch_size: int = 50, item_type: str | None = None):
    init_db()
    db = SessionLocal()

    embedding_service = EmbeddingService()
    vector_store = VectorStore(embedding_service=embedding_service)

    # Get counts before
    for t in ["news", "podcast", "video"]:
        print(f"  ChromaDB {t}: {vector_store.get_collection_count(t)} items")

    # Query items to index
    query = db.query(Item).join(Digest).filter(Item.summary.isnot(None))
    if item_type:
        query = query.filter(Item.type == item_type)
    items = query.order_by(Item.id).all()

    print(f"\nTotal items to index: {len(items)}")

    # Check which items already exist in ChromaDB
    indexed = 0
    skipped = 0
    errors = 0

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        items_to_store = []

        for item in batch:
            item_id = f"{item.type}_{item.id}"
            itype = item.type if item.type in ("news", "podcast", "video") else "news"

            # Check if already indexed
            existing = vector_store.get_item(item_id, itype)
            if existing:
                skipped += 1
                continue

            text = f"{item.title}\n{item.summary or ''}"
            items_to_store.append({
                "id": item_id,
                "text": text,
                "metadata": {
                    "title": item.title,
                    "type": item.type,
                    "source": item.source or "",
                    "score": item.score or 0,
                    "link": item.link,
                    "date": str(item.digest.date) if item.digest else "",
                },
            })

        if not items_to_store:
            continue

        # Group by type and batch store
        by_type = {}
        for it in items_to_store:
            t = it["metadata"]["type"]
            if t not in ("news", "podcast", "video"):
                t = "news"
            by_type.setdefault(t, []).append(it)

        for t, type_items in by_type.items():
            try:
                stored = vector_store.add_items_batch(type_items, t)
                indexed += len(stored)
            except Exception as e:
                errors += len(type_items)
                print(f"  Error indexing batch: {e}")
                # Wait before retrying on rate limit
                time.sleep(5)

        progress = i + len(batch)
        print(f"  Progress: {progress}/{len(items)} (indexed={indexed}, skipped={skipped}, errors={errors})")

        # Small delay between batches to avoid rate limits
        if items_to_store:
            time.sleep(1)

    db.close()

    print(f"\nDone! Indexed: {indexed}, Skipped: {skipped}, Errors: {errors}")
    print("\nFinal ChromaDB counts:")
    for t in ["news", "podcast", "video"]:
        print(f"  {t}: {vector_store.get_collection_count(t)} items")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-index DB items into ChromaDB")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for embedding API calls")
    parser.add_argument("--type", type=str, default=None, help="Only index this item type (news/podcast/video)")
    args = parser.parse_args()

    reindex(batch_size=args.batch_size, item_type=args.type)
