"""Background tasks for topic clustering."""

from datetime import datetime
from typing import Optional

from services.topic_clustering import TopicClusterer
from services.vector_store import VectorStore
from services.embeddings import EmbeddingService
from web.database import SessionLocal
from web.models import Item, TopicCluster, Digest

# Service singletons
_clusterer: Optional[TopicClusterer] = None
_vector_store: Optional[VectorStore] = None
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def get_vector_store() -> VectorStore:
    """Get or create vector store singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(
            persist_dir="chromadb_data",
            embedding_service=get_embedding_service()
        )
    return _vector_store


def get_clusterer() -> TopicClusterer:
    """Get or create topic clusterer singleton."""
    global _clusterer
    if _clusterer is None:
        _clusterer = TopicClusterer()
    return _clusterer


def cluster_digest(digest_id: int, n_clusters: Optional[int] = None) -> dict:
    """Cluster all items in a digest by topic.

    Tries to retrieve embeddings from ChromaDB first, then generates fresh
    embeddings for any items that are missing (e.g. after a container restart).

    Args:
        digest_id: Database ID of the digest to cluster.
        n_clusters: Number of clusters (auto-detected if None).

    Returns:
        Dict with clustering statistics.
    """
    print(f"🔍 Clustering items for digest {digest_id}...")

    db = SessionLocal()
    try:
        # Fetch digest and items
        digest = db.query(Digest).filter(Digest.id == digest_id).first()
        if not digest:
            return {"error": f"Digest {digest_id} not found"}

        items = db.query(Item).filter(Item.digest_id == digest_id).all()
        if not items:
            return {"error": "No items found in digest"}

        if len(items) < 3:
            return {"error": f"Not enough items ({len(items)}), need at least 3"}

        print(f"  📊 Found {len(items)} items to cluster")

        # Phase 1: Try to get embeddings from ChromaDB
        vector_store = get_vector_store()
        embeddings = []
        items_with_embeddings = []
        items_missing_embeddings = []

        for item in items:
            found = False
            if item.embedding_id:
                try:
                    result = vector_store.get_item(item.embedding_id, item.type)
                    if result and result.get("embedding"):
                        embeddings.append(result["embedding"])
                        items_with_embeddings.append(item)
                        found = True
                except Exception:
                    pass
            if not found:
                items_missing_embeddings.append(item)

        chromadb_count = len(items_with_embeddings)
        print(f"  📦 Retrieved {chromadb_count} embeddings from ChromaDB, {len(items_missing_embeddings)} missing")

        # Phase 2: Generate fresh embeddings for items missing from ChromaDB
        if items_missing_embeddings:
            try:
                embedding_service = get_embedding_service()
                texts = [
                    f"{item.title} {item.summary or ''}".strip()
                    for item in items_missing_embeddings
                ]
                fresh_embeddings = embedding_service.batch_embed(texts)

                generated = 0
                for item, emb in zip(items_missing_embeddings, fresh_embeddings):
                    if emb:
                        embeddings.append(emb)
                        items_with_embeddings.append(item)
                        generated += 1

                print(f"  🧠 Generated {generated} fresh embeddings")
            except Exception as e:
                print(f"  ⚠ Fresh embedding generation failed: {e}")

        if len(embeddings) < 3:
            print(f"  ⚠ Not enough embeddings ({len(embeddings)}), need at least 3")
            return {"error": "Not enough embeddings for clustering", "count": len(embeddings)}

        # Convert items to dicts for clustering
        item_dicts = [
            {
                "id": item.id,
                "title": item.title,
                "summary": item.summary or "",
                "source": item.source or "",
                "score": item.score,
                "type": item.type,
            }
            for item in items_with_embeddings
        ]

        # Run clustering
        clusterer = get_clusterer()
        clusters = clusterer.cluster_items(item_dicts, embeddings, n_clusters)

        print(f"  🏷️ Generated {len(clusters)} topic clusters")

        # Clear existing clusters for this digest
        db.query(TopicCluster).filter(TopicCluster.digest_id == digest_id).delete()

        # Store clusters in database
        for cluster in clusters:
            topic_cluster = TopicCluster(
                cluster_id=cluster["cluster_id"],
                digest_id=digest_id,
                label=cluster["label"],
                summary=cluster["summary"],
                item_count=cluster["item_count"],
                avg_score=cluster["avg_score"],
                created_at=datetime.utcnow(),
            )
            db.add(topic_cluster)

            # Update items with cluster info
            for item_dict in cluster["items"]:
                item = db.query(Item).filter(Item.id == item_dict["id"]).first()
                if item:
                    item.cluster_id = cluster["cluster_id"]
                    item.cluster_label = cluster["label"]
                    item.cluster_confidence = item_dict.get("cluster_confidence", 0.5)

        db.commit()

        # Print cluster summary
        for cluster in clusters:
            print(f"    • {cluster['label']}: {cluster['item_count']} items")

        return {
            "digest_id": digest_id,
            "items_clustered": len(embeddings),
            "clusters_created": len(clusters),
            "clusters": [
                {
                    "cluster_id": c["cluster_id"],
                    "label": c["label"],
                    "item_count": c["item_count"],
                }
                for c in clusters
            ],
        }

    except Exception as e:
        db.rollback()
        print(f"  ❌ Clustering failed: {e}")
        return {"error": str(e)}

    finally:
        db.close()


def cluster_latest_digest(n_clusters: Optional[int] = None) -> dict:
    """Cluster items in the most recent digest.

    Args:
        n_clusters: Number of clusters (auto-detected if None).

    Returns:
        Dict with clustering statistics.
    """
    db = SessionLocal()
    try:
        digest = db.query(Digest).order_by(Digest.date.desc()).first()
        if not digest:
            return {"error": "No digests found"}
        return cluster_digest(digest.id, n_clusters)
    finally:
        db.close()


def recluster_recent_digests(days: Optional[int] = 7, n_clusters: Optional[int] = None) -> list[dict]:
    """Re-cluster digests. If days is None, process all digests.

    Args:
        days: Number of days to look back, or None for all digests.
        n_clusters: Number of clusters per digest (auto-detected if None).

    Returns:
        List of clustering results per digest.
    """
    from datetime import timedelta

    db = SessionLocal()
    results = []

    try:
        query = db.query(Digest).order_by(Digest.date.desc())

        if days is not None:
            cutoff = datetime.utcnow() - timedelta(days=days)
            query = query.filter(Digest.created_at >= cutoff)
            print(f"🔄 Re-clustering digests from last {days} days")
        else:
            print("🔄 Re-clustering ALL digests")

        digests = query.all()
        print(f"  📋 Found {len(digests)} digests to process")

        for digest in digests:
            result = cluster_digest(digest.id, n_clusters)
            results.append(result)

        return results

    finally:
        db.close()


if __name__ == "__main__":
    # Test clustering on latest digest
    result = cluster_latest_digest()
    print("\nResult:", result)
