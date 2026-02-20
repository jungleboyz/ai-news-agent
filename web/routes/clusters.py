"""Routes for topic cluster browsing and API."""

from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, Depends, Request, Query
from fastapi.responses import HTMLResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from web.database import get_db
from web.models import TopicCluster, Item, Digest

router = APIRouter()


@router.get("/topics", response_class=HTMLResponse)
async def topics_page(
    request: Request,
    db: Session = Depends(get_db),
    digest_date: Optional[str] = Query(None, description="Filter by digest date (YYYY-MM-DD)"),
):
    """Browse topics page with cluster cards."""
    templates = request.app.state.templates

    # Get clusters, optionally filtered by date
    query = db.query(TopicCluster).join(Digest)

    if digest_date:
        try:
            filter_date = datetime.strptime(digest_date, "%Y-%m-%d").date()
            query = query.filter(Digest.date == filter_date)
        except ValueError:
            pass

    clusters = query.order_by(TopicCluster.item_count.desc()).limit(50).all()

    # Get available dates for filter
    dates = (
        db.query(Digest.date)
        .join(TopicCluster)
        .distinct()
        .order_by(Digest.date.desc())
        .limit(30)
        .all()
    )
    available_dates = [d[0] for d in dates]

    # Get items for each cluster (top 3 preview)
    clusters_with_items = []
    for cluster in clusters:
        items = (
            db.query(Item)
            .filter(Item.cluster_id == cluster.cluster_id)
            .order_by(Item.score.desc())
            .limit(3)
            .all()
        )
        clusters_with_items.append({
            "cluster": cluster,
            "items": items,
            "digest_date": cluster.digest.date if cluster.digest else None,
        })

    return templates.TemplateResponse(
        "clusters.html",
        {
            "request": request,
            "clusters": clusters_with_items,
            "available_dates": available_dates,
            "selected_date": digest_date,
        },
    )


@router.get("/topics/{cluster_id}", response_class=HTMLResponse)
async def topic_detail(
    request: Request,
    cluster_id: str,
    db: Session = Depends(get_db),
):
    """View all items in a specific topic cluster."""
    templates = request.app.state.templates

    cluster = db.query(TopicCluster).filter(TopicCluster.cluster_id == cluster_id).first()
    if not cluster:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": "Topic not found"},
            status_code=404,
        )

    items = (
        db.query(Item)
        .filter(Item.cluster_id == cluster_id)
        .order_by(Item.score.desc())
        .all()
    )

    return templates.TemplateResponse(
        "topic_detail.html",
        {
            "request": request,
            "cluster": cluster,
            "items": items,
        },
    )


# API Endpoints

@router.get("/api/topics")
async def api_list_topics(
    db: Session = Depends(get_db),
    digest_date: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
):
    """List all topic clusters."""
    query = db.query(TopicCluster).join(Digest)

    if digest_date:
        try:
            filter_date = datetime.strptime(digest_date, "%Y-%m-%d").date()
            query = query.filter(Digest.date == filter_date)
        except ValueError:
            pass

    clusters = query.order_by(TopicCluster.item_count.desc()).limit(limit).all()

    return {
        "clusters": [
            {
                "cluster_id": c.cluster_id,
                "label": c.label,
                "summary": c.summary,
                "item_count": c.item_count,
                "avg_score": c.avg_score,
                "digest_date": c.digest.date.isoformat() if c.digest else None,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in clusters
        ],
        "count": len(clusters),
    }


@router.get("/api/topics/{cluster_id}")
async def api_topic_detail(
    cluster_id: str,
    db: Session = Depends(get_db),
):
    """Get details of a specific topic cluster with all items."""
    cluster = db.query(TopicCluster).filter(TopicCluster.cluster_id == cluster_id).first()
    if not cluster:
        return {"error": "Topic not found"}

    items = (
        db.query(Item)
        .filter(Item.cluster_id == cluster_id)
        .order_by(Item.score.desc())
        .all()
    )

    return {
        "cluster": {
            "cluster_id": cluster.cluster_id,
            "label": cluster.label,
            "summary": cluster.summary,
            "item_count": cluster.item_count,
            "avg_score": cluster.avg_score,
            "digest_date": cluster.digest.date.isoformat() if cluster.digest else None,
        },
        "items": [
            {
                "id": item.id,
                "title": item.title,
                "link": item.link,
                "source": item.source,
                "score": item.score,
                "type": item.type,
                "summary": item.summary,
                "cluster_confidence": item.cluster_confidence,
            }
            for item in items
        ],
    }


@router.get("/api/digest/{digest_date}/clusters")
async def api_digest_clusters(
    digest_date: str,
    db: Session = Depends(get_db),
):
    """Get all clusters for a specific digest date."""
    try:
        filter_date = datetime.strptime(digest_date, "%Y-%m-%d").date()
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD"}

    digest = db.query(Digest).filter(Digest.date == filter_date).first()
    if not digest:
        return {"error": f"No digest found for {digest_date}"}

    clusters = (
        db.query(TopicCluster)
        .filter(TopicCluster.digest_id == digest.id)
        .order_by(TopicCluster.item_count.desc())
        .all()
    )

    result = []
    for cluster in clusters:
        items = (
            db.query(Item)
            .filter(Item.cluster_id == cluster.cluster_id)
            .order_by(Item.score.desc())
            .all()
        )
        result.append({
            "cluster_id": cluster.cluster_id,
            "label": cluster.label,
            "summary": cluster.summary,
            "item_count": cluster.item_count,
            "avg_score": cluster.avg_score,
            "items": [
                {
                    "id": item.id,
                    "title": item.title,
                    "link": item.link,
                    "source": item.source,
                    "score": item.score,
                    "type": item.type,
                    "cluster_confidence": item.cluster_confidence,
                }
                for item in items
            ],
        })

    return {
        "digest_date": digest_date,
        "cluster_count": len(clusters),
        "clusters": result,
    }


@router.post("/api/clusters/rebuild")
async def api_rebuild_clusters(
    db: Session = Depends(get_db),
    days: int = Query(7, ge=1, le=30),
):
    """Rebuild topic clusters for recent digests (generates missing embeddings)."""
    from tasks.clustering_tasks import recluster_recent_digests
    results = recluster_recent_digests(days=days)
    total_clusters = sum(r.get("clusters_created", 0) for r in results if "error" not in r)
    return {
        "success": True,
        "digests_processed": len(results),
        "total_clusters_created": total_clusters,
        "results": results,
    }


@router.get("/api/clusters/stats")
async def api_cluster_stats(
    db: Session = Depends(get_db),
):
    """Get clustering statistics."""
    total_clusters = db.query(func.count(TopicCluster.id)).scalar()
    total_items_clustered = db.query(func.count(Item.id)).filter(Item.cluster_id.isnot(None)).scalar()

    # Clusters per digest
    clusters_by_digest = (
        db.query(Digest.date, func.count(TopicCluster.id).label("count"))
        .join(TopicCluster)
        .group_by(Digest.date)
        .order_by(Digest.date.desc())
        .limit(10)
        .all()
    )

    # Largest clusters
    largest = (
        db.query(TopicCluster)
        .order_by(TopicCluster.item_count.desc())
        .limit(5)
        .all()
    )

    return {
        "total_clusters": total_clusters,
        "total_items_clustered": total_items_clustered,
        "clusters_by_date": [
            {"date": d.isoformat(), "count": c} for d, c in clusters_by_digest
        ],
        "largest_clusters": [
            {"label": c.label, "item_count": c.item_count} for c in largest
        ],
    }
