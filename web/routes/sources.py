"""Source discovery and quality management routes."""
import os
import asyncio
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session
from slowapi import Limiter

from services.source_discovery import SourceDiscoveryService, DiscoveredItem
from services.source_scoring import SourceScoringService, process_discovered_items
from services.feed_validator import FeedValidator
from web.database import get_db
from web.models import SourceQuality, DiscoveredSource, FeedSource


router = APIRouter(tags=["sources"])

def _get_real_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

limiter = Limiter(key_func=_get_real_ip)


class SourceResponse(BaseModel):
    domain: str
    quality_score: float
    total_items: int
    matched_items: int
    avg_score: float
    total_clicks: int
    total_saves: int
    citation_count: int
    is_active: bool
    is_suggested: bool


class DiscoveredSourceResponse(BaseModel):
    id: int
    domain: str
    url: str
    title: Optional[str]
    discovered_from: str
    external_score: int
    comments: int
    subreddit: Optional[str]
    status: str


class FeedCreateRequest(BaseModel):
    feed_url: str
    source_type: str  # "news", "podcast", "video"
    name: Optional[str] = None


class FeedUpdateRequest(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None  # "active", "inactive"


# ---- Page Route ----

@router.get("/sources", response_class=HTMLResponse)
async def sources_page(
    request: Request,
    db: Session = Depends(get_db),
):
    """Render sources management page."""
    # Feed sources by type
    news_feeds = db.query(FeedSource).filter(
        FeedSource.source_type == "news"
    ).order_by(FeedSource.name).all()

    podcast_feeds = db.query(FeedSource).filter(
        FeedSource.source_type == "podcast"
    ).order_by(FeedSource.name).all()

    video_feeds = db.query(FeedSource).filter(
        FeedSource.source_type == "video"
    ).order_by(FeedSource.name).all()

    # Feed stats
    total_feeds = db.query(func.count(FeedSource.id)).scalar()
    active_feeds = db.query(func.count(FeedSource.id)).filter(
        FeedSource.status == "active"
    ).scalar()
    error_feeds = db.query(func.count(FeedSource.id)).filter(
        FeedSource.status == "error"
    ).scalar()

    # Discovery data
    recent_discoveries = db.query(DiscoveredSource).filter(
        DiscoveredSource.status == "pending"
    ).order_by(
        DiscoveredSource.external_score.desc()
    ).limit(20).all()

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "sources.html",
        {
            "request": request,
            "news_feeds": news_feeds,
            "podcast_feeds": podcast_feeds,
            "video_feeds": video_feeds,
            "total_feeds": total_feeds,
            "active_feeds": active_feeds,
            "error_feeds": error_feeds,
            "recent_discoveries": recent_discoveries,
        },
    )


# ---- Feed CRUD Endpoints ----

@router.get("/api/feeds")
async def list_feeds(
    db: Session = Depends(get_db),
    source_type: Optional[str] = None,
    status: Optional[str] = None,
):
    """List feeds, filterable by source_type and status."""
    query = db.query(FeedSource)
    if source_type:
        query = query.filter(FeedSource.source_type == source_type)
    if status:
        query = query.filter(FeedSource.status == status)

    feeds = query.order_by(FeedSource.name).all()
    return {
        "feeds": [
            {
                "id": f.id,
                "name": f.name,
                "feed_url": f.feed_url,
                "source_type": f.source_type,
                "status": f.status,
                "error_message": f.error_message,
                "last_fetched": f.last_fetched.isoformat() if f.last_fetched else None,
                "item_count": f.item_count,
                "created_at": f.created_at.isoformat() if f.created_at else None,
            }
            for f in feeds
        ]
    }


@router.post("/api/feeds")
async def create_feed(
    body: FeedCreateRequest,
    db: Session = Depends(get_db),
):
    """Add a new feed source. Auto-validates and names if name not provided."""
    if body.source_type not in ("news", "podcast", "video"):
        raise HTTPException(status_code=400, detail="source_type must be news, podcast, or video")

    # Check for duplicate
    existing = db.query(FeedSource).filter(FeedSource.feed_url == body.feed_url).first()
    if existing:
        raise HTTPException(status_code=409, detail="Feed URL already exists")

    # Auto-name from URL or validation
    name = body.name
    if not name:
        result = FeedValidator.test_feed(body.feed_url)
        if result["title"]:
            name = result["title"]
        else:
            name = FeedValidator.name_from_url(body.feed_url)

    feed = FeedSource(
        name=name,
        feed_url=body.feed_url,
        source_type=body.source_type,
        status="active",
    )
    db.add(feed)
    db.commit()
    db.refresh(feed)

    return {
        "success": True,
        "feed": {
            "id": feed.id,
            "name": feed.name,
            "feed_url": feed.feed_url,
            "source_type": feed.source_type,
            "status": feed.status,
        },
    }


@router.put("/api/feeds/{feed_id}")
async def update_feed(
    feed_id: int,
    body: FeedUpdateRequest,
    db: Session = Depends(get_db),
):
    """Update a feed (name, status)."""
    feed = db.query(FeedSource).filter(FeedSource.id == feed_id).first()
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")

    if body.name is not None:
        feed.name = body.name
    if body.status is not None:
        if body.status not in ("active", "inactive"):
            raise HTTPException(status_code=400, detail="status must be active or inactive")
        feed.status = body.status
        if body.status == "active":
            feed.error_message = None

    db.commit()
    return {"success": True, "status": feed.status, "name": feed.name}


@router.delete("/api/feeds/{feed_id}")
async def delete_feed(
    feed_id: int,
    db: Session = Depends(get_db),
):
    """Remove a feed."""
    feed = db.query(FeedSource).filter(FeedSource.id == feed_id).first()
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")

    db.delete(feed)
    db.commit()
    return {"success": True}


@router.post("/api/feeds/{feed_id}/test")
async def test_feed(
    feed_id: int,
    db: Session = Depends(get_db),
):
    """Test-fetch a feed URL, update status."""
    feed = db.query(FeedSource).filter(FeedSource.id == feed_id).first()
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")

    result = FeedValidator.test_feed(feed.feed_url)

    if result["success"]:
        feed.status = "active"
        feed.error_message = None
        feed.item_count = result["item_count"]
        if result["title"] and feed.name == FeedValidator.name_from_url(feed.feed_url):
            feed.name = result["title"]
    else:
        feed.status = "error"
        feed.error_message = result["error"]

    db.commit()

    return {
        "success": result["success"],
        "title": result["title"],
        "item_count": result["item_count"],
        "error": result["error"],
        "feed_status": feed.status,
    }


@router.post("/api/feeds/import")
async def import_feeds(db: Session = Depends(get_db)):
    """Import feeds from .txt files (sources.txt, podcasts.txt, videos.txt)."""
    imported = import_feeds_from_files(db)
    return {"success": True, "imported": imported}


def import_feeds_from_files(db: Session) -> int:
    """Parse .txt files and insert FeedSource rows. Skip duplicates. Returns count imported."""
    file_map = {
        "sources.txt": "news",
        "podcasts.txt": "podcast",
        "videos.txt": "video",
    }

    # Find project root (where .txt files live)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    imported = 0
    seen_urls = set()

    for filename, source_type in file_map.items():
        filepath = os.path.join(project_root, filename)
        if not os.path.exists(filepath):
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            urls = [
                line.strip() for line in f
                if line.strip() and not line.strip().startswith("#")
            ]

        for url in urls:
            if url in seen_urls:
                continue
            seen_urls.add(url)

            existing = db.query(FeedSource).filter(FeedSource.feed_url == url).first()
            if existing:
                continue

            name = FeedValidator.name_from_url(url)
            feed = FeedSource(
                name=name,
                feed_url=url,
                source_type=source_type,
                status="active",
            )
            db.add(feed)
            imported += 1

        # Flush after each file so cross-file duplicate checks work
        db.flush()

    db.commit()
    return imported


# ---- Existing Source Quality / Discovery Endpoints ----

@router.get("/api/sources")
async def get_sources(
    db: Session = Depends(get_db),
    limit: int = 50,
    sort_by: str = "quality_score",
    suggested_only: bool = False,
):
    """Get sources list with quality scores."""
    query = db.query(SourceQuality)

    if suggested_only:
        query = query.filter(SourceQuality.is_suggested == True)

    if sort_by == "quality_score":
        query = query.order_by(SourceQuality.quality_score.desc())
    elif sort_by == "citation_count":
        query = query.order_by(SourceQuality.citation_count.desc())
    elif sort_by == "total_items":
        query = query.order_by(SourceQuality.total_items.desc())

    sources = query.limit(limit).all()

    return {
        "sources": [
            {
                "domain": s.domain,
                "quality_score": s.quality_score,
                "total_items": s.total_items,
                "matched_items": s.matched_items,
                "avg_score": round(s.avg_score, 2),
                "total_clicks": s.total_clicks,
                "total_saves": s.total_saves,
                "citation_count": s.citation_count,
                "is_active": s.is_active,
                "is_suggested": s.is_suggested,
                "last_seen": s.last_seen.isoformat() if s.last_seen else None,
            }
            for s in sources
        ]
    }


@router.get("/api/sources/stats")
async def get_source_stats(db: Session = Depends(get_db)):
    """Get overall source statistics."""
    total = db.query(func.count(SourceQuality.id)).scalar()
    active = db.query(func.count(SourceQuality.id)).filter(
        SourceQuality.is_active == True
    ).scalar()
    suggested = db.query(func.count(SourceQuality.id)).filter(
        SourceQuality.is_suggested == True
    ).scalar()
    avg_quality = db.query(func.avg(SourceQuality.quality_score)).scalar() or 0

    pending_discoveries = db.query(func.count(DiscoveredSource.id)).filter(
        DiscoveredSource.status == "pending"
    ).scalar()

    return {
        "total_sources": total,
        "active_sources": active,
        "suggested_sources": suggested,
        "avg_quality_score": round(avg_quality, 2),
        "pending_discoveries": pending_discoveries,
    }


@router.post("/api/sources/discover")
@limiter.limit("5/hour")
async def run_discovery(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Trigger source discovery from HN/Reddit."""
    async def _discover():
        service = SourceDiscoveryService()
        try:
            results = await service.discover_all()

            # Flatten all items
            all_items = []
            for source, items in results.items():
                all_items.extend(items)

            # Process in a new session
            from web.database import SessionLocal
            with SessionLocal() as session:
                domain_counts = process_discovered_items(session, all_items)

            return {
                "hackernews_top": len(results.get("hackernews_top", [])),
                "hackernews_new": len(results.get("hackernews_new", [])),
                "reddit": len(results.get("reddit", [])),
                "unique_domains": len(domain_counts),
            }
        finally:
            await service.close()

    # Run discovery
    results = await _discover()

    return {
        "success": True,
        "message": "Discovery completed",
        "results": results,
    }


@router.get("/api/sources/discoveries")
async def get_discoveries(
    db: Session = Depends(get_db),
    status: str = "pending",
    limit: int = 50,
):
    """Get discovered sources."""
    query = db.query(DiscoveredSource)

    if status != "all":
        query = query.filter(DiscoveredSource.status == status)

    discoveries = query.order_by(
        DiscoveredSource.external_score.desc()
    ).limit(limit).all()

    return {
        "discoveries": [
            {
                "id": d.id,
                "domain": d.domain,
                "url": d.url,
                "title": d.title,
                "discovered_from": d.discovered_from,
                "external_score": d.external_score,
                "comments": d.comments,
                "subreddit": d.subreddit,
                "status": d.status,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in discoveries
        ]
    }


@router.put("/api/sources/discoveries/{discovery_id}/approve")
async def approve_discovery(
    discovery_id: int,
    db: Session = Depends(get_db),
):
    """Approve a discovered source — also creates a FeedSource row."""
    discovery = db.query(DiscoveredSource).filter(
        DiscoveredSource.id == discovery_id
    ).first()

    if not discovery:
        raise HTTPException(status_code=404, detail="Discovery not found")

    discovery.status = "approved"
    discovery.reviewed_at = datetime.utcnow()

    # Update source quality to mark as active
    source = db.query(SourceQuality).filter(
        SourceQuality.domain == discovery.domain
    ).first()

    if source:
        source.is_active = True
        source.is_suggested = False

    # Also create a FeedSource if the URL looks like a feed
    existing_feed = db.query(FeedSource).filter(
        FeedSource.feed_url == discovery.url
    ).first()
    if not existing_feed:
        feed = FeedSource(
            name=discovery.title or FeedValidator.name_from_url(discovery.url),
            feed_url=discovery.url,
            source_type="news",
            status="active",
        )
        db.add(feed)

    db.commit()

    return {"success": True}


@router.put("/api/sources/discoveries/{discovery_id}/reject")
async def reject_discovery(
    discovery_id: int,
    db: Session = Depends(get_db),
):
    """Reject a discovered source."""
    discovery = db.query(DiscoveredSource).filter(
        DiscoveredSource.id == discovery_id
    ).first()

    if not discovery:
        raise HTTPException(status_code=404, detail="Discovery not found")

    discovery.status = "rejected"
    discovery.reviewed_at = datetime.utcnow()
    db.commit()

    return {"success": True}


@router.put("/api/sources/{domain}/toggle")
async def toggle_source(
    domain: str,
    db: Session = Depends(get_db),
):
    """Toggle a source's active status."""
    source = db.query(SourceQuality).filter(
        SourceQuality.domain == domain
    ).first()

    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    source.is_active = not source.is_active
    db.commit()

    return {
        "success": True,
        "is_active": source.is_active,
    }


@router.post("/api/sources/recalculate")
async def recalculate_scores(db: Session = Depends(get_db)):
    """Recalculate all source quality scores."""
    scorer = SourceScoringService(db)
    updated = scorer.recalculate_all_scores()

    return {
        "success": True,
        "updated_count": updated,
    }
