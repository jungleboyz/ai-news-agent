"""Source discovery and quality management routes."""
import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from services.source_discovery import SourceDiscoveryService, DiscoveredItem
from services.source_scoring import SourceScoringService, process_discovered_items
from web.database import get_db
from web.models import SourceQuality, DiscoveredSource


router = APIRouter(tags=["sources"])


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


@router.get("/sources", response_class=HTMLResponse)
async def sources_page(
    request: Request,
    db: Session = Depends(get_db),
):
    """Render sources management page."""
    scorer = SourceScoringService(db)

    top_sources = scorer.get_top_sources(limit=30)
    suggested_sources = scorer.get_suggested_sources(limit=20)
    low_quality_sources = scorer.get_low_quality_sources(limit=10)

    # Get recent discoveries
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
            "top_sources": top_sources,
            "suggested_sources": suggested_sources,
            "low_quality_sources": low_quality_sources,
            "recent_discoveries": recent_discoveries,
        },
    )


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
    from sqlalchemy import func

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
async def run_discovery(
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
    """Approve a discovered source."""
    from datetime import datetime

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

    db.commit()

    return {"success": True}


@router.put("/api/sources/discoveries/{discovery_id}/reject")
async def reject_discovery(
    discovery_id: int,
    db: Session = Depends(get_db),
):
    """Reject a discovered source."""
    from datetime import datetime

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
