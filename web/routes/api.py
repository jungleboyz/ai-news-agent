"""JSON API endpoints."""
from datetime import date
from typing import Optional, List
from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import desc

from web.database import get_db
from web.models import Digest, Item

router = APIRouter(prefix="/api")


class ItemResponse(BaseModel):
    """API response model for an item."""
    id: int
    type: str
    title: str
    link: str
    source: Optional[str]
    score: int
    summary: Optional[str]
    show_name: Optional[str]
    position: int
    digest_date: date

    class Config:
        from_attributes = True


class DigestResponse(BaseModel):
    """API response model for a digest."""
    id: int
    date: date
    news_sources_count: int
    podcast_sources_count: int
    total_items_considered: int
    item_count: int

    class Config:
        from_attributes = True


class DigestDetailResponse(DigestResponse):
    """API response model for a digest with items."""
    items: List[ItemResponse]


class PaginatedItemsResponse(BaseModel):
    """Paginated items response."""
    items: List[ItemResponse]
    total: int
    page: int
    per_page: int
    total_pages: int


class PaginatedDigestsResponse(BaseModel):
    """Paginated digests response."""
    digests: List[DigestResponse]
    total: int
    page: int
    per_page: int
    total_pages: int


@router.get("/digests", response_model=PaginatedDigestsResponse)
async def api_digests(
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get a list of all digests."""
    offset = (page - 1) * per_page
    total = db.query(Digest).count()
    digests = (
        db.query(Digest)
        .order_by(desc(Digest.date))
        .offset(offset)
        .limit(per_page)
        .all()
    )
    total_pages = (total + per_page - 1) // per_page if total > 0 else 1

    return PaginatedDigestsResponse(
        digests=[
            DigestResponse(
                id=d.id,
                date=d.date,
                news_sources_count=d.news_sources_count,
                podcast_sources_count=d.podcast_sources_count,
                total_items_considered=d.total_items_considered,
                item_count=len(d.items)
            )
            for d in digests
        ],
        total=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages
    )


@router.get("/digests/{digest_date}", response_model=DigestDetailResponse)
async def api_digest_detail(
    digest_date: str,
    db: Session = Depends(get_db)
):
    """Get a single digest by date with all its items."""
    try:
        parsed_date = date.fromisoformat(digest_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    digest = db.query(Digest).filter(Digest.date == parsed_date).first()
    if not digest:
        raise HTTPException(status_code=404, detail="Digest not found")

    items = (
        db.query(Item)
        .filter(Item.digest_id == digest.id)
        .order_by(Item.position)
        .all()
    )

    return DigestDetailResponse(
        id=digest.id,
        date=digest.date,
        news_sources_count=digest.news_sources_count,
        podcast_sources_count=digest.podcast_sources_count,
        total_items_considered=digest.total_items_considered,
        item_count=len(items),
        items=[
            ItemResponse(
                id=item.id,
                type=item.type,
                title=item.title,
                link=item.link,
                source=item.source,
                score=item.score,
                summary=item.summary,
                show_name=item.show_name,
                position=item.position,
                digest_date=digest.date
            )
            for item in items
        ]
    )


@router.get("/items", response_model=PaginatedItemsResponse)
async def api_items(
    q: Optional[str] = Query(None, description="Search query"),
    date_from: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    type: Optional[str] = Query(None, description="news or podcast"),
    source: Optional[str] = Query(None, description="RSS feed URL"),
    score_type: Optional[str] = Query(None, description="match or fallback"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get items with filters."""
    offset = (page - 1) * per_page

    # Build query
    query = db.query(Item).join(Digest)

    # Search filter
    if q and q.strip():
        search_term = f"%{q.strip()}%"
        query = query.filter(
            (Item.title.ilike(search_term)) | (Item.summary.ilike(search_term))
        )

    # Date range filter
    if date_from:
        try:
            from_date = date.fromisoformat(date_from)
            query = query.filter(Digest.date >= from_date)
        except ValueError:
            pass

    if date_to:
        try:
            to_date = date.fromisoformat(date_to)
            query = query.filter(Digest.date <= to_date)
        except ValueError:
            pass

    # Type filter
    if type in ("news", "podcast"):
        query = query.filter(Item.type == type)

    # Source filter
    if source:
        query = query.filter(Item.source == source)

    # Score type filter
    if score_type == "match":
        query = query.filter(Item.score > 0)
    elif score_type == "fallback":
        query = query.filter(Item.score == 0)

    # Get total count
    total = query.count()

    # Get paginated results
    items_result = (
        query
        .order_by(desc(Digest.date), Item.position)
        .offset(offset)
        .limit(per_page)
        .all()
    )

    total_pages = (total + per_page - 1) // per_page if total > 0 else 1

    return PaginatedItemsResponse(
        items=[
            ItemResponse(
                id=item.id,
                type=item.type,
                title=item.title,
                link=item.link,
                source=item.source,
                score=item.score,
                summary=item.summary,
                show_name=item.show_name,
                position=item.position,
                digest_date=item.digest.date
            )
            for item in items_result
        ],
        total=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages
    )
