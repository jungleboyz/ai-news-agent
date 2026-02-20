"""Search functionality routes."""
from datetime import date
from typing import Optional
from fastapi import APIRouter, Depends, Request, Query
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc

from web.database import get_db
from web.models import Digest, Item

router = APIRouter()


@router.get("/search", response_class=HTMLResponse)
async def search_page(request: Request, db: Session = Depends(get_db)):
    """Search page with filters."""
    # Get unique sources for filter dropdown
    sources = (
        db.query(Item.source)
        .distinct()
        .order_by(Item.source)
        .all()
    )
    sources = [s[0] for s in sources if s[0]]

    return request.app.state.templates.TemplateResponse(
        "search.html",
        {"request": request, "sources": sources, "items": [], "searched": False}
    )


@router.get("/search/results", response_class=HTMLResponse)
async def search_results(
    request: Request,
    q: Optional[str] = Query(None, description="Full-text search query"),
    date_from: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    type: Optional[str] = Query(None, description="news or podcast"),
    source: Optional[str] = Query(None, description="RSS feed URL"),
    score_type: Optional[str] = Query(None, description="match or fallback"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Search results with HTMX support."""
    offset = (page - 1) * per_page

    # Build query
    query = db.query(Item).join(Digest)

    # Full-text search
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
    items = (
        query
        .order_by(desc(Digest.date), Item.position)
        .offset(offset)
        .limit(per_page)
        .all()
    )

    total_pages = (total + per_page - 1) // per_page if total > 0 else 1

    # Get unique sources for filter dropdown
    sources = (
        db.query(Item.source)
        .distinct()
        .order_by(Item.source)
        .all()
    )
    sources = [s[0] for s in sources if s[0]]

    # Check if this is an HTMX request
    is_htmx = request.headers.get("HX-Request") == "true"

    if is_htmx:
        return request.app.state.templates.TemplateResponse(
            "partials/item_list.html",
            {
                "request": request,
                "items": items,
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": total_pages,
                "q": q,
                "date_from": date_from,
                "date_to": date_to,
                "type": type,
                "source": source,
                "score_type": score_type
            }
        )

    return request.app.state.templates.TemplateResponse(
        "search.html",
        {
            "request": request,
            "items": items,
            "sources": sources,
            "searched": True,
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
            "q": q,
            "date_from": date_from,
            "date_to": date_to,
            "type": type,
            "source": source,
            "score_type": score_type
        }
    )
