"""Digest list and detail view routes."""
from datetime import date
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc

from web.database import get_db
from web.models import Digest, Item

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def homepage(request: Request, db: Session = Depends(get_db)):
    """Homepage showing the 5 most recent digests."""
    digests = (
        db.query(Digest)
        .order_by(desc(Digest.date))
        .limit(5)
        .all()
    )
    return request.app.state.templates.TemplateResponse(
        "index.html",
        {"request": request, "digests": digests}
    )


@router.get("/digests", response_class=HTMLResponse)
async def digests_list(
    request: Request,
    page: int = 1,
    per_page: int = 10,
    db: Session = Depends(get_db)
):
    """Browse all digests with pagination."""
    offset = (page - 1) * per_page
    total = db.query(Digest).count()
    digests = (
        db.query(Digest)
        .order_by(desc(Digest.date))
        .offset(offset)
        .limit(per_page)
        .all()
    )
    total_pages = (total + per_page - 1) // per_page

    return request.app.state.templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "digests": digests,
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
            "show_pagination": True
        }
    )


@router.get("/digest/{digest_date}", response_class=HTMLResponse)
async def digest_detail(
    request: Request,
    digest_date: str,
    db: Session = Depends(get_db)
):
    """View a single digest by date."""
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

    # Get previous and next digests for navigation
    prev_digest = (
        db.query(Digest)
        .filter(Digest.date < parsed_date)
        .order_by(desc(Digest.date))
        .first()
    )
    next_digest = (
        db.query(Digest)
        .filter(Digest.date > parsed_date)
        .order_by(Digest.date)
        .first()
    )

    return request.app.state.templates.TemplateResponse(
        "digest.html",
        {
            "request": request,
            "digest": digest,
            "items": items,
            "prev_digest": prev_digest,
            "next_digest": next_digest
        }
    )
