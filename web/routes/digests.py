"""Digest list and detail view routes."""
import json
import re
from collections import OrderedDict
from datetime import date
from typing import Optional, List
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc

from web.database import get_db
from web.models import Digest, Item, UserProfile, PreferencePreset

router = APIRouter()

FOR_YOU_LIMIT = 10  # Max items in the "For You" section


def _get_active_preset(db: Session, request: Request) -> Optional[PreferencePreset]:
    """Get the user's active preference preset from their cookie, if any."""
    user_id = request.cookies.get("user_id")
    if not user_id:
        return None
    preset = (
        db.query(PreferencePreset)
        .filter(
            PreferencePreset.user_id == user_id,
            PreferencePreset.is_active == True,
        )
        .first()
    )
    return preset


def _score_item_for_preset(item: Item, interests: List[str]) -> int:
    """Score a digest item against a list of interest keywords. Returns match count."""
    text = f"{item.title} {item.summary or ''} {item.source or ''} {item.cluster_label or ''}".lower()
    text = re.sub(r"\s+", " ", text)
    score = 0
    for interest in interests:
        kw = interest.lower().strip()
        if kw and kw in text:
            score += 1
    return score


def _get_for_you_items(
    items: List[Item], preset: PreferencePreset, limit: int = FOR_YOU_LIMIT
) -> List[dict]:
    """Score and rank digest items against the active preset's interests.

    Returns list of {"item": Item, "match_count": int} sorted by match count desc.
    Only includes items with at least 1 match.
    """
    interests = json.loads(preset.interests) if preset.interests else []
    if not interests:
        return []

    scored = []
    for item in items:
        match_count = _score_item_for_preset(item, interests)
        if match_count > 0:
            scored.append({"item": item, "match_count": match_count})

    scored.sort(key=lambda x: x["match_count"], reverse=True)
    return scored[:limit]


@router.get("/", response_class=HTMLResponse)
async def homepage(
    request: Request,
    digest_date: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Homepage shows the Daily Brief directly."""
    from services.daily_brief import DailyBriefService

    service = DailyBriefService()

    target_date = None
    if digest_date:
        try:
            target_date = date.fromisoformat(digest_date)
        except ValueError:
            pass

    if not target_date:
        latest_digest = db.query(Digest).order_by(Digest.date.desc()).first()
        target_date = latest_digest.date if latest_digest else date.today()

    summary = service.get_or_generate_summary(db, target_date)

    digests = db.query(Digest).order_by(Digest.date.desc()).limit(14).all()
    available_dates = [d.date for d in digests]

    return request.app.state.templates.TemplateResponse(
        "brief.html",
        {
            "request": request,
            "summary": summary,
            "digest_date": target_date,
            "available_dates": available_dates,
        },
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

    # Group items by cluster label (preserving position order within each)
    clustered_items = OrderedDict()
    unclustered = []
    for item in items:
        label = item.cluster_label
        if label:
            clustered_items.setdefault(label, []).append(item)
        else:
            unclustered.append(item)

    # If no clusters exist, fall back to flat list
    has_clusters = bool(clustered_items)
    if unclustered:
        clustered_items["Other Stories"] = unclustered

    # "For You" section based on active preset
    for_you_items = []
    active_preset_name = None
    preset = _get_active_preset(db, request)
    if preset:
        active_preset_name = preset.name
        for_you_items = _get_for_you_items(items, preset)

    return request.app.state.templates.TemplateResponse(
        "digest.html",
        {
            "request": request,
            "digest": digest,
            "items": items,
            "clustered_items": clustered_items if has_clusters else {},
            "prev_digest": prev_digest,
            "next_digest": next_digest,
            "for_you_items": for_you_items,
            "active_preset_name": active_preset_name,
        }
    )
