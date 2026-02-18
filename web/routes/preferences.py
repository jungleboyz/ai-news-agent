"""Preferences and personalization routes."""
import json
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from services.personalization import PersonalizationService, create_default_presets_for_user
from web.database import get_db
from web.models import UserProfile, Interaction, PreferencePreset, Item


router = APIRouter(tags=["preferences"])

# Cookie settings
USER_ID_COOKIE = "user_id"
COOKIE_MAX_AGE = 365 * 24 * 60 * 60  # 1 year in seconds


def get_personalization_service() -> PersonalizationService:
    """Get personalization service instance."""
    return PersonalizationService()


def get_or_create_user_id(request: Request, response: Response) -> str:
    """Get user ID from cookie or create a new one."""
    user_id = request.cookies.get(USER_ID_COOKIE)
    if not user_id:
        user_id = str(uuid.uuid4())
        response.set_cookie(
            key=USER_ID_COOKIE,
            value=user_id,
            max_age=COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
        )
    return user_id


# Request/Response models
class InteractionRequest(BaseModel):
    item_id: int
    action: str  # "click", "save", "skip", "hide"


class CreatePresetRequest(BaseModel):
    name: str
    interests: list[str]
    activate: bool = False


class InteractionResponse(BaseModel):
    success: bool
    message: str = ""


# Routes

@router.get("/preferences", response_class=HTMLResponse)
async def preferences_page(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    """Render preferences management page."""
    user_id = get_or_create_user_id(request, response)
    service = get_personalization_service()

    # Ensure user exists and has default presets
    service.get_or_create_user(db, user_id)
    create_default_presets_for_user(db, user_id, service)

    # Get user data
    presets_raw = service.get_user_presets(db, user_id)
    interactions = service.get_recent_interactions(db, user_id, limit=50)

    # Parse interests JSON for each preset
    presets = []
    for p in presets_raw:
        preset_dict = {
            "id": p.id,
            "name": p.name,
            "interests": json.loads(p.interests) if p.interests else [],
            "is_active": p.is_active,
            "created_at": p.created_at,
        }
        presets.append(preset_dict)

    # Get active preset
    active_preset = next((p for p in presets if p["is_active"]), None)

    # Enrich interactions with item data
    interaction_data = []
    for interaction in interactions:
        item = db.query(Item).filter(Item.id == interaction.item_id).first()
        if item:
            interaction_data.append({
                "id": interaction.id,
                "action": interaction.action,
                "created_at": interaction.created_at,
                "item": item,
            })

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "preferences.html",
        {
            "request": request,
            "presets": presets,
            "active_preset": active_preset,
            "interactions": interaction_data,
            "user_id": user_id[:8] + "...",
        },
    )


@router.post("/api/interactions", response_model=InteractionResponse)
async def track_interaction(
    interaction: InteractionRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    """Track a user interaction with an item."""
    user_id = get_or_create_user_id(request, response)
    service = get_personalization_service()

    try:
        service.track_interaction(db, user_id, interaction.item_id, interaction.action)
        return InteractionResponse(success=True)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to track interaction: {str(e)}")


@router.get("/api/preferences")
async def get_preferences(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    """Get user preferences data."""
    user_id = get_or_create_user_id(request, response)
    service = get_personalization_service()

    user = service.get_or_create_user(db, user_id)
    presets = service.get_user_presets(db, user_id)
    interactions = service.get_recent_interactions(db, user_id, limit=20)

    return {
        "user": {
            "id": user.user_id,
            "name": user.name,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "has_embedding": user.preference_embedding is not None,
        },
        "presets": [
            {
                "id": p.id,
                "name": p.name,
                "interests": json.loads(p.interests) if p.interests else [],
                "is_active": p.is_active,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in presets
        ],
        "recent_interactions": [
            {
                "id": i.id,
                "item_id": i.item_id,
                "action": i.action,
                "created_at": i.created_at.isoformat() if i.created_at else None,
            }
            for i in interactions
        ],
    }


@router.post("/api/preferences/presets")
async def create_preset(
    preset: CreatePresetRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    """Create a new preference preset."""
    user_id = get_or_create_user_id(request, response)
    service = get_personalization_service()

    try:
        new_preset = service.create_preset(
            db, user_id, preset.name, preset.interests, preset.activate
        )
        return {
            "success": True,
            "preset": {
                "id": new_preset.id,
                "name": new_preset.name,
                "interests": json.loads(new_preset.interests) if new_preset.interests else [],
                "is_active": new_preset.is_active,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create preset: {str(e)}")


@router.put("/api/preferences/presets/{preset_id}/activate")
async def activate_preset(
    preset_id: int,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    """Activate a preference preset."""
    user_id = get_or_create_user_id(request, response)
    service = get_personalization_service()

    success = service.activate_preset(db, user_id, preset_id)
    if not success:
        raise HTTPException(status_code=404, detail="Preset not found")

    return {"success": True}


@router.delete("/api/preferences/presets/{preset_id}")
async def delete_preset(
    preset_id: int,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    """Delete a preference preset."""
    user_id = get_or_create_user_id(request, response)
    service = get_personalization_service()

    success = service.delete_preset(db, user_id, preset_id)
    if not success:
        raise HTTPException(status_code=404, detail="Preset not found")

    return {"success": True}


@router.get("/api/recommendations")
async def get_recommendations(
    request: Request,
    response: Response,
    limit: int = 20,
    db: Session = Depends(get_db),
):
    """Get personalized item recommendations."""
    user_id = get_or_create_user_id(request, response)
    service = get_personalization_service()

    recommendations = service.get_recommendations(db, user_id, limit)

    return {
        "recommendations": [
            {
                "item_id": r["item"].id,
                "title": r["item"].title,
                "link": r["item"].link,
                "source": r["item"].source,
                "type": r["item"].type,
                "score": r["item"].score,
                "similarity": r.get("similarity", 0),
                "personalized": r.get("personalized", False),
            }
            for r in recommendations
        ],
    }


@router.post("/api/preferences/reset")
async def reset_preferences(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    """Reset user preferences and interaction history."""
    user_id = get_or_create_user_id(request, response)

    # Delete all interactions for this user
    db.query(Interaction).filter(Interaction.user_id == user_id).delete()

    # Delete all presets for this user
    db.query(PreferencePreset).filter(PreferencePreset.user_id == user_id).delete()

    # Clear preference embedding
    user = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
    if user:
        user.preference_embedding = None
        user.embedding_updated_at = None

    db.commit()

    # Recreate default presets
    service = get_personalization_service()
    create_default_presets_for_user(db, user_id, service)

    return {"success": True, "message": "Preferences reset successfully"}
