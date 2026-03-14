"""Chat and Daily Brief routes."""
import asyncio
import hashlib
import hmac
import json
import re
import uuid
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session
from slowapi import Limiter

from config import settings
from services.chat_rag import ChatRAGService, ChatMessage, conversation_manager
from services.daily_brief import DailyBriefService
from web.database import get_db
from web.models import Digest, EmailSubscriber


router = APIRouter(tags=["chat"])

# Shared service instance — avoids re-creating Anthropic client and VectorStore per request
_chat_service = ChatRAGService()

_SENTINEL = object()  # used to detect generator exhaustion in to_thread


def _sign_unsubscribe_token(email: str) -> str:
    """Create HMAC token for unsubscribe verification."""
    return hmac.new(
        settings.secret_key.encode(), email.lower().encode(), hashlib.sha256
    ).hexdigest()


def _get_real_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

limiter = Limiter(key_func=_get_real_ip)


# Request/Response models
class ChatRequest(BaseModel):
    message: str = Field(..., max_length=2000)
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    sources: list


# Daily Brief Routes

@router.get("/brief", response_class=HTMLResponse)
async def brief_page(
    request: Request,
    digest_date: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Render the daily brief page."""
    service = DailyBriefService()

    # Parse date or use latest
    target_date = None
    if digest_date:
        try:
            target_date = date.fromisoformat(digest_date)
        except ValueError:
            pass

    if not target_date:
        latest_digest = db.query(Digest).order_by(Digest.date.desc()).first()
        target_date = latest_digest.date if latest_digest else date.today()

    # Generate summary (cached)
    summary = service.get_or_generate_summary(db, target_date)

    # Get available dates
    digests = db.query(Digest).order_by(Digest.date.desc()).limit(14).all()
    available_dates = [d.date for d in digests]

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "brief.html",
        {
            "request": request,
            "summary": summary,
            "digest_date": target_date,
            "available_dates": available_dates,
        },
    )


@router.get("/api/brief")
async def get_brief(
    digest_date: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Get daily brief as JSON."""
    service = DailyBriefService()

    target_date = None
    if digest_date:
        try:
            target_date = date.fromisoformat(digest_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format")

    summary = service.get_or_generate_summary(db, target_date)
    return summary


@router.get("/api/brief/html")
async def get_brief_html(
    digest_date: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Get daily brief as HTML (for email)."""
    service = DailyBriefService()

    target_date = None
    if digest_date:
        try:
            target_date = date.fromisoformat(digest_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format")

    if not target_date:
        latest_digest = db.query(Digest).order_by(Digest.date.desc()).first()
        target_date = latest_digest.date if latest_digest else date.today()

    summary = service.get_or_generate_summary(db, target_date)
    html = service.generate_brief_html(summary, target_date)

    return Response(content=html, media_type="text/html")


@router.post("/api/brief/regenerate")
async def regenerate_brief(
    digest_date: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Force-regenerate the daily brief, bypassing cache."""
    service = DailyBriefService()

    target_date = None
    if digest_date:
        try:
            target_date = date.fromisoformat(digest_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format")

    summary = service.get_or_generate_summary(db, target_date, force_refresh=True)
    return summary


# Chat Routes

@router.get("/chat", response_class=HTMLResponse)
async def chat_page(
    request: Request,
    db: Session = Depends(get_db),
):
    """Render the chat interface."""
    suggestions = _chat_service.get_suggested_questions(db)

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "suggestions": suggestions,
        },
    )


@router.post("/api/chat")
@limiter.limit("20/minute")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    db: Session = Depends(get_db),
):
    """Process a chat message."""
    service = _chat_service

    # Get or create conversation ID
    conversation_id = chat_request.conversation_id or str(uuid.uuid4())

    # Get conversation history
    history = conversation_manager.get_conversation(conversation_id)

    # Add user message to history
    user_message = ChatMessage(role="user", content=chat_request.message)
    conversation_manager.add_message(conversation_id, user_message)

    # Get response — run sync Anthropic call in thread to avoid blocking event loop
    response = await asyncio.to_thread(service.chat, chat_request.message, history, db=db)

    # Add assistant response to history
    conversation_manager.add_message(conversation_id, response)

    return {
        "response": response.content,
        "conversation_id": conversation_id,
        "sources": response.sources,
        "timestamp": response.timestamp.isoformat(),
    }


@router.get("/api/chat/stream")
@limiter.limit("20/minute")
async def chat_stream(
    request: Request,
    message: str,
    conversation_id: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Stream a chat response using Server-Sent Events."""
    if len(message) > 2000:
        raise HTTPException(status_code=400, detail="Message too long (max 2000 characters)")

    service = _chat_service

    # Get or create conversation ID
    conv_id = conversation_id or str(uuid.uuid4())

    # Get conversation history
    history = conversation_manager.get_conversation(conv_id)

    # Add user message
    user_message = ChatMessage(role="user", content=message)
    conversation_manager.add_message(conv_id, user_message)

    async def generate():
        full_response = ""
        sources = []

        try:
            # Stream response — run sync generator in thread to avoid blocking event loop
            generator = service.chat_stream(message, history, db=db)

            while True:
                try:
                    chunk = await asyncio.to_thread(next, generator, _SENTINEL)
                    if chunk is _SENTINEL:
                        break
                    if isinstance(chunk, dict):
                        sources = chunk.get("sources", [])
                    else:
                        full_response += chunk
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                except StopIteration:
                    break

            # Send sources at the end
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            # Send done signal
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': conv_id})}\n\n"

            # Add to history
            assistant_message = ChatMessage(
                role="assistant",
                content=full_response,
                sources=sources,
            )
            conversation_manager.add_message(conv_id, assistant_message)

        except Exception as e:
            error_msg = str(e) if settings.is_development else "An error occurred"
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/api/chat/history/{conversation_id}")
async def get_chat_history(conversation_id: str):
    """Get conversation history."""
    history = conversation_manager.get_conversation(conversation_id)

    return {
        "conversation_id": conversation_id,
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                "sources": msg.sources,
            }
            for msg in history
        ],
    }


@router.delete("/api/chat/history/{conversation_id}")
async def clear_chat_history(conversation_id: str):
    """Clear conversation history."""
    conversation_manager.clear_conversation(conversation_id)
    return {"success": True}


@router.get("/api/chat/suggestions")
async def get_suggestions(db: Session = Depends(get_db)):
    """Get suggested questions."""
    return {"suggestions": _chat_service.get_suggested_questions(db)}


# Email Subscription Routes

class SubscribeRequest(BaseModel):
    email: str = Field(..., max_length=254)
    name: Optional[str] = Field(None, max_length=100)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(pattern, v):
            raise ValueError("Invalid email format")
        return v.lower().strip()


@router.post("/api/subscribe")
@limiter.limit("10/hour")
async def subscribe(
    request: Request,
    subscribe_request: SubscribeRequest,
    db: Session = Depends(get_db),
):
    """Subscribe to daily brief emails."""
    # Check if already subscribed
    existing = db.query(EmailSubscriber).filter(
        EmailSubscriber.email == subscribe_request.email
    ).first()

    if existing:
        if not existing.is_active:
            # Reactivate
            existing.is_active = True
            existing.unsubscribed_at = None
            db.commit()
    else:
        # Create new subscriber
        subscriber = EmailSubscriber(
            email=subscribe_request.email,
            name=subscribe_request.name,
        )
        db.add(subscriber)
        db.commit()

    # Always return same message to prevent email enumeration
    return {"success": True, "message": "Subscribed successfully"}


@router.post("/api/unsubscribe")
@limiter.limit("10/hour")
async def unsubscribe(
    request: Request,
    email: str,
    token: str,
    db: Session = Depends(get_db),
):
    """Unsubscribe from daily brief emails. Requires signed token."""
    from datetime import datetime

    # Verify HMAC token to prevent unauthorized unsubscribe
    expected = _sign_unsubscribe_token(email)
    if not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=403, detail="Invalid unsubscribe token")

    subscriber = db.query(EmailSubscriber).filter(
        EmailSubscriber.email == email.lower()
    ).first()

    if subscriber and subscriber.is_active:
        subscriber.is_active = False
        subscriber.unsubscribed_at = datetime.utcnow()
        db.commit()

    # Always return success to prevent email enumeration
    return {"success": True, "message": "Unsubscribed successfully"}


@router.post("/api/brief/send")
@limiter.limit("5/hour")
async def send_brief_email(
    request: Request,
    to_email: str,
    digest_date: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Send daily brief to a specific email (subscribers only)."""
    from services.email_delivery import EmailDeliveryService

    # Only allow sending to active subscribers
    subscriber = db.query(EmailSubscriber).filter(
        EmailSubscriber.email == to_email.lower().strip(),
        EmailSubscriber.is_active == True,
    ).first()

    if not subscriber:
        raise HTTPException(
            status_code=403,
            detail="Email must belong to an active subscriber"
        )

    service = EmailDeliveryService()

    if not service.is_configured():
        raise HTTPException(
            status_code=503,
            detail="Email service not configured"
        )

    target_date = None
    if digest_date:
        try:
            target_date = date.fromisoformat(digest_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format")

    result = service.send_brief(db, to_email, target_date)

    if not result.get("success"):
        detail = result.get("error") if settings.is_development else "Failed to send email"
        raise HTTPException(status_code=500, detail=detail)

    return result
