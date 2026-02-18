"""Chat and Daily Brief routes."""
import json
import uuid
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from services.chat_rag import ChatRAGService, ChatMessage, conversation_manager
from services.daily_brief import DailyBriefService
from web.database import get_db
from web.models import Digest, EmailSubscriber


router = APIRouter(tags=["chat"])


# Request/Response models
class ChatRequest(BaseModel):
    message: str
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

    # Generate summary
    summary = service.generate_executive_summary(db, target_date)

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

    summary = service.generate_executive_summary(db, target_date)
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

    summary = service.generate_executive_summary(db, target_date)
    html = service.generate_brief_html(summary, target_date)

    return Response(content=html, media_type="text/html")


# Chat Routes

@router.get("/chat", response_class=HTMLResponse)
async def chat_page(
    request: Request,
    db: Session = Depends(get_db),
):
    """Render the chat interface."""
    chat_service = ChatRAGService()
    suggestions = chat_service.get_suggested_questions(db)

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "suggestions": suggestions,
        },
    )


@router.post("/api/chat")
async def chat(
    chat_request: ChatRequest,
    db: Session = Depends(get_db),
):
    """Process a chat message."""
    service = ChatRAGService()

    # Get or create conversation ID
    conversation_id = chat_request.conversation_id or str(uuid.uuid4())

    # Get conversation history
    history = conversation_manager.get_conversation(conversation_id)

    # Add user message to history
    user_message = ChatMessage(role="user", content=chat_request.message)
    conversation_manager.add_message(conversation_id, user_message)

    # Get response
    response = service.chat(chat_request.message, history)

    # Add assistant response to history
    conversation_manager.add_message(conversation_id, response)

    return {
        "response": response.content,
        "conversation_id": conversation_id,
        "sources": response.sources,
        "timestamp": response.timestamp.isoformat(),
    }


@router.get("/api/chat/stream")
async def chat_stream(
    message: str,
    conversation_id: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Stream a chat response using Server-Sent Events."""
    service = ChatRAGService()

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
            # Stream response
            generator = service.chat_stream(message, history)

            for chunk in generator:
                if isinstance(chunk, dict):
                    # Final metadata
                    sources = chunk.get("sources", [])
                else:
                    full_response += chunk
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"

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
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

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
    service = ChatRAGService()
    return {"suggestions": service.get_suggested_questions(db)}


# Email Subscription Routes

class SubscribeRequest(BaseModel):
    email: str
    name: Optional[str] = None


@router.post("/api/subscribe")
async def subscribe(
    request: SubscribeRequest,
    db: Session = Depends(get_db),
):
    """Subscribe to daily brief emails."""
    # Check if already subscribed
    existing = db.query(EmailSubscriber).filter(
        EmailSubscriber.email == request.email
    ).first()

    if existing:
        if existing.is_active:
            return {"success": True, "message": "Already subscribed"}
        else:
            # Reactivate
            existing.is_active = True
            existing.unsubscribed_at = None
            db.commit()
            return {"success": True, "message": "Subscription reactivated"}

    # Create new subscriber
    subscriber = EmailSubscriber(
        email=request.email,
        name=request.name,
    )
    db.add(subscriber)
    db.commit()

    return {"success": True, "message": "Successfully subscribed"}


@router.post("/api/unsubscribe")
async def unsubscribe(
    email: str,
    db: Session = Depends(get_db),
):
    """Unsubscribe from daily brief emails."""
    from datetime import datetime

    subscriber = db.query(EmailSubscriber).filter(
        EmailSubscriber.email == email
    ).first()

    if not subscriber:
        return {"success": False, "message": "Email not found"}

    subscriber.is_active = False
    subscriber.unsubscribed_at = datetime.utcnow()
    db.commit()

    return {"success": True, "message": "Successfully unsubscribed"}


@router.post("/api/brief/send")
async def send_brief_email(
    to_email: str,
    digest_date: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Send daily brief to a specific email."""
    from services.email_delivery import EmailDeliveryService

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
        raise HTTPException(status_code=500, detail=result.get("error"))

    return result
