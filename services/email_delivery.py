"""Email delivery service for daily briefs."""
import os
from datetime import date
from typing import Optional

import resend

from services.daily_brief import DailyBriefService


class EmailDeliveryService:
    """Service for sending daily briefs via email."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        from_email: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("RESEND_API_KEY")
        self.from_email = from_email or os.getenv("FROM_EMAIL") or os.getenv("EMAIL_FROM", "onboarding@resend.dev")

    def is_configured(self) -> bool:
        """Check if email is properly configured."""
        return bool(self.api_key)

    def send_brief(
        self,
        db,
        to_email: str,
        digest_date: Optional[date] = None,
    ) -> dict:
        """
        Send daily brief to an email address.

        Args:
            db: Database session
            to_email: Recipient email address
            digest_date: Date of digest (defaults to latest)

        Returns:
            Dict with success status and message
        """
        if not self.is_configured():
            return {
                "success": False,
                "error": "Email not configured. Set RESEND_API_KEY environment variable.",
            }

        # Generate brief
        brief_service = DailyBriefService()

        if not digest_date:
            from web.models import Digest
            latest = db.query(Digest).order_by(Digest.date.desc()).first()
            digest_date = latest.date if latest else date.today()

        summary = brief_service.generate_executive_summary(db, digest_date)
        html_content = brief_service.generate_brief_html(summary, digest_date)
        text_content = brief_service.generate_brief_text(summary, digest_date)

        try:
            resend.api_key = self.api_key
            resend.Emails.send({
                "from": self.from_email,
                "to": [to_email],
                "subject": f"AI News Brief - {digest_date.strftime('%B %d, %Y')}",
                "html": html_content,
                "text": text_content,
            })

            return {
                "success": True,
                "message": f"Brief sent to {to_email}",
                "digest_date": digest_date.isoformat(),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def send_to_subscribers(self, db, digest_date: Optional[date] = None) -> dict:
        """
        Send daily brief to all subscribers.

        Returns:
            Dict with results for each subscriber
        """
        from web.models import EmailSubscriber

        subscribers = db.query(EmailSubscriber).filter(
            EmailSubscriber.is_active == True
        ).all()

        if not subscribers:
            return {
                "success": True,
                "message": "No active subscribers",
                "sent": 0,
            }

        results = []
        for subscriber in subscribers:
            result = self.send_brief(db, subscriber.email, digest_date)
            results.append({
                "email": subscriber.email,
                **result,
            })

        successful = sum(1 for r in results if r.get("success"))

        return {
            "success": True,
            "total": len(subscribers),
            "sent": successful,
            "failed": len(subscribers) - successful,
            "results": results,
        }


# Add subscriber model if it doesn't exist
def ensure_subscriber_model():
    """Ensure EmailSubscriber model exists in database."""
    # This is handled in models.py - just a placeholder
    pass
