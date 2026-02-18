"""Email delivery service for daily briefs."""
import os
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from services.daily_brief import DailyBriefService


class EmailDeliveryService:
    """Service for sending daily briefs via email."""

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        from_email: Optional[str] = None,
    ):
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = smtp_user or os.getenv("SMTP_USER")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD")
        self.from_email = from_email or os.getenv("FROM_EMAIL", self.smtp_user)

    def is_configured(self) -> bool:
        """Check if email is properly configured."""
        return bool(self.smtp_user and self.smtp_password)

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
                "error": "Email not configured. Set SMTP_USER and SMTP_PASSWORD environment variables.",
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

        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"AI News Brief - {digest_date.strftime('%B %d, %Y')}"
        msg["From"] = self.from_email
        msg["To"] = to_email

        # Attach both plain text and HTML versions
        part1 = MIMEText(text_content, "plain")
        part2 = MIMEText(html_content, "html")

        msg.attach(part1)
        msg.attach(part2)

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_email, to_email, msg.as_string())

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
