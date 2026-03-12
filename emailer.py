import os

import resend
from dotenv import load_dotenv

load_dotenv()

# Email configuration from environment variables
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", "onboarding@resend.dev")
EMAIL_TO = os.getenv("EMAIL_TO", "robert.burden@gmail.com")


def send_digest_email(html_path: str, md_path: str, date_str: str) -> bool:
    """
    Send the daily digest via email.

    Args:
        html_path: Path to the HTML digest file
        md_path: Path to the markdown digest file
        date_str: Date string for the digest

    Returns:
        True if email sent successfully, False otherwise
    """
    if not RESEND_API_KEY:
        print("⚠ Email not configured. Set RESEND_API_KEY in .env")
        return False

    try:
        # Read HTML content
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Read markdown content for plain text fallback
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Simple markdown to plain text conversion
        plain_text = md_content
        plain_text = plain_text.replace("**", "").replace("### ", "").replace("## ", "").replace("# ", "")
        plain_text = plain_text.replace("---", "\n" + "-" * 50 + "\n")

        # Send email via Resend
        print(f"📧 Sending digest email to {EMAIL_TO}...")
        resend.api_key = RESEND_API_KEY
        resend.Emails.send({
            "from": EMAIL_FROM,
            "to": [EMAIL_TO],
            "subject": f"AI News Digest — {date_str}",
            "html": html_content,
            "text": plain_text,
        })

        print(f"✓ Email sent successfully to {EMAIL_TO}")
        return True

    except Exception as e:
        print(f"✗ Failed to send email: {e}")
        import traceback
        traceback.print_exc()
        return False
