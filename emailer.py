import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

# Email configuration from environment variables
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USERNAME)
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
    if not SMTP_USERNAME or not SMTP_PASSWORD:
        print("⚠ Email not configured. Set SMTP_USERNAME and SMTP_PASSWORD in .env")
        return False
    
    try:
        # Read HTML content
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        # Read markdown content for plain text fallback
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        
        # Create message
        msg = MIMEMultipart("alternative")
        msg["From"] = EMAIL_FROM
        msg["To"] = EMAIL_TO
        msg["Subject"] = f"AI News Digest — {date_str}"
        
        # Create plain text version (from markdown, strip markdown syntax)
        plain_text = md_content
        # Simple markdown to plain text conversion
        plain_text = plain_text.replace("**", "").replace("### ", "").replace("## ", "").replace("# ", "")
        plain_text = plain_text.replace("---", "\n" + "-" * 50 + "\n")
        
        # Create HTML and plain text parts
        part1 = MIMEText(plain_text, "plain")
        part2 = MIMEText(html_content, "html")
        
        msg.attach(part1)
        msg.attach(part2)
        
        # Send email
        print(f"📧 Sending digest email to {EMAIL_TO}...")
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        
        print(f"✓ Email sent successfully to {EMAIL_TO}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to send email: {e}")
        import traceback
        traceback.print_exc()
        return False
