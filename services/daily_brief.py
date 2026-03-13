"""Daily brief and executive summary generation service."""
import json
import logging
import os
from datetime import date, datetime
from typing import Optional

from anthropic import Anthropic

logger = logging.getLogger(__name__)


class DailyBriefService:
    """Service for generating executive summaries and daily briefs."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = None
        if self.api_key:
            self.client = Anthropic(api_key=self.api_key)

    def _get_items_for_brief(self, db, digest_date: Optional[date] = None) -> list:
        """Get items for generating the brief."""
        from web.models import Item, Digest

        query = db.query(Item).join(Digest)

        if digest_date:
            query = query.filter(Digest.date == digest_date)
        else:
            # Get latest digest
            latest_digest = db.query(Digest).order_by(Digest.date.desc()).first()
            if latest_digest:
                query = query.filter(Digest.digest_id == latest_digest.id)

        return query.filter(Item.score > 0).order_by(Item.score.desc()).limit(50).all()

    def _format_items_for_prompt(self, items: list) -> str:
        """Format items for the Claude prompt."""
        formatted = []
        for i, item in enumerate(items[:30], 1):
            formatted.append(f"{i}. [{item.type.upper()}] {item.title}")
            if item.summary:
                formatted.append(f"   Summary: {item.summary[:200]}...")
            formatted.append(f"   Source: {item.source or 'Unknown'} | Score: {item.score}")
            formatted.append("")
        return "\n".join(formatted)

    def generate_executive_summary(
        self,
        db,
        digest_date: Optional[date] = None,
        max_tokens: int = 1500,
    ) -> dict:
        """
        Generate an executive summary for the daily digest.

        Returns a dict with:
        - headline: One-line summary
        - key_insights: 3-5 bullet points
        - top_stories: Top 3 stories with brief descriptions
        - emerging_trends: Notable trends or themes
        - full_summary: Comprehensive summary paragraph
        """
        if not self.client:
            return {
                "error": "Anthropic API key not configured",
                "headline": "Daily AI News Summary",
                "key_insights": ["API key required for AI-generated summaries"],
                "top_stories": [],
                "emerging_trends": [],
                "full_summary": "Configure ANTHROPIC_API_KEY to enable AI summaries.",
            }

        items = self._get_items_for_brief(db, digest_date)
        if not items:
            return {
                "headline": "No news items found",
                "key_insights": [],
                "top_stories": [],
                "emerging_trends": [],
                "full_summary": "No items available for the selected date.",
            }

        items_text = self._format_items_for_prompt(items)

        prompt = f"""You are an AI news analyst. Based on the following news items and podcast episodes from today's AI news digest, create an executive summary.

NEWS ITEMS:
{items_text}

Generate a JSON response with the following structure:
{{
    "headline": "One compelling headline summarizing the day's most important AI news (max 100 chars)",
    "key_insights": [
        "Key insight 1 (one sentence)",
        "Key insight 2 (one sentence)",
        "Key insight 3 (one sentence)"
    ],
    "top_stories": [
        {{
            "title": "Story title",
            "summary": "2-3 sentence summary of why this matters",
            "category": "research|product|business|policy|other"
        }}
    ],
    "emerging_trends": [
        "Trend or theme observed across multiple stories"
    ],
    "full_summary": "A 2-3 paragraph comprehensive summary of the day's AI news, highlighting the most significant developments and their implications."
}}

Focus on:
- What's genuinely newsworthy and significant
- Connections between stories
- Implications for the AI industry
- Balance between technical and business news

Return ONLY valid JSON, no markdown formatting."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response
            content = response.content[0].text

            # Clean up potential markdown formatting
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            summary = json.loads(content.strip())
            summary["generated_at"] = datetime.utcnow().isoformat()
            summary["item_count"] = len(items)

            return summary

        except Exception as e:
            return {
                "error": str(e),
                "headline": "Error generating summary",
                "key_insights": [f"Error: {str(e)}"],
                "top_stories": [],
                "emerging_trends": [],
                "full_summary": f"Failed to generate summary: {str(e)}",
            }

    def get_or_generate_summary(
        self,
        db,
        digest_date: Optional[date] = None,
        force_refresh: bool = False,
    ) -> dict:
        """Return cached brief from DB, or generate and cache it."""
        from web.models import Digest

        # Resolve date
        if not digest_date:
            latest_digest = db.query(Digest).order_by(Digest.date.desc()).first()
            digest_date = latest_digest.date if latest_digest else date.today()

        digest = db.query(Digest).filter(Digest.date == digest_date).first()

        # Try cache first
        if not force_refresh and digest and digest.brief_json:
            logger.info("Using cached brief for %s", digest_date)
            return json.loads(digest.brief_json)

        # Generate fresh summary
        logger.info("Generating new brief for %s", digest_date)
        summary = self.generate_executive_summary(db, digest_date)

        # Cache it if we have a digest row and no error
        if digest and "error" not in summary:
            digest.brief_json = json.dumps(summary)
            digest.brief_generated_at = datetime.utcnow()
            db.commit()
            logger.info("Cached brief for %s", digest_date)

        return summary

    def generate_brief_html(self, summary: dict, digest_date: date) -> str:
        """Generate HTML version of the brief for email/web."""
        from config import settings

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NEURAL_FEED Brief - {digest_date}</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            background: #0a0a0f;
            color: #e2e8f0;
        }}
        h1 {{
            font-family: 'Orbitron', sans-serif;
            color: #00f5ff;
            font-size: 24px;
            margin-bottom: 4px;
            letter-spacing: 2px;
        }}
        .date {{
            color: #ff00ff;
            font-size: 13px;
            margin-bottom: 24px;
            letter-spacing: 1px;
        }}
        .divider {{
            height: 2px;
            background: linear-gradient(90deg, #00f5ff, #ff00ff, #00f5ff);
            border: none;
            margin: 24px 0;
            border-radius: 1px;
        }}
        .headline {{
            font-size: 20px;
            font-weight: 600;
            color: #f8fafc;
            margin-bottom: 24px;
            padding: 16px;
            background: #0f0f18;
            border-left: 4px solid #00f5ff;
            border-radius: 4px;
        }}
        h2 {{
            font-family: 'Orbitron', sans-serif;
            color: #00f5ff;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-top: 32px;
            margin-bottom: 16px;
        }}
        ul {{
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 12px;
            color: #cbd5e1;
        }}
        .story {{
            background: #0f0f18;
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 16px;
            border-top: 2px solid;
            border-image: linear-gradient(90deg, #00f5ff, #ff00ff) 1;
        }}
        .story-title {{
            font-weight: 600;
            color: #f8fafc;
            margin-bottom: 8px;
        }}
        .story-summary {{
            color: #94a3b8;
            font-size: 14px;
        }}
        .category {{
            display: inline-block;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 2px 10px;
            background: rgba(0, 245, 255, 0.1);
            color: #00f5ff;
            border: 1px solid rgba(0, 245, 255, 0.3);
            border-radius: 12px;
            margin-top: 8px;
        }}
        .full-summary {{
            color: #cbd5e1;
            white-space: pre-line;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid;
            border-image: linear-gradient(90deg, #00f5ff, #ff00ff, #00f5ff) 1;
            color: #64748b;
            font-size: 12px;
            text-align: center;
        }}
        .footer .brand {{
            font-family: 'Orbitron', sans-serif;
            color: #00f5ff;
            font-size: 13px;
            letter-spacing: 2px;
        }}
        a {{
            color: #00f5ff;
        }}
        a:hover {{
            color: #ff00ff;
        }}
    </style>
</head>
<body>
    <h1>NEURAL_FEED</h1>
    <div class="date">{digest_date.strftime('%B %d, %Y').upper()}</div>

    <hr class="divider">

    <div class="headline">{summary.get('headline', 'Daily Summary')}</div>

    <h2>Key Insights</h2>
    <ul>
"""
        for insight in summary.get('key_insights', []):
            html += f"        <li>{insight}</li>\n"

        html += """    </ul>

    <hr class="divider">

    <h2>Top Stories</h2>
"""
        for story in summary.get('top_stories', []):
            html += f"""    <div class="story">
        <div class="story-title">{story.get('title', '')}</div>
        <div class="story-summary">{story.get('summary', '')}</div>
        <span class="category">{story.get('category', 'news')}</span>
    </div>
"""

        if summary.get('emerging_trends'):
            html += """    <hr class="divider">

    <h2>Emerging Trends</h2>
    <ul>
"""
            for trend in summary.get('emerging_trends', []):
                html += f"        <li>{trend}</li>\n"
            html += "    </ul>\n"

        html += f"""
    <hr class="divider">

    <h2>Full Summary</h2>
    <div class="full-summary">{summary.get('full_summary', '')}</div>

    <div class="footer">
        <p class="brand">NEURAL_FEED</p>
        <p><a href="{settings.app_url}">View Full Digest</a></p>
    </div>
</body>
</html>
"""
        return html

    def generate_brief_text(self, summary: dict, digest_date: date) -> str:
        """Generate plain text version of the brief."""
        from config import settings

        text = f"""NEURAL_FEED DAILY BRIEF - {digest_date.strftime('%B %d, %Y')}
{'=' * 50}

{summary.get('headline', 'Daily Summary')}

KEY INSIGHTS
{'-' * 20}
"""
        for i, insight in enumerate(summary.get('key_insights', []), 1):
            text += f"{i}. {insight}\n"

        text += f"""
TOP STORIES
{'-' * 20}
"""
        for story in summary.get('top_stories', []):
            text += f"\n* {story.get('title', '')}\n"
            text += f"  {story.get('summary', '')}\n"
            text += f"  [{story.get('category', 'news').upper()}]\n"

        if summary.get('emerging_trends'):
            text += f"""
EMERGING TRENDS
{'-' * 20}
"""
            for trend in summary.get('emerging_trends', []):
                text += f"- {trend}\n"

        text += f"""
FULL SUMMARY
{'-' * 20}
{summary.get('full_summary', '')}

---
NEURAL_FEED
View full digest: {settings.app_url}
"""
        return text


# Convenience function
def generate_daily_brief(db, digest_date: Optional[date] = None) -> dict:
    """Generate a daily brief for the given date."""
    service = DailyBriefService()
    return service.generate_executive_summary(db, digest_date)
