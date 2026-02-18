"""Source quality scoring service."""
import math
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlparse

from sqlalchemy import func
from sqlalchemy.orm import Session


class SourceScoringService:
    """Service for calculating and updating source quality scores."""

    # Scoring weights
    WEIGHTS = {
        "match_rate": 0.25,      # % of items that match AI keywords
        "avg_score": 0.20,       # Average relevance score of items
        "engagement": 0.25,     # User clicks + saves
        "citations": 0.15,       # Times cited in HN/Reddit
        "recency": 0.15,         # How recently we've seen content
    }

    def __init__(self, db: Session):
        self.db = db

    def calculate_quality_score(
        self,
        total_items: int,
        matched_items: int,
        avg_score: float,
        total_clicks: int,
        total_saves: int,
        citation_count: int,
        last_seen: Optional[datetime] = None,
    ) -> float:
        """
        Calculate a quality score (0-100) for a source.

        Components:
        - Match rate: What % of items from this source match AI keywords
        - Avg score: Average relevance/keyword score of matched items
        - Engagement: User interactions (clicks + saves weighted)
        - Citations: How often this source appears in HN/Reddit
        - Recency: How recently we've seen content from this source
        """
        # Match rate (0-100)
        match_rate = 0.0
        if total_items > 0:
            match_rate = (matched_items / total_items) * 100

        # Avg score normalized (0-100)
        # Assuming scores typically range 0-10
        avg_score_normalized = min(100, avg_score * 10)

        # Engagement score (0-100)
        # Use log scale to prevent high-traffic sources from dominating
        engagement_raw = total_clicks + (total_saves * 3)  # Saves worth more
        engagement_score = min(100, math.log(engagement_raw + 1) * 15)

        # Citation score (0-100)
        citation_score = min(100, math.log(citation_count + 1) * 20)

        # Recency score (0-100)
        recency_score = 100.0
        if last_seen:
            days_ago = (datetime.utcnow() - last_seen).days
            if days_ago > 30:
                recency_score = max(0, 100 - (days_ago - 30) * 2)

        # Weighted combination
        score = (
            self.WEIGHTS["match_rate"] * match_rate +
            self.WEIGHTS["avg_score"] * avg_score_normalized +
            self.WEIGHTS["engagement"] * engagement_score +
            self.WEIGHTS["citations"] * citation_score +
            self.WEIGHTS["recency"] * recency_score
        )

        return round(score, 2)

    def update_source_from_item(self, domain: str, item_score: float, matched: bool = True):
        """Update source quality metrics when a new item is processed."""
        from web.models import SourceQuality

        source = self.db.query(SourceQuality).filter(
            SourceQuality.domain == domain
        ).first()

        if not source:
            source = SourceQuality(
                domain=domain,
                total_items=0,
                matched_items=0,
                avg_score=0.0,
            )
            self.db.add(source)

        # Update counts
        source.total_items += 1
        if matched:
            source.matched_items += 1

        # Update running average score
        if source.matched_items > 0:
            old_total = source.avg_score * (source.matched_items - 1)
            source.avg_score = (old_total + item_score) / source.matched_items

        source.last_seen = datetime.utcnow()

        # Recalculate quality score
        source.quality_score = self.calculate_quality_score(
            total_items=source.total_items,
            matched_items=source.matched_items,
            avg_score=source.avg_score,
            total_clicks=source.total_clicks,
            total_saves=source.total_saves,
            citation_count=source.citation_count,
            last_seen=source.last_seen,
        )

        self.db.commit()
        return source

    def update_source_engagement(self, domain: str, action: str):
        """Update engagement metrics when user interacts with an item."""
        from web.models import SourceQuality

        source = self.db.query(SourceQuality).filter(
            SourceQuality.domain == domain
        ).first()

        if not source:
            return None

        if action == "click":
            source.total_clicks += 1
        elif action == "save":
            source.total_saves += 1

        # Recalculate quality score
        source.quality_score = self.calculate_quality_score(
            total_items=source.total_items,
            matched_items=source.matched_items,
            avg_score=source.avg_score,
            total_clicks=source.total_clicks,
            total_saves=source.total_saves,
            citation_count=source.citation_count,
            last_seen=source.last_seen,
        )

        self.db.commit()
        return source

    def update_citation_count(self, domain: str, count: int = 1):
        """Update citation count when source is seen in HN/Reddit."""
        from web.models import SourceQuality

        source = self.db.query(SourceQuality).filter(
            SourceQuality.domain == domain
        ).first()

        if not source:
            source = SourceQuality(
                domain=domain,
                is_suggested=True,  # Discovered from external sources
                citation_count=0,
                total_items=0,
                matched_items=0,
                avg_score=0.0,
                total_clicks=0,
                total_saves=0,
            )
            self.db.add(source)

        source.citation_count = (source.citation_count or 0) + count
        source.last_seen = datetime.utcnow()

        # Recalculate quality score
        source.quality_score = self.calculate_quality_score(
            total_items=source.total_items,
            matched_items=source.matched_items,
            avg_score=source.avg_score,
            total_clicks=source.total_clicks,
            total_saves=source.total_saves,
            citation_count=source.citation_count,
            last_seen=source.last_seen,
        )

        self.db.commit()
        return source

    def recalculate_all_scores(self):
        """Recalculate quality scores for all sources."""
        from web.models import SourceQuality

        sources = self.db.query(SourceQuality).all()
        updated = 0

        for source in sources:
            old_score = source.quality_score
            source.quality_score = self.calculate_quality_score(
                total_items=source.total_items,
                matched_items=source.matched_items,
                avg_score=source.avg_score,
                total_clicks=source.total_clicks,
                total_saves=source.total_saves,
                citation_count=source.citation_count,
                last_seen=source.last_seen,
            )
            if source.quality_score != old_score:
                updated += 1

        self.db.commit()
        return updated

    def get_top_sources(self, limit: int = 20) -> list:
        """Get top sources by quality score."""
        from web.models import SourceQuality

        return self.db.query(SourceQuality).filter(
            SourceQuality.is_active == True
        ).order_by(
            SourceQuality.quality_score.desc()
        ).limit(limit).all()

    def get_suggested_sources(self, limit: int = 20) -> list:
        """Get auto-discovered sources not yet in our feed list."""
        from web.models import SourceQuality

        return self.db.query(SourceQuality).filter(
            SourceQuality.is_suggested == True,
            SourceQuality.citation_count >= 2,  # At least 2 citations
        ).order_by(
            SourceQuality.quality_score.desc()
        ).limit(limit).all()

    def get_low_quality_sources(self, threshold: float = 30.0, limit: int = 20) -> list:
        """Get sources with low quality scores for review."""
        from web.models import SourceQuality

        return self.db.query(SourceQuality).filter(
            SourceQuality.is_active == True,
            SourceQuality.quality_score < threshold,
        ).order_by(
            SourceQuality.quality_score.asc()
        ).limit(limit).all()


def extract_domain(url: str) -> Optional[str]:
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return None


def process_discovered_items(db: Session, items: list):
    """Process discovered items and update source quality."""
    from web.models import DiscoveredSource

    scorer = SourceScoringService(db)
    domain_counts = {}

    for item in items:
        domain = item.domain
        if not domain:
            continue

        # Clean domain
        if domain.startswith("www."):
            domain = domain[4:]

        # Skip reddit/imgur domains (hosting, not sources)
        if domain in ["reddit.com", "i.redd.it", "v.redd.it", "imgur.com", "i.imgur.com"]:
            continue

        # Count citations
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        # Store discovered source
        existing = db.query(DiscoveredSource).filter(
            DiscoveredSource.url == item.url
        ).first()

        if not existing:
            discovered = DiscoveredSource(
                domain=domain,
                url=item.url,
                title=item.title,
                discovered_from=item.source,
                source_id=item.source_id,
                external_score=item.score,
                comments=item.comments,
                subreddit=item.subreddit,
            )
            db.add(discovered)

    # Update citation counts
    for domain, count in domain_counts.items():
        scorer.update_citation_count(domain, count)

    db.commit()
    return domain_counts
