"""SQLAlchemy models for the news agent database."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Date, DateTime, ForeignKey, Float, Boolean
from sqlalchemy.orm import relationship
from web.database import Base


class Digest(Base):
    """A daily digest containing news items and podcasts."""
    __tablename__ = "digests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    news_sources_count = Column(Integer, default=0)
    podcast_sources_count = Column(Integer, default=0)
    total_items_considered = Column(Integer, default=0)
    md_path = Column(Text)
    html_path = Column(Text)

    # Relationship to items
    items = relationship("Item", back_populates="digest", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Digest(id={self.id}, date={self.date})>"


class Item(Base):
    """A news article or podcast episode within a digest."""
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    digest_id = Column(Integer, ForeignKey("digests.id"), nullable=False, index=True)
    item_hash = Column(String(24), nullable=False, index=True)
    type = Column(String(10), nullable=False)  # "news" or "podcast"
    title = Column(Text, nullable=False)
    link = Column(Text, nullable=False)
    source = Column(Text)
    score = Column(Integer, default=0)
    summary = Column(Text)
    show_name = Column(Text)  # For podcasts only
    position = Column(Integer, default=0)  # Order in digest

    # Semantic search fields
    embedding_id = Column(String(64), nullable=True, index=True)  # ChromaDB reference
    semantic_score = Column(Float, nullable=True)  # 0.0-1.0 similarity score

    # Topic clustering fields
    cluster_id = Column(String(64), nullable=True, index=True)  # Reference to TopicCluster
    cluster_label = Column(String(255), nullable=True)  # Cached cluster label
    cluster_confidence = Column(Float, nullable=True)  # 0.0-1.0 confidence

    # Relationship to digest
    digest = relationship("Digest", back_populates="items")

    def __repr__(self):
        return f"<Item(id={self.id}, type={self.type}, title={self.title[:30]}...)>"

    @property
    def is_match(self) -> bool:
        """Return True if this item matched keywords (score > 0) or has high semantic score."""
        if self.semantic_score is not None:
            return self.semantic_score >= 0.3  # Semantic threshold
        return self.score > 0  # Fallback to keyword score


class TopicCluster(Base):
    """A topic cluster grouping related items."""
    __tablename__ = "topic_clusters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cluster_id = Column(String(64), unique=True, nullable=False, index=True)
    digest_id = Column(Integer, ForeignKey("digests.id"), nullable=False, index=True)
    label = Column(String(255), nullable=False)  # AI-generated topic label
    summary = Column(Text)  # Cross-source synthesis summary
    item_count = Column(Integer, default=0)
    avg_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to digest
    digest = relationship("Digest")

    def __repr__(self):
        return f"<TopicCluster(id={self.id}, label={self.label}, items={self.item_count})>"


class UserProfile(Base):
    """User profile for personalization (cookie-based, no auth required)."""
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), unique=True, nullable=False, index=True)  # Cookie-based ID
    name = Column(String(100), nullable=True)  # Optional display name
    created_at = Column(DateTime, default=datetime.utcnow)

    # Cached preference embedding (JSON array stored as text)
    preference_embedding = Column(Text, nullable=True)
    embedding_updated_at = Column(DateTime, nullable=True)

    # Relationships
    interactions = relationship("Interaction", back_populates="user", cascade="all, delete-orphan")
    presets = relationship("PreferencePreset", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<UserProfile(id={self.id}, user_id={self.user_id[:8]}...)>"


class Interaction(Base):
    """User interaction with an item (click, save, skip, hide)."""
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), ForeignKey("user_profiles.user_id"), nullable=False, index=True)
    item_id = Column(Integer, ForeignKey("items.id"), nullable=False, index=True)
    action = Column(String(20), nullable=False)  # "click", "save", "skip", "hide"
    created_at = Column(DateTime, default=datetime.utcnow)

    # Action weights for learning
    # click: +1.0, save: +3.0, skip: -0.5, hide: -2.0

    # Relationships
    user = relationship("UserProfile", back_populates="interactions")
    item = relationship("Item")

    def __repr__(self):
        return f"<Interaction(user={self.user_id[:8]}..., item={self.item_id}, action={self.action})>"


class PreferencePreset(Base):
    """Named preference preset with interest list."""
    __tablename__ = "preference_presets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), ForeignKey("user_profiles.user_id"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    interests = Column(Text, nullable=False)  # JSON array of interest strings
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Cached embedding for this preset
    preset_embedding = Column(Text, nullable=True)

    # Relationship
    user = relationship("UserProfile", back_populates="presets")

    def __repr__(self):
        return f"<PreferencePreset(id={self.id}, name={self.name}, active={self.is_active})>"


class SourceQuality(Base):
    """Quality metrics for a content source (domain/feed)."""
    __tablename__ = "source_quality"

    id = Column(Integer, primary_key=True, autoincrement=True)
    domain = Column(String(255), unique=True, nullable=False, index=True)  # e.g., "arxiv.org"
    feed_url = Column(Text, nullable=True)  # RSS feed URL if known

    # Quality metrics
    total_items = Column(Integer, default=0)  # Total items from this source
    matched_items = Column(Integer, default=0)  # Items that matched AI keywords
    avg_score = Column(Float, default=0.0)  # Average relevance score
    total_clicks = Column(Integer, default=0)  # User clicks on items from this source
    total_saves = Column(Integer, default=0)  # User saves
    citation_count = Column(Integer, default=0)  # Times cited in HN/Reddit

    # Computed quality score (0-100)
    quality_score = Column(Float, default=50.0)

    # Status
    is_active = Column(Boolean, default=True)
    is_suggested = Column(Boolean, default=False)  # True if auto-discovered
    last_seen = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<SourceQuality(domain={self.domain}, score={self.quality_score:.1f})>"


class DiscoveredSource(Base):
    """A source discovered from HackerNews or Reddit."""
    __tablename__ = "discovered_sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    domain = Column(String(255), nullable=False, index=True)
    url = Column(Text, nullable=False)  # Original URL where discovered
    title = Column(Text)  # Title of the content
    discovered_from = Column(String(20), nullable=False)  # "hackernews" or "reddit"
    source_id = Column(String(64))  # HN story ID or Reddit post ID
    external_score = Column(Integer, default=0)  # HN points or Reddit upvotes
    comments = Column(Integer, default=0)
    subreddit = Column(String(100), nullable=True)  # For Reddit discoveries

    # Processing status
    status = Column(String(20), default="pending")  # pending, approved, rejected
    reviewed_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<DiscoveredSource(domain={self.domain}, from={self.discovered_from})>"


class EmailSubscriber(Base):
    """Email subscriber for daily briefs."""
    __tablename__ = "email_subscribers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    unsubscribed_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<EmailSubscriber(email={self.email}, active={self.is_active})>"
