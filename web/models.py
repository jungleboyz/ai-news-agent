"""SQLAlchemy models for the news agent database."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Date, DateTime, ForeignKey
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

    # Relationship to digest
    digest = relationship("Digest", back_populates="items")

    def __repr__(self):
        return f"<Item(id={self.id}, type={self.type}, title={self.title[:30]}...)>"

    @property
    def is_match(self) -> bool:
        """Return True if this item matched keywords (score > 0)."""
        return self.score > 0
