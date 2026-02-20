"""Database connection and session management.

Supports both SQLite (development) and PostgreSQL (production).
Set DATABASE_URL environment variable for PostgreSQL.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool

# Determine database URL
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    # Production: Use DATABASE_URL (PostgreSQL)
    # Railway uses postgres:// but SQLAlchemy needs postgresql://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,  # Handle connection drops
        pool_size=5,
        max_overflow=10,
    )
else:
    # Development: Use SQLite
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATABASE_PATH = os.path.join(BASE_DIR, "data", "news_agent.db")
    DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

    # Ensure data directory exists
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """Dependency for FastAPI routes to get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def is_sqlite() -> bool:
    """Check if using SQLite database."""
    return str(engine.url).startswith("sqlite")


def init_db():
    """Create all tables."""
    from sqlalchemy import text
    from web.models import (
        Digest, Item, TopicCluster, UserProfile, Interaction,
        PreferencePreset, SourceQuality, DiscoveredSource, EmailSubscriber,
        FeedSource
    )  # noqa: F401

    Base.metadata.create_all(bind=engine)

    # Create FTS virtual table for SQLite only
    if is_sqlite():
        with engine.connect() as conn:
            try:
                conn.execute(text("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
                        title,
                        summary,
                        content='items',
                        content_rowid='id'
                    )
                """))
                conn.commit()
            except Exception:
                # FTS might already exist or not be supported
                pass
