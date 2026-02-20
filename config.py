"""Application configuration."""
import os
from functools import lru_cache
from typing import Optional


class Settings:
    """Application settings loaded from environment variables."""

    # Environment
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = environment == "development"

    # Security
    secret_key: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    allowed_hosts: list[str] = os.getenv("ALLOWED_HOSTS", "*").split(",")
    site_username: str = os.getenv("SITE_USERNAME", "admin")
    site_password: Optional[str] = os.getenv("SITE_PASSWORD")

    # API Keys
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    firecrawl_api_key: Optional[str] = os.getenv("FIRECRAWL_API_KEY")
    jina_api_key: Optional[str] = os.getenv("JINA_API_KEY")

    # Embedding Providers (comma-separated cascade order, e.g. "openai,jina" or "jina")
    embedding_providers: str = os.getenv("EMBEDDING_PROVIDERS", "openai,jina")

    # Cron / Scheduler
    cron_secret: str = os.getenv("CRON_SECRET", "dev-cron-secret")
    scheduler_enabled: bool = os.getenv("SCHEDULER_ENABLED", "true").lower() == "true"

    # Database
    database_url: Optional[str] = os.getenv("DATABASE_URL")

    # Redis
    redis_url: Optional[str] = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Email
    smtp_host: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: Optional[str] = os.getenv("SMTP_USER")
    smtp_password: Optional[str] = os.getenv("SMTP_PASSWORD")
    from_email: Optional[str] = os.getenv("FROM_EMAIL")

    # Feature Flags
    use_semantic_scoring: bool = os.getenv("USE_SEMANTIC_SCORING", "true").lower() == "true"
    use_topic_clustering: bool = os.getenv("USE_TOPIC_CLUSTERING", "true").lower() == "true"
    use_personalization: bool = os.getenv("USE_PERSONALIZATION", "true").lower() == "true"

    # Application
    app_name: str = "NEURAL_FEED"
    app_version: str = "3.0.0"
    app_url: str = os.getenv("APP_URL", "http://localhost:8000")

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience access
settings = get_settings()
