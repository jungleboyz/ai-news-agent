"""Celery application configuration."""

import os

from celery import Celery

# Redis URL from environment or default to localhost
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "ai_news_agent",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks.embedding_tasks"],
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task execution settings
    task_acks_late=True,  # Acknowledge after task completes
    task_reject_on_worker_lost=True,  # Requeue if worker dies
    task_time_limit=600,  # 10 minute hard limit
    task_soft_time_limit=540,  # 9 minute soft limit

    # Rate limiting for API calls
    task_default_rate_limit="100/m",  # 100 tasks per minute

    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour

    # Worker settings
    worker_prefetch_multiplier=1,  # One task at a time for rate limiting
    worker_concurrency=2,  # Two concurrent workers

    # Retry settings
    task_default_retry_delay=60,  # 1 minute between retries
    task_max_retries=3,
)

# Task routes - can be expanded for different queues
celery_app.conf.task_routes = {
    "tasks.embedding_tasks.*": {"queue": "embeddings"},
}

# Beat schedule for periodic tasks (if needed later)
celery_app.conf.beat_schedule = {
    # Example: Process any pending embeddings every 5 minutes
    # "process-pending-embeddings": {
    #     "task": "tasks.embedding_tasks.process_pending_embeddings",
    #     "schedule": 300.0,
    # },
}
