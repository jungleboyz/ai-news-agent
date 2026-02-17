"""Background task processing for AI News Agent."""

from .celery_app import celery_app
from .embedding_tasks import embed_new_items, embed_item

__all__ = ["celery_app", "embed_new_items", "embed_item"]
