"""MCP resource handlers for AI News Agent."""

from .config_resources import (
    get_interests,
    get_sources,
    get_podcasts,
    get_videos,
    update_interests,
)
from .digest_resources import (
    get_digest_markdown,
    get_digest_html,
    list_available_digests,
)

__all__ = [
    "get_interests",
    "get_sources",
    "get_podcasts",
    "get_videos",
    "update_interests",
    "get_digest_markdown",
    "get_digest_html",
    "list_available_digests",
]
