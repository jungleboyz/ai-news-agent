"""Generic JSON cache helpers used by all agents.

Replaces the duplicated load_seen/save_seen/load_summaries/save_summaries
functions across agent.py, podcast_agent.py, video_agent.py, and
web_scraper_agent.py.
"""
import json
import os
from typing import Any, Dict, Set


def load_json(path: str, default: Any = None) -> Any:
    """Load a JSON file, returning *default* if it doesn't exist or is corrupt."""
    if default is None:
        default = {}
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def save_json(path: str, data: Any) -> None:
    """Atomically-ish write *data* as JSON to *path*."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_set(path: str) -> Set[str]:
    """Load a set stored as ``{"seen": [...]}`` (video_agent format)."""
    data = load_json(path, {})
    return set(data.get("seen", []))


def save_set(path: str, data: Set[str]) -> None:
    """Save a set as ``{"seen": [...]}``."""
    save_json(path, {"seen": list(data)})
