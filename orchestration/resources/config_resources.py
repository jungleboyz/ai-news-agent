"""Configuration resource handlers for MCP."""

import os
import sys
import json
from typing import List

# Add parent directories to path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)


def get_interests() -> dict:
    """Get current user interests configuration.

    Returns:
        Dict with interests list and metadata.
    """
    try:
        from agent import USER_INTERESTS
        from services.semantic_scorer import SemanticScorer

        # Get default semantic interests
        scorer = SemanticScorer()
        semantic_interests = scorer.interests

        return {
            "success": True,
            "keyword_interests": USER_INTERESTS,
            "semantic_interests": semantic_interests,
            "description": "Keyword interests are used for legacy scoring. Semantic interests are embedded for similarity matching.",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def get_sources() -> dict:
    """Get RSS news sources configuration.

    Returns:
        Dict with sources list and count.
    """
    try:
        sources_path = os.path.join(PROJECT_DIR, "sources.txt")

        if not os.path.exists(sources_path):
            return {
                "success": True,
                "source_count": 0,
                "sources": [],
                "file_path": sources_path,
            }

        with open(sources_path, "r") as f:
            lines = f.readlines()

        sources = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                sources.append(line)

        return {
            "success": True,
            "source_count": len(sources),
            "sources": sources,
            "file_path": sources_path,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def get_podcasts() -> dict:
    """Get podcast feeds configuration.

    Returns:
        Dict with podcast feeds list and count.
    """
    try:
        podcasts_path = os.path.join(PROJECT_DIR, "podcasts.txt")

        if not os.path.exists(podcasts_path):
            return {
                "success": True,
                "podcast_count": 0,
                "podcasts": [],
                "file_path": podcasts_path,
            }

        with open(podcasts_path, "r") as f:
            lines = f.readlines()

        podcasts = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                podcasts.append(line)

        return {
            "success": True,
            "podcast_count": len(podcasts),
            "podcasts": podcasts,
            "file_path": podcasts_path,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def get_videos() -> dict:
    """Get video channel configuration.

    Returns:
        Dict with video channels list and count.
    """
    try:
        videos_path = os.path.join(PROJECT_DIR, "videos.txt")

        if not os.path.exists(videos_path):
            return {
                "success": True,
                "channel_count": 0,
                "channels": [],
                "file_path": videos_path,
            }

        with open(videos_path, "r") as f:
            lines = f.readlines()

        channels = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                channels.append(line)

        return {
            "success": True,
            "channel_count": len(channels),
            "channels": channels,
            "file_path": videos_path,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def update_interests(interests: List[str]) -> dict:
    """Update semantic interests for scoring.

    Note: This creates a custom interests file but doesn't modify
    the source code. The agent would need to be modified to read
    from this file.

    Args:
        interests: List of interest descriptions.

    Returns:
        Dict with update status.
    """
    try:
        interests_path = os.path.join(PROJECT_DIR, "config", "interests.json")

        # Ensure config directory exists
        os.makedirs(os.path.dirname(interests_path), exist_ok=True)

        with open(interests_path, "w") as f:
            json.dump({
                "semantic_interests": interests,
                "updated_at": str(__import__("datetime").datetime.now()),
            }, f, indent=2)

        return {
            "success": True,
            "message": f"Saved {len(interests)} interests to {interests_path}",
            "note": "Agent code needs to be updated to read from this config file.",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def add_source(url: str, source_type: str = "news") -> dict:
    """Add a new source URL.

    Args:
        url: RSS feed URL to add.
        source_type: Type of source (news, podcast, video).

    Returns:
        Dict with update status.
    """
    try:
        file_map = {
            "news": "sources.txt",
            "podcast": "podcasts.txt",
            "video": "videos.txt",
        }

        if source_type not in file_map:
            return {
                "success": False,
                "error": f"Invalid source type: {source_type}. Must be: news, podcast, video",
            }

        file_path = os.path.join(PROJECT_DIR, file_map[source_type])

        # Read existing sources
        existing = []
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                existing = [line.strip() for line in f.readlines()]

        # Check for duplicate
        if url in existing:
            return {
                "success": False,
                "error": f"Source already exists: {url}",
            }

        # Append new source
        with open(file_path, "a") as f:
            f.write(f"\n{url}")

        return {
            "success": True,
            "message": f"Added {source_type} source: {url}",
            "file_path": file_path,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def remove_source(url: str, source_type: str = "news") -> dict:
    """Remove a source URL.

    Args:
        url: RSS feed URL to remove.
        source_type: Type of source (news, podcast, video).

    Returns:
        Dict with update status.
    """
    try:
        file_map = {
            "news": "sources.txt",
            "podcast": "podcasts.txt",
            "video": "videos.txt",
        }

        if source_type not in file_map:
            return {
                "success": False,
                "error": f"Invalid source type: {source_type}",
            }

        file_path = os.path.join(PROJECT_DIR, file_map[source_type])

        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}",
            }

        # Read and filter sources
        with open(file_path, "r") as f:
            lines = f.readlines()

        new_lines = [line for line in lines if line.strip() != url]

        if len(new_lines) == len(lines):
            return {
                "success": False,
                "error": f"Source not found: {url}",
            }

        # Write back
        with open(file_path, "w") as f:
            f.writelines(new_lines)

        return {
            "success": True,
            "message": f"Removed {source_type} source: {url}",
            "file_path": file_path,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
