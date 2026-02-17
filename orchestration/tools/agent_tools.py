"""Agent execution tools for MCP."""

import os
import sys
from typing import Optional

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_news_digest() -> dict:
    """Run the news agent and return results.

    Returns:
        Dict with digest_path and item summaries.
    """
    from agent import run_agent

    try:
        digest_path = run_agent()
        return {
            "success": True,
            "digest_path": digest_path,
            "message": f"News digest generated at {digest_path}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def run_podcast_digest(skip_transcription: bool = False) -> dict:
    """Run the podcast agent and return results.

    Args:
        skip_transcription: Skip audio transcription for faster processing.

    Returns:
        Dict with episode list and summaries.
    """
    from podcast_agent import run_podcast_agent

    try:
        episodes = run_podcast_agent(skip_transcription=skip_transcription)
        return {
            "success": True,
            "episode_count": len(episodes),
            "episodes": [
                {
                    "title": ep["title"],
                    "show_name": ep.get("show_name", "Unknown"),
                    "score": ep["score"],
                    "semantic_score": ep.get("semantic_score"),
                    "link": ep["link"],
                    "summary": ep.get("summary", "")[:500],
                }
                for ep in episodes
            ],
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def run_video_digest(max_videos: int = 10, days_back: int = 7) -> dict:
    """Run the video agent and return results.

    Args:
        max_videos: Maximum number of videos to return.
        days_back: How many days back to look.

    Returns:
        Dict with video list and summaries.
    """
    from video_agent import run_video_agent

    try:
        videos = run_video_agent(max_videos=max_videos, days_back=days_back)
        return {
            "success": True,
            "video_count": len(videos),
            "videos": [
                {
                    "title": v["title"],
                    "channel": v.get("channel", "Unknown"),
                    "score": v["score"],
                    "semantic_score": v.get("semantic_score"),
                    "link": v["link"],
                    "summary": v.get("summary", "")[:500],
                }
                for v in videos
            ],
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def run_full_digest(skip_transcription: bool = False) -> dict:
    """Run all agents and generate complete digest.

    Args:
        skip_transcription: Skip audio transcription for faster processing.

    Returns:
        Dict with all results combined.
    """
    # Set environment variable for podcast agent
    if skip_transcription:
        os.environ["SKIP_PODCAST_TRANSCRIPTION"] = "1"
    else:
        os.environ.pop("SKIP_PODCAST_TRANSCRIPTION", None)

    results = {
        "success": True,
        "news": None,
        "podcasts": None,
        "videos": None,
    }

    # Run news agent (this also triggers podcast and video agents internally)
    news_result = run_news_digest()
    results["news"] = news_result

    if not news_result.get("success"):
        results["success"] = False

    return results
