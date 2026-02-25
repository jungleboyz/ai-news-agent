"""Shared YouTube Transcript API instance with optional Webshare proxy.

Both video_agent.py and services/transcript_service.py import from here
so the proxy is configured exactly once.
"""
from typing import Optional

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
    )
    from youtube_transcript_api.proxies import WebshareProxyConfig

    YOUTUBE_TRANSCRIPT_AVAILABLE = True

    from config import settings

    if settings.webshare_proxy_username and settings.webshare_proxy_password:
        youtube_api: Optional[YouTubeTranscriptApi] = YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username=settings.webshare_proxy_username,
                proxy_password=settings.webshare_proxy_password,
            )
        )
        print("YouTube transcript API: using Webshare proxy")
    else:
        youtube_api = YouTubeTranscriptApi()

except ImportError:
    YOUTUBE_TRANSCRIPT_AVAILABLE = False
    youtube_api = None

    # Provide stub error classes so callers can still reference them
    class TranscriptsDisabled(Exception):  # type: ignore[no-redef]
        pass

    class NoTranscriptFound(Exception):  # type: ignore[no-redef]
        pass

    class VideoUnavailable(Exception):  # type: ignore[no-redef]
        pass

    print("Warning: youtube-transcript-api not installed. Run: pip install youtube-transcript-api")
