"""Cascading transcript service — eliminates Whisper entirely.

Cascade order:
1. Web transcript scraping (episode page via Firecrawl/BS4)
2. YouTube transcript (youtube-transcript-api)
3. RSS description fallback (always available)
"""
import re
from typing import Optional

import requests
from bs4 import BeautifulSoup

from services.firecrawl_service import get_firecrawl_service

# Try to import youtube-transcript-api
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
    )
    YOUTUBE_TRANSCRIPT_AVAILABLE = True
    _youtube_api = YouTubeTranscriptApi()
except ImportError:
    YOUTUBE_TRANSCRIPT_AVAILABLE = False
    _youtube_api = None

# Markers that indicate a transcript section on a web page
TRANSCRIPT_MARKERS = [
    "transcript", "full text", "show notes", "episode transcript",
    "read the transcript", "full transcript",
]

# Minimum length to consider scraped text as a usable transcript
MIN_TRANSCRIPT_LENGTH = 500


class TranscriptService:
    """Cascading transcript fetcher: web → YouTube → description."""

    def get_transcript(
        self,
        title: str,
        link: str,
        audio_url: Optional[str] = None,
        video_id: Optional[str] = None,
        description: Optional[str] = None,
    ) -> tuple[str, str]:
        """Try each source in order and return (text, source).

        Args:
            title: Episode/video title.
            link: Episode/video web page URL.
            audio_url: Podcast audio URL (unused, kept for interface compat).
            video_id: YouTube video ID if known.
            description: RSS description/summary text.

        Returns:
            Tuple of (transcript_text, source) where source is
            'web', 'youtube', or 'description'.
        """
        # 1. Try web transcript scraping
        web_text = self._try_web_transcript(link)
        if web_text:
            return web_text, "web"

        # 2. Try YouTube transcript
        yt_text = self._try_youtube_transcript(title, video_id=video_id)
        if yt_text:
            return yt_text, "youtube"

        # 3. Fall back to RSS description
        return self._use_description(description or ""), "description"

    def _try_web_transcript(self, link: str) -> Optional[str]:
        """Scrape the episode page for transcript content."""
        if not link:
            return None

        # Try Firecrawl first
        fc = get_firecrawl_service()
        if fc.available:
            markdown = fc.scrape_article(link)
            if markdown and self._looks_like_transcript(markdown):
                return markdown

        # Fall back to BS4
        try:
            html = requests.get(link, timeout=10, headers={
                "User-Agent": "Mozilla/5.0 (compatible; AINewsAgent/1.0)"
            }).text
            soup = BeautifulSoup(html, "html.parser")

            # Look for transcript-specific sections
            for marker in TRANSCRIPT_MARKERS:
                # Check headings
                for tag in soup.find_all(["h1", "h2", "h3", "h4", "div", "section"]):
                    if marker in (tag.get_text() or "").lower():
                        # Grab the sibling/child text
                        parent = tag.parent or tag
                        text = parent.get_text(separator=" ", strip=True)
                        if len(text) >= MIN_TRANSCRIPT_LENGTH:
                            return text[:12000]

            # If no transcript markers, check for large text blocks
            paragraphs = soup.find_all("p")
            full_text = " ".join(p.get_text(strip=True) for p in paragraphs)
            if len(full_text) >= MIN_TRANSCRIPT_LENGTH * 2:
                return full_text[:12000]

        except Exception:
            pass

        return None

    def _looks_like_transcript(self, text: str) -> bool:
        """Heuristic: does scraped text look like a transcript?"""
        if len(text) < MIN_TRANSCRIPT_LENGTH:
            return False
        # Transcripts tend to have many sentences / dialogue markers
        sentence_count = len(re.findall(r'[.!?]\s', text))
        return sentence_count >= 10

    def _try_youtube_transcript(
        self, title: str, video_id: Optional[str] = None
    ) -> Optional[str]:
        """Fetch YouTube transcript if video_id is available."""
        if not YOUTUBE_TRANSCRIPT_AVAILABLE or _youtube_api is None:
            return None
        if not video_id:
            # Try to extract video ID from title search (simple heuristic)
            video_id = self._search_youtube_id(title)
        if not video_id:
            return None

        try:
            transcript = _youtube_api.fetch(video_id)
            full_text = " ".join(entry.text for entry in transcript)
            return full_text[:12000] if full_text else None
        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
            return None
        except Exception:
            return None

    def _search_youtube_id(self, title: str) -> Optional[str]:
        """Try to find a YouTube video ID by searching for the episode title.

        Uses a simple YouTube search scrape — no API key needed.
        """
        try:
            query = f"{title} full episode"
            url = f"https://www.youtube.com/results?search_query={requests.utils.quote(query)}"
            headers = {"User-Agent": "Mozilla/5.0 (compatible; AINewsAgent/1.0)"}
            resp = requests.get(url, timeout=10, headers=headers)
            # Extract first video ID from search results page
            match = re.search(r'"videoId":"([a-zA-Z0-9_-]{11})"', resp.text)
            if match:
                return match.group(1)
        except Exception:
            pass
        return None

    def _use_description(self, description: str) -> str:
        """Return RSS description as-is (always available)."""
        # Clean HTML tags from description
        if "<" in description:
            soup = BeautifulSoup(description, "html.parser")
            return soup.get_text(separator=" ", strip=True)
        return description.strip()
