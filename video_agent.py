"""
Video Agent - Scans video feeds (YouTube, etc.) for AI/GenAI insights.
Fetches transcripts and generates key learnings summaries.
"""

import os
import re
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

import feedparser
import requests
from dotenv import load_dotenv

# Try to import youtube-transcript-api
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable
    )
    YOUTUBE_TRANSCRIPT_AVAILABLE = True
    # Create a global API instance
    _youtube_api = YouTubeTranscriptApi()
except ImportError:
    YOUTUBE_TRANSCRIPT_AVAILABLE = False
    _youtube_api = None
    print("Warning: youtube-transcript-api not installed. Run: pip install youtube-transcript-api")

load_dotenv()

# Cache directory for video data
CACHE_DIR = Path("cache/videos")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Seen videos file
SEEN_FILE = Path("out/seen_videos.json")

# AI/GenAI keywords for scoring
AI_KEYWORDS = [
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "neural network", "gpt", "chatgpt", "claude", "llm", "large language model",
    "generative ai", "genai", "openai", "anthropic", "google ai", "gemini",
    "transformer", "diffusion", "stable diffusion", "midjourney", "dall-e",
    "copilot", "agent", "ai agent", "rag", "retrieval", "embedding",
    "fine-tuning", "prompt engineering", "inference", "training",
    "computer vision", "nlp", "natural language", "speech recognition",
    "text-to-speech", "text-to-image", "multimodal", "foundation model"
]


def load_seen_videos() -> set:
    """Load set of previously seen video IDs."""
    if SEEN_FILE.exists():
        try:
            with open(SEEN_FILE, "r") as f:
                data = json.load(f)
                return set(data.get("seen", []))
        except:
            pass
    return set()


def save_seen_videos(seen: set):
    """Save seen video IDs."""
    SEEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SEEN_FILE, "w") as f:
        json.dump({"seen": list(seen)}, f)


def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com/shorts/([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_youtube_channel_feed(channel_id: str) -> str:
    """Get RSS feed URL for a YouTube channel."""
    return f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"


def get_youtube_playlist_feed(playlist_id: str) -> str:
    """Get RSS feed URL for a YouTube playlist."""
    return f"https://www.youtube.com/feeds/videos.xml?playlist_id={playlist_id}"


def extract_channel_id(url: str) -> Optional[str]:
    """Extract YouTube channel ID from URL."""
    # Direct channel ID URL
    match = re.search(r'youtube\.com/channel/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)

    # Handle @username format - need to resolve to channel ID
    match = re.search(r'youtube\.com/@([a-zA-Z0-9_-]+)', url)
    if match:
        username = match.group(1)
        return resolve_youtube_handle(username)

    return None


def resolve_youtube_handle(handle: str) -> Optional[str]:
    """Resolve a YouTube @handle to a channel ID."""
    try:
        # Fetch the channel page and extract channel ID from meta tags
        url = f"https://www.youtube.com/@{handle}"
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
        response = requests.get(url, timeout=10, headers=headers)
        if response.status_code == 200:
            # Try multiple patterns to find channel ID
            patterns = [
                r'"externalId":"([a-zA-Z0-9_-]+)"',
                r'"browseId":"(UC[a-zA-Z0-9_-]+)"',
                r'"channelId":"([a-zA-Z0-9_-]+)"',
            ]
            for pattern in patterns:
                match = re.search(pattern, response.text)
                if match:
                    return match.group(1)
    except Exception as e:
        print(f"  Warning: Could not resolve @{handle}: {e}")
    return None


def fetch_youtube_transcript(video_id: str) -> Optional[str]:
    """Fetch transcript for a YouTube video."""
    if not YOUTUBE_TRANSCRIPT_AVAILABLE or _youtube_api is None:
        return None

    try:
        # Fetch transcript using the new API
        transcript = _youtube_api.fetch(video_id)

        # Combine all text segments (new API uses .text attribute)
        full_text = " ".join([entry.text for entry in transcript])
        return full_text

    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        return None
    except Exception as e:
        print(f"  Warning: Error fetching transcript for {video_id}: {e}")
        return None


def score_video(title: str, description: str = "", transcript: str = "") -> int:
    """Score a video based on AI/GenAI keyword matches."""
    text = f"{title} {description} {transcript}".lower()
    score = 0

    for keyword in AI_KEYWORDS:
        if keyword in text:
            # Higher weight for title matches
            if keyword in title.lower():
                score += 3
            else:
                score += 1

    return score


def parse_video_feeds(feed_urls: List[str], max_per_feed: int = 5, days_back: int = 7) -> List[Dict[str, Any]]:
    """Parse video RSS feeds and return recent videos."""
    videos = []
    cutoff = datetime.now() - timedelta(days=days_back)

    for feed_url in feed_urls:
        try:
            # Handle YouTube URLs - convert to RSS feed
            if "youtube.com/@" in feed_url:
                channel_id = extract_channel_id(feed_url)
                if channel_id:
                    feed_url = get_youtube_channel_feed(channel_id)
                else:
                    print(f"  ✗ Could not resolve: {feed_url}")
                    continue
            elif "youtube.com/channel/" in feed_url:
                channel_id = extract_channel_id(feed_url)
                if channel_id:
                    feed_url = get_youtube_channel_feed(channel_id)

            feed = feedparser.parse(feed_url)

            if feed.bozo and not feed.entries:
                print(f"  ✗ Failed to parse: {feed_url}")
                continue

            feed_title = feed.feed.get('title', 'Unknown Channel')
            count = 0

            for entry in feed.entries[:max_per_feed]:
                # Parse published date
                pub_date = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6])
                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                    pub_date = datetime(*entry.updated_parsed[:6])

                # Skip old videos
                if pub_date and pub_date < cutoff:
                    continue

                video_id = extract_video_id(entry.get('link', ''))

                videos.append({
                    'title': entry.get('title', 'Unknown'),
                    'link': entry.get('link', ''),
                    'video_id': video_id,
                    'published': pub_date,
                    'channel': feed_title,
                    'description': entry.get('summary', ''),
                    'feed_url': feed_url
                })
                count += 1

            if count > 0:
                print(f"  ✓ {feed_url}: {count} videos")

        except Exception as e:
            print(f"  ✗ Error parsing {feed_url}: {e}")

    return videos


def process_videos(videos: List[Dict[str, Any]], max_videos: int = 10) -> List[Dict[str, Any]]:
    """Process videos: fetch transcripts, score, and filter top AI-related videos."""
    from summarizer import summarize_podcast, generate_fallback_podcast_summary

    seen = load_seen_videos()
    processed = []

    print(f"📹 Processing {len(videos)} videos for AI content...")

    for video in videos:
        video_id = video.get('video_id')

        # Skip if no video ID or already seen
        if not video_id:
            continue
        if video_id in seen:
            continue

        # Initial score based on title/description
        initial_score = score_video(video['title'], video.get('description', ''))

        # If title looks relevant, try to get transcript
        transcript = None
        if initial_score > 0 or any(kw in video['title'].lower() for kw in ['ai', 'gpt', 'llm', 'machine learning']):
            print(f"  🔍 Fetching transcript: {video['title'][:50]}...")
            transcript = fetch_youtube_transcript(video_id)

        # Re-score with transcript
        final_score = score_video(
            video['title'],
            video.get('description', ''),
            transcript or ''
        )

        video['transcript'] = transcript
        video['score'] = final_score
        video['hash'] = hashlib.sha256(video['link'].encode()).hexdigest()[:24]

        # Mark as seen
        seen.add(video_id)

        if final_score > 0:
            processed.append(video)

    # Save seen videos
    save_seen_videos(seen)

    # Sort by score and return top videos
    processed.sort(key=lambda x: x['score'], reverse=True)
    return processed[:max_videos]


def summarize_videos(videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate AI summaries for videos with transcripts."""
    from summarizer import summarize_podcast, generate_fallback_podcast_summary

    for video in videos:
        if video.get('transcript'):
            print(f"  📝 Summarizing: {video['title'][:50]}...")
            summary = summarize_podcast(
                video['transcript'],
                title=video['title'],
                show_name=video.get('channel', '')
            )
            if summary:
                video['summary'] = summary
            else:
                video['summary'] = generate_fallback_podcast_summary(
                    title=video['title'],
                    show_name=video.get('channel', '')
                )
        else:
            video['summary'] = f"Watch '{video['title']}' from {video.get('channel', 'this channel')} for AI insights."

    return videos


def load_video_feeds(filepath: str = "videos.txt") -> List[str]:
    """Load video feed URLs from file."""
    feeds = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    feeds.append(line)
    except FileNotFoundError:
        print(f"Warning: {filepath} not found")
    return feeds


def run_video_agent(max_videos: int = 10, days_back: int = 7) -> List[Dict[str, Any]]:
    """Main entry point for video agent."""
    print("📺 Starting Video Agent...")

    # Load feeds
    feeds = load_video_feeds()
    if not feeds:
        print("No video feeds configured in videos.txt")
        return []

    print(f"📡 Processing {len(feeds)} video feeds...")

    # Parse feeds
    videos = parse_video_feeds(feeds, max_per_feed=5, days_back=days_back)
    print(f"📋 Total videos fetched: {len(videos)}")

    # Process and score
    relevant_videos = process_videos(videos, max_videos=max_videos)
    print(f"🎯 AI-relevant videos: {len(relevant_videos)}")

    # Generate summaries
    if relevant_videos:
        summarized = summarize_videos(relevant_videos)
        return summarized

    return []


if __name__ == "__main__":
    videos = run_video_agent()
    for v in videos:
        print(f"\n{'='*60}")
        print(f"Title: {v['title']}")
        print(f"Channel: {v.get('channel', 'Unknown')}")
        print(f"Score: {v['score']}")
        print(f"Link: {v['link']}")
        if v.get('summary'):
            print(f"\nSummary:\n{v['summary']}")
