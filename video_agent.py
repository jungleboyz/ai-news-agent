"""
Video Agent - Scans video feeds (YouTube, etc.) for AI/GenAI insights.
Fetches transcripts and generates key learnings summaries.
"""

import os
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from pathlib import Path

import feedparser
import requests

from services.youtube_service import (
    youtube_api,
    YOUTUBE_TRANSCRIPT_AVAILABLE,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from services.service_registry import get_vector_store
from services.scoring_service import (
    USE_SEMANTIC_SCORING,
    score_keywords,
    score_semantic,
    score_single_with_embedding,
)
from services.cache_service import load_set, save_set
from services.feed_service import fetch_feeds_parallel, update_feed_statuses

# Cache directory for video data
CACHE_DIR = Path("cache/videos")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Seen videos file
SEEN_FILE = Path("out/seen_videos.json")


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
    if not YOUTUBE_TRANSCRIPT_AVAILABLE or youtube_api is None:
        return None

    try:
        transcript = youtube_api.fetch(video_id)
        full_text = " ".join([entry.text for entry in transcript])
        return full_text

    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return None
    except Exception as e:
        print(f"  Warning: Error fetching transcript for {video_id}: {e}")
        return None


def _fetch_transcript_for_video(video: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch transcript for a single video (used in parallel execution)."""
    video_id = video.get("video_id")
    if not video_id:
        return video

    transcript = fetch_youtube_transcript(video_id)
    if not transcript:
        try:
            from services.transcript_service import TranscriptService
            ts = TranscriptService()
            transcript, _source = ts.get_transcript(
                title=video['title'],
                link=video['link'],
                video_id=video_id,
            )
            if _source == "description":
                transcript = None
        except Exception:
            pass

    video['transcript'] = transcript
    return video


def store_video_embeddings(videos: List[Dict[str, Any]]) -> None:
    """Store video embeddings in ChromaDB."""
    if not USE_SEMANTIC_SCORING:
        return

    try:
        vector_store = get_vector_store()

        items_to_store = []
        for video in videos:
            if video.get("embedding"):
                items_to_store.append({
                    "id": video.get("hash") or video.get("video_id", ""),
                    "text": f"{video['title']} {video.get('description', '')}".strip(),
                    "metadata": {
                        "title": video["title"],
                        "link": video["link"],
                        "channel": video.get("channel", ""),
                        "semantic_score": video.get("semantic_score"),
                    },
                    "embedding": video["embedding"],
                })

        if items_to_store:
            vector_store.add_items_batch(items_to_store, "video")
            print(f"  Stored {len(items_to_store)} video embeddings in ChromaDB")

    except Exception as e:
        print(f"  Failed to store video embeddings: {e}")


def _parse_single_feed(feed_url: str, max_per_feed: int = 5, days_back: int = 7) -> List[Dict[str, Any]]:
    """Parse a single video RSS feed and return recent videos."""
    videos = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

    try:
        actual_url = feed_url
        # Handle YouTube URLs - convert to RSS feed
        if "youtube.com/@" in feed_url:
            channel_id = extract_channel_id(feed_url)
            if channel_id:
                actual_url = get_youtube_channel_feed(channel_id)
            else:
                print(f"  Could not resolve: {feed_url}")
                return []
        elif "youtube.com/channel/" in feed_url:
            channel_id = extract_channel_id(feed_url)
            if channel_id:
                actual_url = get_youtube_channel_feed(channel_id)

        feed = feedparser.parse(actual_url)

        if feed.bozo and not feed.entries:
            print(f"  Failed to parse: {feed_url}")
            return []

        feed_title = feed.feed.get('title', 'Unknown Channel')
        count = 0

        for entry in feed.entries[:max_per_feed]:
            pub_date = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                pub_date = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)

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
            print(f"  {feed_url}: {count} videos")

    except Exception as e:
        print(f"  Error parsing {feed_url}: {e}")

    return videos


def parse_video_feeds(feed_urls: List[str], max_per_feed: int = 5, days_back: int = 7) -> List[Dict[str, Any]]:
    """Parse video RSS feeds in parallel and return recent videos."""
    def _parse(url):
        return _parse_single_feed(url, max_per_feed=max_per_feed, days_back=days_back)

    results = fetch_feeds_parallel(feed_urls, _parse, max_workers=10)
    videos = []
    for url in feed_urls:
        videos.extend(results.get(url, []))
    return videos


def process_videos(videos: List[Dict[str, Any]], max_videos: int = 10) -> List[Dict[str, Any]]:
    """Process videos: fetch transcripts, score, and filter top AI-related videos."""
    from summarizer import summarize_podcast, generate_fallback_podcast_summary

    seen = load_set(str(SEEN_FILE))

    # Merge DB-backed seen video IDs to survive container restarts
    try:
        from web.db_writer import get_seen_links_from_db
        db_links = get_seen_links_from_db(days=30)
        for link in db_links:
            m = re.search(r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})', link)
            if m:
                seen.add(m.group(1))
        print(f"Video seen cache: {len(seen)} total (including DB)")
    except Exception as e:
        print(f"  DB dedup check skipped: {e}")

    processed = []

    print(f"Processing {len(videos)} videos for AI content...")
    if USE_SEMANTIC_SCORING:
        print("Using semantic scoring for video relevance...")

    # Filter out seen and no-ID videos first
    candidates = []
    for video in videos:
        video_id = video.get('video_id')
        if not video_id or video_id in seen:
            continue
        candidates.append(video)

    # Initial score based on title/description to decide which need transcripts
    need_transcript = []
    for video in candidates:
        initial_score = score_semantic(video['title'], video.get('description', ''))
        if initial_score > 0 or any(kw in video['title'].lower() for kw in ['ai', 'gpt', 'llm', 'machine learning']):
            need_transcript.append(video)
        else:
            video['transcript'] = None
            need_transcript.append(video)  # Still process, just skip transcript fetch

    # Fetch transcripts in parallel for candidates that need them
    videos_needing_fetch = [v for v in need_transcript
                           if score_semantic(v['title'], v.get('description', '')) > 0
                           or any(kw in v['title'].lower() for kw in ['ai', 'gpt', 'llm', 'machine learning'])]

    if videos_needing_fetch:
        print(f"  Fetching transcripts for {len(videos_needing_fetch)} videos in parallel...")
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(_fetch_transcript_for_video, v): v for v in videos_needing_fetch}
            for future in as_completed(futures):
                try:
                    future.result(timeout=30)
                except Exception as e:
                    video = futures[future]
                    print(f"  Transcript fetch failed for {video['title'][:40]}: {e}")
                    video['transcript'] = None

    # Score all candidates with transcripts
    for video in candidates:
        video['hash'] = hashlib.sha256(video['link'].encode()).hexdigest()[:24]
        video['id'] = video['hash']

        int_score, semantic_score, embedding = score_single_with_embedding(
            video['title'], video.get('description', ''), video.get('transcript') or ''
        )
        video['score'] = int_score
        video['semantic_score'] = semantic_score
        video['embedding'] = embedding

        seen.add(video['video_id'])

        if video['score'] > 0:
            processed.append(video)

    save_set(str(SEEN_FILE), seen)

    # Sort by score and return top videos
    processed.sort(key=lambda x: x['score'], reverse=True)
    top_videos = processed[:max_videos]

    # Store embeddings for top videos
    if USE_SEMANTIC_SCORING and top_videos:
        store_video_embeddings(top_videos)

    return top_videos


def summarize_videos(videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate AI summaries for videos with transcripts."""
    from summarizer import summarize_podcast, generate_fallback_podcast_summary

    for video in videos:
        if video.get('transcript'):
            print(f"  Summarizing: {video['title'][:50]}...")
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
    """Load video feed URLs from DB (active FeedSources), falling back to videos.txt."""
    try:
        from web.database import SessionLocal
        from web.models import FeedSource
        with SessionLocal() as session:
            urls = session.query(FeedSource.feed_url).filter(
                FeedSource.source_type == "video",
                FeedSource.status == "active",
            ).all()
            if urls:
                return [u[0] for u in urls]
    except Exception:
        pass  # DB not available, fall back to file

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
    print("Starting Video Agent...")

    # Load feeds
    feeds = load_video_feeds()
    if not feeds:
        print("No video feeds configured in videos.txt")
        return []

    print(f"Processing {len(feeds)} video feeds...")

    # Parse feeds (parallel)
    videos = parse_video_feeds(feeds, max_per_feed=5, days_back=days_back)
    print(f"Total videos fetched: {len(videos)}")

    # Process and score
    relevant_videos = process_videos(videos, max_videos=max_videos)
    print(f"AI-relevant videos: {len(relevant_videos)}")

    # Generate summaries
    if relevant_videos:
        summarized = summarize_videos(relevant_videos)
        update_feed_statuses("video", feeds)
        return summarized

    update_feed_statuses("video", feeds)
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
