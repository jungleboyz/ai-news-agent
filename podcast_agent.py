import os
import re
from typing import List, Dict, Optional
import feedparser
import time
import hashlib
from datetime import datetime, timezone

from services.service_registry import get_vector_store
from services.scoring_service import (
    USE_SEMANTIC_SCORING,
    score_semantic,
    score_single_with_embedding,
    score_keywords,
)
from services.cache_service import load_json, save_json
from services.feed_service import fetch_feeds_parallel, update_feed_statuses

# --- Config ---
PODCASTS_FILE = "podcasts.txt"
OUT_DIR = "out"
MAX_EPISODES_PER_FEED = 5
TRANSCRIBE_MINUTES = 15
DIGEST_TOP_PODCASTS = 5
PODCAST_SEEN_PATH = os.path.join(OUT_DIR, "podcast_seen.json")
PODCAST_TRANSCRIPTS_PATH = os.path.join(OUT_DIR, "podcast_transcripts.json")
PODCAST_SUMMARIES_PATH = os.path.join(OUT_DIR, "podcast_summaries.json")


def ensure_out_dir() -> None:
    """Create the output folder if it doesn't exist."""
    os.makedirs(OUT_DIR, exist_ok=True)


def load_podcast_feeds(path: str = PODCASTS_FILE) -> List[str]:
    """Load podcast feed URLs from DB (active FeedSources), falling back to podcasts.txt."""
    try:
        from web.database import SessionLocal
        from web.models import FeedSource
        with SessionLocal() as session:
            urls = session.query(FeedSource.feed_url).filter(
                FeedSource.source_type == "podcast",
                FeedSource.status == "active",
            ).all()
            if urls:
                return [u[0] for u in urls]
    except Exception:
        pass  # DB not available, fall back to file

    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]


def store_podcast_embeddings(episodes: List[dict]) -> None:
    """Store podcast episode embeddings in ChromaDB."""
    if not USE_SEMANTIC_SCORING:
        return

    try:
        vector_store = get_vector_store()

        items_to_store = []
        for ep in episodes:
            if ep.get("embedding"):
                items_to_store.append({
                    "id": ep["id"],
                    "text": f"{ep['title']} {ep.get('description', '')}".strip(),
                    "metadata": {
                        "title": ep["title"],
                        "link": ep["link"],
                        "show_name": ep.get("show_name", ""),
                        "semantic_score": ep.get("semantic_score"),
                    },
                    "embedding": ep["embedding"],
                })

        if items_to_store:
            vector_store.add_items_batch(items_to_store, "podcast")
            print(f"  Stored {len(items_to_store)} podcast embeddings in ChromaDB")

    except Exception as e:
        print(f"  Failed to store podcast embeddings: {e}")


def fetch_podcast_episodes(feed_url: str, limit: int = MAX_EPISODES_PER_FEED) -> List[dict]:
    """
    Fetch podcast episodes from a single RSS feed URL.
    Returns a list of dicts with title, link, description, and audio_url.
    """
    try:
        feed = feedparser.parse(feed_url)

        # Check for parsing errors
        if feed.get("bozo") and feed.get("bozo_exception"):
            print(f"  Feed parsing warning for {feed_url}: {feed.bozo_exception}")

        # Get podcast/show name
        show_name = getattr(feed.feed, "title", "") or ""

        episodes = []
        for entry in feed.entries[:limit]:
            title = getattr(entry, "title", "").strip()
            link = getattr(entry, "link", "").strip()
            description = getattr(entry, "summary", "") or getattr(entry, "description", "")
            description = re.sub(r"\s+", " ", description).strip()

            # Get audio URL from enclosure
            audio_url = None
            if hasattr(entry, "enclosures") and entry.enclosures:
                for enc in entry.enclosures:
                    if enc.get("type", "").startswith("audio"):
                        audio_url = enc.get("href", "").strip()
                        break

            if title and link:
                episodes.append({
                    "title": title,
                    "link": link,
                    "description": description,
                    "audio_url": audio_url,
                    "show_name": show_name,
                    "feed_url": feed_url
                })

        return episodes
    except Exception as e:
        print(f"  Failed to fetch podcast feed {feed_url}: {e}")
        import traceback
        traceback.print_exc()
        return []


def make_id(title: str, link: str) -> str:
    """Generate a stable ID for an episode based on its title and link."""
    raw = f"{title}|{link}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def run_podcast_agent(skip_transcription: bool = None) -> List[dict]:
    """
    Main function to process podcast feeds and return scored episodes.
    Returns a list of picked episodes with transcripts and summaries.

    Args:
        skip_transcription: Deprecated — transcription now uses the smart cascade
                          (web scrape -> YouTube -> RSS description). No audio download.
    """
    from services.transcript_service import TranscriptService
    from summarizer import summarize_podcast, generate_fallback_podcast_summary
    from services.service_registry import is_quota_exceeded

    ts = TranscriptService()

    ensure_out_dir()
    seen = load_json(PODCAST_SEEN_PATH, {})
    transcripts_cache = load_json(PODCAST_TRANSCRIPTS_PATH, {})
    summaries_cache = load_json(PODCAST_SUMMARIES_PATH, {})
    now = time.time()

    # Merge DB-backed seen hashes to survive container restarts
    try:
        from web.db_writer import get_seen_hashes_from_db
        db_hashes = get_seen_hashes_from_db(days=30, item_type="podcast")
        if db_hashes:
            merged = 0
            for h in db_hashes:
                if h not in seen:
                    seen[h] = now
                    merged += 1
            if merged:
                print(f"Loaded {merged} previously seen podcast hashes from database")
    except Exception as e:
        print(f"  DB dedup check skipped: {e}")

    feeds = load_podcast_feeds()
    if not feeds:
        print("No podcast feeds found in podcasts.txt")
        return []

    print(f"Processing {len(feeds)} podcast feeds...")

    # Fetch episodes from all feeds in parallel
    def _fetch(url):
        eps = fetch_podcast_episodes(url, limit=MAX_EPISODES_PER_FEED)
        for ep in eps:
            ep["id"] = make_id(ep["title"], ep["link"])
        return eps

    feed_results = fetch_feeds_parallel(feeds, _fetch, max_workers=10)

    all_episodes = []
    for feed_url in feeds:
        eps = feed_results.get(feed_url, [])
        if eps:
            print(f"  {feed_url}: {len(eps)} episodes")
        all_episodes.extend(eps)

    print(f"Total episodes fetched: {len(all_episodes)}")

    # Remove already-seen episodes
    fresh = [ep for ep in all_episodes if ep["id"] not in seen]
    print(f"Fresh episodes: {len(fresh)} (seen: {len(all_episodes) - len(fresh)})")

    # If no fresh episodes, optionally include recently seen ones (within last 7 days)
    if len(fresh) == 0 and len(all_episodes) > 0:
        seven_days_ago = now - (7 * 24 * 60 * 60)
        recent_seen = [
            ep for ep in all_episodes
            if ep["id"] in seen and seen[ep["id"]] >= seven_days_ago
        ]
        if recent_seen:
            print(f"Including {len(recent_seen)} recently seen episodes (last 7 days)")
            fresh = recent_seen

    # Initial score episodes (will update after transcription)
    total = len(fresh)
    if USE_SEMANTIC_SCORING and not is_quota_exceeded():
        print(f"Generating initial embeddings and semantic scores for {total} episodes...")
    else:
        print(f"Scoring {total} episodes with keyword matching...")
    for i, ep in enumerate(fresh):
        ep["score"] = score_semantic(ep["title"], ep.get("description", ""), "")
        if (i + 1) % 50 == 0:
            print(f"    Progress: {i + 1}/{total} episodes scored")

    # Sort by score desc
    fresh.sort(key=lambda x: x["score"], reverse=True)

    # Process top episodes: transcribe and summarize
    picked = []
    transcripts_updated = False
    summaries_updated = False

    for ep in fresh:
        if len(picked) >= DIGEST_TOP_PODCASTS:
            break

        # Use episode link or audio_url as cache key
        cache_key = ep.get("audio_url") or ep["link"]

        # Get or generate transcript via smart cascade
        if cache_key in transcripts_cache:
            transcript = transcripts_cache[cache_key]
            transcript_source = "cache"
        else:
            transcript, transcript_source = ts.get_transcript(
                title=ep["title"],
                link=ep["link"],
                audio_url=ep.get("audio_url"),
                description=ep.get("description", ""),
            )
            if transcript:
                transcripts_cache[cache_key] = transcript
                transcripts_updated = True

        print(f"  {ep['title'][:50]}... [transcript: {transcript_source}]")

        ep["transcript"] = transcript

        # Re-score with transcript (using semantic scoring if enabled)
        int_score, semantic_score, embedding = score_single_with_embedding(
            ep["title"], ep.get("description", ""), transcript or ""
        )
        ep["score"] = int_score
        ep["semantic_score"] = semantic_score
        ep["embedding"] = embedding

        # Get or generate summary with 5 key learnings
        if cache_key in summaries_cache:
            summary = summaries_cache[cache_key]
        else:
            # Summarize using transcript or description
            summary_text = transcript if transcript else ep.get("description", "")
            if summary_text:
                summary = summarize_podcast(summary_text, ep["title"], ep.get("show_name", ""))
                if summary is None:
                    summary = generate_fallback_podcast_summary(ep["title"], ep.get("show_name", ""))
            else:
                summary = generate_fallback_podcast_summary(ep["title"], ep.get("show_name", ""))
            summaries_cache[cache_key] = summary
            summaries_updated = True

        ep["summary"] = summary

        picked.append(ep)
        seen[ep["id"]] = now

    # Save caches
    save_json(PODCAST_SEEN_PATH, seen)
    if transcripts_updated:
        save_json(PODCAST_TRANSCRIPTS_PATH, transcripts_cache)
    if summaries_updated:
        save_json(PODCAST_SUMMARIES_PATH, summaries_cache)

    # Store embeddings in ChromaDB
    if USE_SEMANTIC_SCORING and picked:
        store_podcast_embeddings(picked)

    # Update feed source statuses in DB
    update_feed_statuses("podcast", feeds)

    return picked
