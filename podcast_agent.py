import os
import re
from typing import List, Dict, Optional
import feedparser
import json
import time
import hashlib
from datetime import datetime, timezone

# Import shared scoring from agent.py
# We'll import this from agent module when integrating
USER_INTERESTS = [
    "genai", "generative ai", "llm", "agent", "agents",
    "openai", "anthropic", "gemini", "mistral", "claude",
    "cursor", "copilot", "aider", "enterprise", "bank",
    "marketing", "automation", "workflow", "funding", "acquisition"
]

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
    """Read podcast RSS feed URLs from podcasts.txt (one URL per line)."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]


def norm(text: str) -> str:
    """Lowercase + collapse whitespace so matching is consistent."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def score_item(title: str, description: str = "", transcript: str = "") -> int:
    """Score podcast episode based on keyword matching."""
    text = norm(f"{title} {description} {transcript}")
    score = 0
    for kw in USER_INTERESTS:
        if kw in text:
            score += 2
    return score


def fetch_podcast_episodes(feed_url: str, limit: int = MAX_EPISODES_PER_FEED) -> List[dict]:
    """
    Fetch podcast episodes from a single RSS feed URL.
    Returns a list of dicts with title, link, description, and audio_url.
    """
    try:
        feed = feedparser.parse(feed_url)
        
        # Check for parsing errors
        if feed.get("bozo") and feed.get("bozo_exception"):
            print(f"  ⚠ Feed parsing warning for {feed_url}: {feed.bozo_exception}")
        
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
        print(f"  ⚠ Failed to fetch podcast feed {feed_url}: {e}")
        import traceback
        traceback.print_exc()
        return []


def make_id(title: str, link: str) -> str:
    """Generate a stable ID for an episode based on its title and link."""
    raw = f"{title}|{link}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def load_podcast_seen() -> Dict[str, float]:
    """Load seen podcast episodes."""
    if not os.path.exists(PODCAST_SEEN_PATH):
        return {}
    with open(PODCAST_SEEN_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_podcast_seen(seen: Dict[str, float]) -> None:
    """Save seen podcast episodes."""
    with open(PODCAST_SEEN_PATH, "w", encoding="utf-8") as f:
        json.dump(seen, f, indent=2)


def load_podcast_transcripts() -> Dict[str, str]:
    """Load cached transcripts by episode audio URL."""
    if not os.path.exists(PODCAST_TRANSCRIPTS_PATH):
        return {}
    with open(PODCAST_TRANSCRIPTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_podcast_transcripts(transcripts: Dict[str, str]) -> None:
    """Save cached transcripts by episode audio URL."""
    with open(PODCAST_TRANSCRIPTS_PATH, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, indent=2)


def load_podcast_summaries() -> Dict[str, str]:
    """Load cached summaries by episode audio URL."""
    if not os.path.exists(PODCAST_SUMMARIES_PATH):
        return {}
    with open(PODCAST_SUMMARIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_podcast_summaries(summaries: Dict[str, str]) -> None:
    """Save cached summaries by episode audio URL."""
    with open(PODCAST_SUMMARIES_PATH, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)


def run_podcast_agent() -> List[dict]:
    """
    Main function to process podcast feeds and return scored episodes.
    Returns a list of picked episodes with transcripts and summaries.
    """
    from transcriber import transcribe_episode
    from summarizer import summarize_podcast, generate_fallback_podcast_summary
    
    ensure_out_dir()
    seen = load_podcast_seen()
    transcripts_cache = load_podcast_transcripts()
    summaries_cache = load_podcast_summaries()
    now = time.time()
    
    feeds = load_podcast_feeds()
    if not feeds:
        print("⚠ No podcast feeds found in podcasts.txt")
        return []
    
    print(f"📡 Processing {len(feeds)} podcast feeds...")
    all_episodes = []
    for feed_url in feeds:
        episodes = fetch_podcast_episodes(feed_url, limit=MAX_EPISODES_PER_FEED)
        if episodes:
            print(f"  ✓ {feed_url}: {len(episodes)} episodes")
        for ep in episodes:
            ep["id"] = make_id(ep["title"], ep["link"])
            all_episodes.append(ep)
    
    print(f"📋 Total episodes fetched: {len(all_episodes)}")
    
    # Remove already-seen episodes
    fresh = [ep for ep in all_episodes if ep["id"] not in seen]
    print(f"🆕 Fresh episodes: {len(fresh)} (seen: {len(all_episodes) - len(fresh)})")
    
    # If no fresh episodes, optionally include recently seen ones (within last 7 days)
    if len(fresh) == 0 and len(all_episodes) > 0:
        seven_days_ago = now - (7 * 24 * 60 * 60)
        recent_seen = [
            ep for ep in all_episodes 
            if ep["id"] in seen and seen[ep["id"]] >= seven_days_ago
        ]
        if recent_seen:
            print(f"📅 Including {len(recent_seen)} recently seen episodes (last 7 days)")
            fresh = recent_seen
    
    # Score episodes (will update after transcription)
    for ep in fresh:
        ep["score"] = score_item(ep["title"], ep.get("description", ""), "")
    
    # Sort by score desc
    fresh.sort(key=lambda x: x["score"], reverse=True)
    
    # Process top episodes: transcribe and summarize
    picked = []
    transcripts_updated = False
    summaries_updated = False
    
    for ep in fresh:
        if len(picked) >= DIGEST_TOP_PODCASTS:
            break
        
        # Skip if no audio URL
        if not ep.get("audio_url"):
            print(f"  ⚠ Skipping '{ep['title'][:50]}...' - no audio URL")
            continue
        
        audio_url = ep["audio_url"]
        
        # Get or generate transcript
        if audio_url in transcripts_cache:
            transcript = transcripts_cache[audio_url]
        else:
            transcript = transcribe_episode(audio_url, minutes=TRANSCRIBE_MINUTES)
            if transcript:
                transcripts_cache[audio_url] = transcript
                transcripts_updated = True
            else:
                # Use description as fallback if transcription fails
                transcript = ep.get("description", "")
        
        ep["transcript"] = transcript
        
        # Re-score with transcript
        ep["score"] = score_item(ep["title"], ep.get("description", ""), transcript)
        
        # Get or generate summary with 5 key learnings
        if audio_url in summaries_cache:
            summary = summaries_cache[audio_url]
        else:
            # Summarize using transcript or description
            summary_text = transcript if transcript else ep.get("description", "")
            if summary_text:
                summary = summarize_podcast(summary_text, ep["title"], ep.get("show_name", ""))
                if summary is None:
                    summary = generate_fallback_podcast_summary(ep["title"], ep.get("show_name", ""))
            else:
                summary = generate_fallback_podcast_summary(ep["title"], ep.get("show_name", ""))
            summaries_cache[audio_url] = summary
            summaries_updated = True
        
        ep["summary"] = summary
        
        picked.append(ep)
        seen[ep["id"]] = now
    
    # Save caches
    save_podcast_seen(seen)
    if transcripts_updated:
        save_podcast_transcripts(transcripts_cache)
    if summaries_updated:
        save_podcast_summaries(summaries_cache)
    
    return picked
