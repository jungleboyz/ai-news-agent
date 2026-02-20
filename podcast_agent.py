import os
import re
from typing import List, Dict, Optional
import feedparser
import json
import time
import hashlib
from datetime import datetime, timezone

# Semantic scoring imports
from services.embeddings import EmbeddingService
from services.vector_store import VectorStore
from services.semantic_scorer import SemanticScorer

# Legacy keyword list for fallback scoring
USER_INTERESTS = [
    "genai", "generative ai", "llm", "agent", "agents",
    "openai", "anthropic", "gemini", "mistral", "claude",
    "cursor", "copilot", "aider", "enterprise", "bank",
    "marketing", "automation", "workflow", "funding", "acquisition"
]

# --- Config ---
PODCASTS_FILE = "podcasts.txt"
OUT_DIR = "out"
CHROMADB_DIR = "chromadb_data"
MAX_EPISODES_PER_FEED = 5
TRANSCRIBE_MINUTES = 15
DIGEST_TOP_PODCASTS = 5
PODCAST_SEEN_PATH = os.path.join(OUT_DIR, "podcast_seen.json")
PODCAST_TRANSCRIPTS_PATH = os.path.join(OUT_DIR, "podcast_transcripts.json")
PODCAST_SUMMARIES_PATH = os.path.join(OUT_DIR, "podcast_summaries.json")

# Enable/disable semantic scoring
USE_SEMANTIC_SCORING = True

# Service singletons
_semantic_scorer: Optional[SemanticScorer] = None
_vector_store: Optional[VectorStore] = None
_embedding_service: Optional[EmbeddingService] = None

# Quota exceeded flag - prevents repeated API calls after 429 error
_quota_exceeded: bool = False


def get_embedding_service() -> EmbeddingService:
    """Get or create the embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def get_vector_store() -> VectorStore:
    """Get or create the vector store singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(
            persist_dir=CHROMADB_DIR,
            embedding_service=get_embedding_service()
        )
    return _vector_store


def get_semantic_scorer() -> SemanticScorer:
    """Get or create the semantic scorer singleton."""
    global _semantic_scorer
    if _semantic_scorer is None:
        _semantic_scorer = SemanticScorer(embedding_service=get_embedding_service())
    return _semantic_scorer


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


def norm(text: str) -> str:
    """Lowercase + collapse whitespace so matching is consistent."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def score_item_keywords(title: str, description: str = "", transcript: str = "") -> int:
    """Legacy keyword-based scoring for podcast episodes."""
    text = norm(f"{title} {description} {transcript}")
    score = 0
    for kw in USER_INTERESTS:
        if kw in text:
            score += 2
    return score


def score_item(title: str, description: str = "", transcript: str = "") -> int:
    """Score podcast episode using semantic or keyword scoring."""
    global _quota_exceeded

    if not USE_SEMANTIC_SCORING or _quota_exceeded:
        return score_item_keywords(title, description, transcript)

    try:
        scorer = get_semantic_scorer()
        text = f"{title} {description} {transcript}".strip()
        semantic_score = scorer.score_text(text)
        return scorer.score_to_int(semantic_score, scale=10)
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
            if not _quota_exceeded:
                print(f"  ⚠ OpenAI API quota/rate limit hit - switching to keyword scoring")
                _quota_exceeded = True
        else:
            print(f"  ⚠ Semantic scoring failed, using keywords: {e}")
        return score_item_keywords(title, description, transcript)


def score_episode_semantic(ep: dict, transcript: str = "") -> dict:
    """Score an episode and store embedding information.

    Args:
        ep: Episode dict with title, description, id.
        transcript: Optional transcript text.

    Returns:
        Episode dict with score, semantic_score, and embedding added.
    """
    global _quota_exceeded

    if not USE_SEMANTIC_SCORING or _quota_exceeded:
        ep["score"] = score_item_keywords(ep["title"], ep.get("description", ""), transcript)
        ep["semantic_score"] = None
        ep["embedding"] = None
        return ep

    try:
        embedding_service = get_embedding_service()
        scorer = get_semantic_scorer()

        text = f"{ep['title']} {ep.get('description', '')} {transcript}".strip()
        embedding = embedding_service.get_embedding(text)
        semantic_score = scorer.score_item(embedding)

        ep["score"] = scorer.score_to_int(semantic_score, scale=10)
        ep["semantic_score"] = semantic_score
        ep["embedding"] = embedding

        return ep

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
            if not _quota_exceeded:
                print(f"  ⚠ OpenAI API quota/rate limit hit - switching to keyword scoring")
                _quota_exceeded = True
        else:
            print(f"  ⚠ Semantic scoring failed for episode: {e}")
        ep["score"] = score_item_keywords(ep["title"], ep.get("description", ""), transcript)
        ep["semantic_score"] = None
        ep["embedding"] = None
        return ep


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
            print(f"  📦 Stored {len(items_to_store)} podcast embeddings in ChromaDB")

    except Exception as e:
        print(f"  ⚠ Failed to store podcast embeddings: {e}")


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


def run_podcast_agent(skip_transcription: bool = None) -> List[dict]:
    """
    Main function to process podcast feeds and return scored episodes.
    Returns a list of picked episodes with transcripts and summaries.

    Args:
        skip_transcription: Deprecated — transcription now uses the smart cascade
                          (web scrape → YouTube → RSS description). No audio download.
    """
    from services.transcript_service import TranscriptService
    from summarizer import summarize_podcast, generate_fallback_podcast_summary

    ts = TranscriptService()
    
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
    
    # Initial score episodes (will update after transcription)
    total = len(fresh)
    if USE_SEMANTIC_SCORING and not _quota_exceeded:
        print(f"🧠 Generating initial embeddings and semantic scores for {total} episodes...")
    else:
        print(f"📊 Scoring {total} episodes with keyword matching...")
    for i, ep in enumerate(fresh):
        ep["score"] = score_item(ep["title"], ep.get("description", ""), "")
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

        print(f"  📝 {ep['title'][:50]}... [transcript: {transcript_source}]")
        
        ep["transcript"] = transcript

        # Re-score with transcript (using semantic scoring if enabled)
        ep = score_episode_semantic(ep, transcript)
        
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
    save_podcast_seen(seen)
    if transcripts_updated:
        save_podcast_transcripts(transcripts_cache)
    if summaries_updated:
        save_podcast_summaries(summaries_cache)

    # Store embeddings in ChromaDB
    if USE_SEMANTIC_SCORING and picked:
        store_podcast_embeddings(picked)

    # Update feed source statuses in DB
    _update_feed_statuses(feeds)

    return picked


def _update_feed_statuses(feed_urls: list):
    """Update FeedSource last_fetched after a podcast agent run."""
    try:
        from web.database import SessionLocal
        from web.models import FeedSource
        with SessionLocal() as session:
            feeds = session.query(FeedSource).filter(
                FeedSource.source_type == "podcast",
                FeedSource.status.in_(["active", "error"]),
            ).all()
            for feed in feeds:
                if feed.feed_url in feed_urls:
                    feed.last_fetched = datetime.now(timezone.utc)
                    feed.status = "active"
                    feed.error_message = None
            session.commit()
    except Exception:
        pass  # Non-critical
