import os
import re
from typing import List, Optional
import feedparser
import json
import time
import hashlib
from datetime import datetime, timezone
from typing import Dict

# Semantic scoring imports
from services.embeddings import EmbeddingService
from services.vector_store import VectorStore
from services.semantic_scorer import SemanticScorer


# --- Config ---
SOURCES_FILE = "sources.txt"
OUT_DIR = "out"
DIGEST_TOP_N = 10
SEEN_PATH = os.path.join(OUT_DIR, "seen.json")
SUMMARIES_PATH = os.path.join(OUT_DIR, "summaries.json")
CHROMADB_DIR = "chromadb_data"

# Enable/disable semantic scoring (set to False to use keyword scoring)
USE_SEMANTIC_SCORING = True

# Keywords for fallback scoring (used when semantic scoring is disabled or fails)
USER_INTERESTS = [
    "genai", "generative ai", "llm", "agent", "agents",
    "openai", "anthropic", "gemini", "mistral", "claude",
    "cursor", "copilot", "aider", "enterprise", "bank",
    "marketing", "automation", "workflow", "funding", "acquisition"
]

# Semantic scorer singleton
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


def load_sources(path: str = SOURCES_FILE) -> List[str]:
    """Read RSS feed URLs from sources.txt (one URL per line)."""
    with open(path, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]


def norm(text: str) -> str:
    """Lowercase + collapse whitespace so matching is consistent."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def score_item_keywords(title: str, summary: str = "") -> int:
    """Legacy keyword-based scoring."""
    text = norm(f"{title} {summary}")
    score = 0
    for kw in USER_INTERESTS:
        if kw in text:
            score += 2
    return score


def score_item(title: str, summary: str = "") -> int:
    """Score an item using semantic scoring or fallback to keywords."""
    global _quota_exceeded

    if not USE_SEMANTIC_SCORING or _quota_exceeded:
        return score_item_keywords(title, summary)

    try:
        scorer = get_semantic_scorer()
        text = f"{title} {summary}".strip()
        semantic_score = scorer.score_text(text)
        # Convert to integer scale (0-10) for compatibility
        return scorer.score_to_int(semantic_score, scale=10)
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
            if not _quota_exceeded:
                print(f"  ⚠ OpenAI API quota/rate limit hit - switching to keyword scoring")
                _quota_exceeded = True
        else:
            print(f"  ⚠ Semantic scoring failed, using keywords: {e}")
        return score_item_keywords(title, summary)


def score_items_batch(items: List[dict]) -> List[dict]:
    """Score multiple items using batch embedding for efficiency.

    Args:
        items: List of items with 'title', 'summary', and 'id' keys.

    Returns:
        Items with 'score', 'semantic_score', and 'embedding' added.
    """
    global _quota_exceeded
    total = len(items)

    if not USE_SEMANTIC_SCORING or not items or _quota_exceeded:
        print(f"  📊 Scoring {total} items with keyword matching...")
        for i, item in enumerate(items):
            item["score"] = score_item_keywords(item["title"], item.get("summary", ""))
            item["semantic_score"] = None
            item["embedding"] = None
            if (i + 1) % 200 == 0:
                print(f"    Progress: {i + 1}/{total} items scored")
        print(f"  ✓ Keyword scoring complete")
        return items

    try:
        print(f"  🧠 Generating embeddings for {total} items...")
        print(f"    (This may take a moment - calling OpenAI API)")

        embedding_service = get_embedding_service()
        scorer = get_semantic_scorer()

        # Prepare texts for batch embedding
        texts = [f"{it['title']} {it.get('summary', '')}".strip() for it in items]

        # Batch generate embeddings
        embeddings = embedding_service.batch_embed(texts)

        print(f"  ✓ Embeddings received, scoring items...")

        # Score all items
        scored_count = 0
        fallback_count = 0
        for item, embedding in zip(items, embeddings):
            if embedding:
                semantic_score = scorer.score_item(embedding)
                item["score"] = scorer.score_to_int(semantic_score, scale=10)
                item["semantic_score"] = semantic_score
                item["embedding"] = embedding
                scored_count += 1
            else:
                # Fallback for failed embeddings
                item["score"] = score_item_keywords(item["title"], item.get("summary", ""))
                item["semantic_score"] = None
                item["embedding"] = None
                fallback_count += 1

        print(f"  ✓ Semantic scoring complete: {scored_count} semantic, {fallback_count} keyword fallback")
        return items

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate" in error_msg.lower() or "quota" in error_msg.lower():
            print(f"  ⚠ OpenAI API quota/rate limit hit - falling back to keyword scoring")
            _quota_exceeded = True
        else:
            print(f"  ⚠ Batch semantic scoring failed: {e}")

        print(f"  📊 Scoring {total} items with keyword matching...")
        for i, item in enumerate(items):
            item["score"] = score_item_keywords(item["title"], item.get("summary", ""))
            item["semantic_score"] = None
            item["embedding"] = None
            if (i + 1) % 200 == 0:
                print(f"    Progress: {i + 1}/{total} items scored")
        print(f"  ✓ Keyword scoring complete")
        return items


def store_embeddings(items: List[dict], item_type: str = "news") -> None:
    """Store item embeddings in ChromaDB.

    Args:
        items: List of items with 'id', 'title', 'link', 'embedding' keys.
        item_type: Type of items ("news", "podcast", "video").
    """
    if not USE_SEMANTIC_SCORING:
        return

    try:
        vector_store = get_vector_store()

        items_to_store = []
        for item in items:
            if item.get("embedding"):
                items_to_store.append({
                    "id": item["id"],
                    "text": f"{item['title']} {item.get('summary', '')}".strip(),
                    "metadata": {
                        "title": item["title"],
                        "link": item["link"],
                        "source": item.get("source", ""),
                        "semantic_score": item.get("semantic_score"),
                    },
                    "embedding": item["embedding"],
                })

        if items_to_store:
            vector_store.add_items_batch(items_to_store, item_type)
            print(f"  📦 Stored {len(items_to_store)} embeddings in ChromaDB")

    except Exception as e:
        print(f"  ⚠ Failed to store embeddings: {e}")


def check_duplicates(items: List[dict], item_type: str = "news", threshold: float = 0.95) -> List[dict]:
    """Filter out duplicate items based on embedding similarity.

    Args:
        items: List of items with 'id' and 'embedding' keys.
        item_type: Type of items.
        threshold: Similarity threshold for duplicate detection.

    Returns:
        Filtered list with duplicates removed.
    """
    if not USE_SEMANTIC_SCORING:
        return items

    try:
        vector_store = get_vector_store()
        filtered = []

        for item in items:
            if not item.get("embedding"):
                filtered.append(item)
                continue

            # Check if similar item already exists
            similar = vector_store.find_similar(
                embedding=item["embedding"],
                item_type=item_type,
                threshold=threshold,
            )

            if not similar:
                filtered.append(item)
            else:
                print(f"  🔄 Duplicate detected: {item['title'][:50]}...")

        return filtered

    except Exception as e:
        print(f"  ⚠ Duplicate check failed: {e}")
        return items


def fetch_rss_items(feed_url: str, limit: int = 10) -> List[dict]:
    """
    Fetch RSS items from a single feed URL.
    Returns a list of dicts with title + link.
    """
    try:
        feed = feedparser.parse(feed_url)

        items = []
        for entry in feed.entries[:limit]:
            title = getattr(entry, "title", "").strip()
            link = getattr(entry, "link", "").strip()
            if title and link:
                summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
                summary = re.sub(r"\s+", " ", summary).strip()
                items.append({"title": title, "link": link, "summary": summary})

        return items
    except Exception as e:
        print(f"  ⚠ Error fetching {feed_url}: {e}")
        return []


def make_id(title: str, link: str) -> str:
    """Generate a stable ID for an article based on its title and link."""
    raw = f"{title}|{link}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def load_seen() -> Dict[str, float]:
    if not os.path.exists(SEEN_PATH):
        return {}
    with open(SEEN_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_seen(seen: Dict[str, float]) -> None:
    with open(SEEN_PATH, "w", encoding="utf-8") as f:
        json.dump(seen, f, indent=2)


def load_summaries() -> Dict[str, str]:
    """Load cached summaries by article URL."""
    if not os.path.exists(SUMMARIES_PATH):
        return {}
    with open(SUMMARIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_summaries(summaries: Dict[str, str]) -> None:
    """Save cached summaries by article URL."""
    with open(SUMMARIES_PATH, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)


def run_agent() -> str:
    from summarizer import summarize_article, generate_fallback_summary
    ensure_out_dir()
    seen = load_seen()
    summaries_cache = load_summaries()
    now = time.time()

    sources = load_sources()
    print(f"📰 Processing {len(sources)} news sources...")
    
    # Run podcast agent (will return empty list if no podcasts configured)
    # This is independent - news processing continues even if podcasts fail
    try:
        from podcast_agent import run_podcast_agent
        podcast_episodes = run_podcast_agent()
        if podcast_episodes:
            print(f"✓ Found {len(podcast_episodes)} podcast episodes")
        else:
            print("⚠ No new podcast episodes found (all may have been seen already)")
    except Exception as e:
        print(f"⚠ Warning: Podcast agent failed: {e}")
        print("   (News processing will continue independently)")
        podcast_episodes = []

    # Run video agent (will return empty list if no videos configured)
    try:
        from video_agent import run_video_agent
        video_episodes = run_video_agent(max_videos=5, days_back=7)
        if video_episodes:
            print(f"✓ Found {len(video_episodes)} AI-relevant videos")
        else:
            print("⚠ No new AI-relevant videos found")
    except Exception as e:
        print(f"⚠ Warning: Video agent failed: {e}")
        print("   (News processing will continue independently)")
        video_episodes = []

    all_items = []
    for src in sources:
        items = fetch_rss_items(src, limit=15)
        if items:
            print(f"  ✓ {src}: {len(items)} items")
        for it in items:
            it["source"] = src
            it["id"] = make_id(it["title"], it["link"])
            all_items.append(it)

    print(f"📋 Total news items fetched: {len(all_items)}")

    # Batch score all items using semantic scoring
    if USE_SEMANTIC_SCORING:
        print("🧠 Generating embeddings and semantic scores...")
        all_items = score_items_batch(all_items)
    else:
        for it in all_items:
            it["score"] = score_item(it["title"], it.get("summary", ""))
    
    # Remove already-seen items
    fresh = [it for it in all_items if it["id"] not in seen]
    print(f"🆕 Fresh news items: {len(fresh)} (seen: {len(all_items) - len(fresh)})")

    # Check for semantic duplicates
    if USE_SEMANTIC_SCORING:
        fresh = check_duplicates(fresh, item_type="news")
        print(f"🔍 After duplicate check: {len(fresh)} unique items")

    # Sort by score desc
    fresh.sort(key=lambda x: x["score"], reverse=True)

    # Keep top N, but also avoid "all from one source" dominance (light diversity)
    picked = []
    if len(fresh) == 0:
        print("⚠ No fresh news items found - all items have been seen already")
        print("   (Delete out/seen.json to reset and see all items again)")
    per_source_cap = 4
    per_source_count = {}
    summaries_updated = False

    for it in fresh:
        # Keep keyword matches first, but allow fallback items too
        src = it["source"]
        per_source_count[src] = per_source_count.get(src, 0)

        if per_source_count[src] >= per_source_cap:
            continue

        # Check cache first
        article_url = it["link"]
        if article_url in summaries_cache:
            it["ai_summary"] = summaries_cache[article_url]
        else:
            # Try to get summary from API
            summary = summarize_article(article_url, it["title"])
            if summary is None:
                # Use fallback when API is unavailable
                it["ai_summary"] = generate_fallback_summary(article_url, it["title"])
            else:
                it["ai_summary"] = summary
            # Cache the summary (whether from API or fallback)
            summaries_cache[article_url] = it["ai_summary"]
            summaries_updated = True

        picked.append(it)
        per_source_count[src] += 1
        seen[it["id"]] = now

        if len(picked) >= DIGEST_TOP_N:
            break

    save_seen(seen)
    if summaries_updated:
        save_summaries(summaries_cache)

    # Store embeddings in ChromaDB for picked items
    if USE_SEMANTIC_SCORING and picked:
        store_embeddings(picked, item_type="news")

    # Combine news and podcasts into unified list, sorted by score
    all_digest_items = []
    
    # Add news items
    for it in picked:
        all_digest_items.append({
            "type": "news",
            "title": it["title"],
            "link": it["link"],
            "score": it["score"],
            "semantic_score": it.get("semantic_score"),
            "embedding_id": it.get("id"),  # Use item ID as embedding reference
            "source": it.get("source", ""),
            "summary": it.get("ai_summary", ""),
            "show_name": None
        })
    
    # Add podcast episodes
    for ep in podcast_episodes:
        all_digest_items.append({
            "type": "podcast",
            "title": ep["title"],
            "link": ep["link"],
            "score": ep["score"],
            "semantic_score": ep.get("semantic_score"),
            "embedding_id": ep.get("id"),
            "source": ep.get("show_name", "Unknown"),
            "summary": ep.get("summary", ""),
            "show_name": ep.get("show_name", "Unknown")
        })

    # Add video episodes
    for vid in video_episodes:
        all_digest_items.append({
            "type": "video",
            "title": vid["title"],
            "link": vid["link"],
            "score": vid["score"],
            "semantic_score": vid.get("semantic_score"),
            "embedding_id": vid.get("id"),
            "source": vid.get("channel", "Unknown"),
            "summary": vid.get("summary", ""),
            "show_name": vid.get("channel", "Unknown")
        })
    
    # Sort all items by score (descending)
    all_digest_items.sort(key=lambda x: x["score"], reverse=True)

    # Write digest
    date_str = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    md_path = os.path.join(OUT_DIR, f"digest-{date_str}.md")
    html_path = os.path.join(OUT_DIR, f"digest-{date_str}.html")

    # Write markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# AI News Digest — {date_str}\n\n")
        f.write(f"Sources: {len(sources)} | New items considered: {len(fresh)}")
        if podcast_episodes:
            f.write(f" | Podcasts: {len(podcast_episodes)}")
        f.write(f" | Total items: {len(all_digest_items)}\n\n")
        
        # Unified list of all items (news + podcasts)
        for i, item in enumerate(all_digest_items, 1):
            tag = "MATCH" if item["score"] > 0 else "FALLBACK"
            if item["type"] == "news":
                item_type = "📰"
            elif item["type"] == "podcast":
                item_type = "🎙️"
            else:
                item_type = "📺"

            f.write(f"### {i}. {item_type} [{item['score']}] ({tag}) {item['title']}\n\n")

            if item["type"] == "news":
                f.write(f"- Link: {item['link']}\n")
                f.write(f"- Source: {item['source']}\n\n")
            elif item["type"] == "podcast":
                f.write(f"- Show: {item['show_name']}\n")
                f.write(f"- Link: {item['link']}\n\n")
            else:
                f.write(f"- Channel: {item['show_name']}\n")
                f.write(f"- Link: {item['link']}\n\n")

            # Show summary if it exists
            if item["summary"]:
                label = "**Why this matters:**" if item["type"] == "news" else "**Summary:**"
                f.write(f"{label}\n")
                f.write(f"{item['summary']}\n\n")

            f.write("---\n\n")

    # Write HTML with Tailwind CSS
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<!doctype html>\n")
        f.write('<html lang="en">\n')
        f.write("<head>\n")
        f.write('  <meta charset="UTF-8" />\n')
        f.write('  <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n')
        f.write(f'  <title>AI News Digest — {date_str}</title>\n')
        f.write('  <script src="https://cdn.tailwindcss.com"></script>\n')
        f.write("</head>\n")
        f.write('<body class="bg-slate-950 text-slate-100 font-sans antialiased">\n')
        f.write('  <main class="max-w-4xl mx-auto px-4 py-10 space-y-8">\n')
        f.write('    <header class="space-y-2">\n')
        f.write('      <p class="text-sm uppercase tracking-wide text-slate-400">Digest</p>\n')
        f.write(f'      <h1 class="text-3xl font-bold">AI News Digest — {date_str}</h1>\n')
        f.write(f'      <p class="text-slate-300">Sources: {len(sources)} · New items considered: {len(fresh)}')
        if podcast_episodes:
            f.write(f' · Podcasts: {len(podcast_episodes)}')
        f.write(f' · Total items: {len(all_digest_items)}</p>\n')
        f.write("    </header>\n\n")
        
        # Unified section with all items (news + podcasts) sorted by score
        f.write('    <section class="space-y-6">\n')

        for i, item in enumerate(all_digest_items, 1):
            tag = "MATCH" if item["score"] > 0 else "FALLBACK"
            tag_class = "text-emerald-400" if item["score"] > 0 else "text-slate-400"
            if item["type"] == "news":
                item_type_icon = "📰"
                item_type_label = "News"
            elif item["type"] == "podcast":
                item_type_icon = "🎙️"
                item_type_label = "Podcast"
            else:
                item_type_icon = "📺"
                item_type_label = "Video"
            
            f.write('      <article class="relative rounded-lg border border-slate-800 bg-slate-900/70 p-6 shadow-sm hover:bg-slate-900 transition-colors">\n')
            f.write('        <div class="absolute top-3 right-3">\n')
            f.write('          <span class="px-2 py-1 text-xs font-bold uppercase tracking-wide rounded-md bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 shadow-sm">Fresh</span>\n')
            f.write('        </div>\n')
            f.write(f'        <div class="flex items-start gap-3 mb-3">\n')
            f.write(f'          <span class="text-sm font-mono text-slate-500">{i}.</span>\n')
            f.write(f'          <div class="flex-1">\n')
            f.write(f'            <div class="flex items-center gap-2 mb-1">\n')
            f.write(f'              <span class="text-lg">{item_type_icon}</span>\n')
            f.write(f'              <span class="text-xs text-slate-400 uppercase">{item_type_label}</span>\n')
            f.write(f'            </div>\n')
            f.write(f'            <h2 class="text-xl font-semibold text-slate-50 leading-tight">{item["title"]}</h2>\n')
            f.write(f'            <div class="mt-2 flex items-center gap-2 text-sm">\n')
            f.write(f'              <span class="px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">[{item["score"]}]</span>\n')
            f.write(f'              <span class="px-2 py-0.5 rounded {tag_class} font-medium">({tag})</span>\n')
            f.write(f'            </div>\n')
            f.write(f'          </div>\n')
            f.write(f'        </div>\n')
            
            f.write('        <ul class="mt-4 space-y-2 text-sm text-slate-200">\n')
            if item["type"] == "news":
                f.write(f'          <li><span class="font-semibold text-slate-100">Link:</span> <a href="{item["link"]}" target="_blank" rel="noopener noreferrer" class="text-blue-400 hover:text-blue-300 underline break-all">{item["link"]}</a></li>\n')
                f.write(f'          <li><span class="font-semibold text-slate-100">Source:</span> <span class="text-slate-300">{item["source"]}</span></li>\n')
            elif item["type"] == "podcast":
                f.write(f'          <li><span class="font-semibold text-slate-100">Show:</span> <span class="text-slate-300">{item["show_name"]}</span></li>\n')
                f.write(f'          <li><span class="font-semibold text-slate-100">Link:</span> <a href="{item["link"]}" target="_blank" rel="noopener noreferrer" class="text-blue-400 hover:text-blue-300 underline break-all">{item["link"]}</a></li>\n')
            else:
                f.write(f'          <li><span class="font-semibold text-slate-100">Channel:</span> <span class="text-slate-300">{item["show_name"]}</span></li>\n')
                f.write(f'          <li><span class="font-semibold text-slate-100">Link:</span> <a href="{item["link"]}" target="_blank" rel="noopener noreferrer" class="text-blue-400 hover:text-blue-300 underline break-all">{item["link"]}</a></li>\n')
            f.write("        </ul>\n")

            # Show summary if it exists
            if item["summary"]:
                summary_label = "Why this matters:" if item["type"] == "news" else "Summary:"
                is_error = item["summary"].startswith("(Failed") or item["summary"].startswith("(OpenAI")
                summary_class = "text-red-300" if is_error else "text-slate-200"
                f.write('        <div class="mt-4 pt-4 border-t border-slate-800">\n')
                f.write(f'          <p class="font-semibold text-slate-100 mb-2">{summary_label}</p>\n')
                f.write(f'          <div class="{summary_class} text-sm leading-relaxed whitespace-pre-wrap">{item["summary"]}</div>\n')
                f.write("        </div>\n")

            f.write("      </article>\n\n")

        f.write("    </section>\n")
        
        f.write("  </main>\n")
        f.write("</body>\n")
        f.write("</html>\n")

    # Save to database for web interface
    try:
        from web.db_writer import save_digest_to_db
        from datetime import date as date_type
        digest_date = date_type.fromisoformat(date_str)

        # Count podcast sources (unique show names)
        podcast_shows = set(ep.get("show_name", "Unknown") for ep in podcast_episodes)

        save_digest_to_db(
            digest_date=digest_date,
            news_sources_count=len(sources),
            podcast_sources_count=len(podcast_shows),
            total_items_considered=len(fresh),
            items=all_digest_items,
            md_path=md_path,
            html_path=html_path
        )
        print(f"💾 Saved digest to database")

        # Run topic clustering on the new digest
        if USE_SEMANTIC_SCORING:
            try:
                from tasks.clustering_tasks import cluster_latest_digest
                print("🏷️ Running topic clustering...")
                result = cluster_latest_digest()
                if "error" not in result:
                    print(f"  ✓ Created {result.get('clusters_created', 0)} topic clusters")
                else:
                    print(f"  ⚠ Clustering: {result.get('error')}")
            except Exception as ce:
                print(f"  ⚠ Topic clustering failed: {ce}")

    except Exception as e:
        print(f"⚠ Database save failed: {e}")

    # Send email with digest
    try:
        from emailer import send_digest_email
        send_digest_email(html_path, md_path, date_str)
    except Exception as e:
        print(f"⚠ Email sending failed: {e}")

    return md_path


if __name__ == "__main__":
    path = run_agent()
    print(f"Digest written to: {path}")

 