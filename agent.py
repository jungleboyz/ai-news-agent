import os
import re
from typing import List, Optional
import feedparser
import json
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict

from config import settings
from services.service_registry import (
    get_embedding_service,
    get_vector_store,
    get_semantic_scorer,
)
from services.scoring_service import (
    USE_SEMANTIC_SCORING,
    score_keywords,
    score_semantic,
    score_items_batch,
)
from services.cache_service import load_json, save_json
from services.feed_service import update_feed_statuses

# Re-export for backwards compatibility (used by orchestration/)
from services.scoring_service import AI_KEYWORDS as USER_INTERESTS


# --- Config ---
SOURCES_FILE = "sources.txt"
OUT_DIR = "out"
DIGEST_TOP_N = None  # No overall digest limit
PER_SOURCE_CAP = 10  # Max items per RSS feed (most recent)
SEEN_PATH = os.path.join(OUT_DIR, "seen.json")
SUMMARIES_PATH = os.path.join(OUT_DIR, "summaries.json")


def ensure_out_dir() -> None:
    """Create the output folder if it doesn't exist."""
    os.makedirs(OUT_DIR, exist_ok=True)


def load_sources(path: str = SOURCES_FILE) -> List[str]:
    """Load news feed URLs from DB (active FeedSources), falling back to sources.txt."""
    try:
        from web.database import SessionLocal
        from web.models import FeedSource
        with SessionLocal() as session:
            urls = session.query(FeedSource.feed_url).filter(
                FeedSource.source_type == "news",
                FeedSource.status == "active",
            ).all()
            if urls:
                return [u[0] for u in urls]
    except Exception:
        pass  # DB not available, fall back to file

    with open(path, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]


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
            print(f"  Stored {len(items_to_store)} embeddings in ChromaDB")

    except Exception as e:
        print(f"  Failed to store embeddings: {e}")


def check_duplicates(items: List[dict], item_type: str = "news", threshold: float = 0.95) -> List[dict]:
    """Filter out duplicate items based on embedding similarity."""
    if not USE_SEMANTIC_SCORING:
        return items

    try:
        vector_store = get_vector_store()
        filtered = []

        for item in items:
            if not item.get("embedding"):
                filtered.append(item)
                continue

            similar = vector_store.find_similar(
                embedding=item["embedding"],
                item_type=item_type,
                threshold=threshold,
            )

            if not similar:
                filtered.append(item)
            else:
                print(f"  Duplicate detected: {item['title'][:50]}...")

        return filtered

    except Exception as e:
        print(f"  Duplicate check failed: {e}")
        return items


def fetch_rss_items(feed_url: str, limit: int = PER_SOURCE_CAP) -> List[dict]:
    """
    Fetch RSS items from a single feed URL.
    Returns the most recent `limit` items with title + link.
    """
    try:
        feed = feedparser.parse(feed_url)

        items = []
        for entry in feed.entries:
            title = getattr(entry, "title", "").strip()
            link = getattr(entry, "link", "").strip()
            if title and link:
                summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
                summary = re.sub(r"\s+", " ", summary).strip()
                items.append({"title": title, "link": link, "summary": summary})
                if limit and len(items) >= limit:
                    break

        return items
    except Exception as e:
        print(f"  Error fetching {feed_url}: {e}")
        return []


def make_id(title: str, link: str) -> str:
    """Generate a stable ID for an article based on its title and link."""
    raw = f"{title}|{link}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def run_agent() -> str:
    from summarizer import summarize_article, generate_fallback_summary
    ensure_out_dir()
    seen = load_json(SEEN_PATH, {})
    summaries_cache = load_json(SUMMARIES_PATH, {})
    now = time.time()

    # Merge DB-backed seen hashes to survive container restarts
    try:
        from web.db_writer import get_seen_hashes_from_db
        db_hashes = get_seen_hashes_from_db(days=30)
        if db_hashes:
            merged = 0
            for h in db_hashes:
                if h not in seen:
                    seen[h] = now  # Add with current timestamp
                    merged += 1
            if merged:
                print(f"Loaded {merged} previously seen hashes from database")
    except Exception as e:
        print(f"  DB dedup check skipped: {e}")

    sources = load_sources()
    print(f"Processing {len(sources)} news sources...")

    # Run podcast, video, and web scraper agents concurrently
    podcast_episodes = []
    video_episodes = []
    web_articles = []

    def _run_podcast():
        from podcast_agent import run_podcast_agent
        return run_podcast_agent()

    def _run_video():
        from video_agent import run_video_agent
        return run_video_agent(max_videos=5, days_back=7)

    def _run_web():
        from web_scraper_agent import run_web_scraper_agent
        return run_web_scraper_agent()

    with ThreadPoolExecutor(max_workers=3) as executor:
        podcast_f = executor.submit(_run_podcast)
        video_f = executor.submit(_run_video)
        web_f = executor.submit(_run_web)

        try:
            podcast_episodes = podcast_f.result(timeout=600)
            if podcast_episodes:
                print(f"Found {len(podcast_episodes)} podcast episodes")
            else:
                print("No new podcast episodes found (all may have been seen already)")
        except Exception as e:
            print(f"Warning: Podcast agent failed: {e}")
            print("   (News processing will continue independently)")

        try:
            video_episodes = video_f.result(timeout=600)
            if video_episodes:
                print(f"Found {len(video_episodes)} AI-relevant videos")
            else:
                print("No new AI-relevant videos found")
        except Exception as e:
            print(f"Warning: Video agent failed: {e}")
            print("   (News processing will continue independently)")

        try:
            web_articles = web_f.result(timeout=600)
            if web_articles:
                print(f"Found {len(web_articles)} web articles")
            else:
                print("No new web articles found")
        except Exception as e:
            print(f"Warning: Web scraper agent failed: {e}")
            print("   (News processing will continue independently)")

    all_items = []
    for src in sources:
        items = fetch_rss_items(src)
        if items:
            print(f"  {src}: {len(items)} items")
        for it in items:
            it["source"] = src
            it["id"] = make_id(it["title"], it["link"])
            all_items.append(it)

    print(f"Total news items fetched: {len(all_items)}")

    # Batch score all items using semantic scoring
    if USE_SEMANTIC_SCORING:
        print("Generating embeddings and semantic scores...")
        all_items = score_items_batch(all_items)
    else:
        for it in all_items:
            it["score"] = score_semantic(it["title"], it.get("summary", ""))

    # Remove already-seen items and deduplicate within batch by ID
    seen_ids = set()
    fresh = []
    for it in all_items:
        if it["id"] not in seen and it["id"] not in seen_ids:
            seen_ids.add(it["id"])
            fresh.append(it)
    print(f"Fresh news items: {len(fresh)} (seen: {len(all_items) - len(fresh)})")

    # Check for semantic duplicates
    if USE_SEMANTIC_SCORING:
        fresh = check_duplicates(fresh, item_type="news")
        print(f"After duplicate check: {len(fresh)} unique items")

    # Sort by score desc
    fresh.sort(key=lambda x: x["score"], reverse=True)

    picked = []
    if len(fresh) == 0:
        print("No fresh news items found - all items have been seen already")
        print("   (Delete out/seen.json to reset and see all items again)")
    summaries_updated = False

    # Batch pre-fetch article content via Firecrawl (if available)
    # Only pre-fetch articles that don't already have an RSS summary (saves credits)
    candidate_urls = [
        it["link"] for it in fresh
        if it["link"] not in summaries_cache
        and len(it.get("summary", "") or "") < 100  # Skip if RSS summary is good enough
    ]

    PREFETCH_LIMIT = 50  # Cap Firecrawl usage per run
    if candidate_urls:
        if len(candidate_urls) > PREFETCH_LIMIT:
            print(f"Capping Firecrawl pre-fetch to {PREFETCH_LIMIT} of {len(candidate_urls)} articles (saving credits)")
            candidate_urls = candidate_urls[:PREFETCH_LIMIT]
        try:
            from services.firecrawl_service import get_firecrawl_service
            from summarizer import set_article_cache
            fc = get_firecrawl_service()
            if fc.available:
                print(f"Batch pre-fetching {len(candidate_urls)} articles via Firecrawl...")
                prefetched = fc.batch_scrape(candidate_urls)
                if prefetched:
                    set_article_cache(prefetched)
                    print(f"  Pre-fetched {len(prefetched)} articles")
        except Exception as e:
            print(f"  Batch pre-fetch failed: {e}")

    for it in fresh:
        # Check cache first
        article_url = it["link"]
        if article_url in summaries_cache:
            it["ai_summary"] = summaries_cache[article_url]
        else:
            # Try to get summary from API (pass RSS summary to avoid Firecrawl when possible)
            summary = summarize_article(article_url, it["title"], rss_summary=it.get("summary"))
            if summary is None:
                # Use fallback when API is unavailable
                it["ai_summary"] = generate_fallback_summary(article_url, it["title"])
            else:
                it["ai_summary"] = summary
            # Cache the summary (whether from API or fallback)
            summaries_cache[article_url] = it["ai_summary"]
            summaries_updated = True

        picked.append(it)
        seen[it["id"]] = now

    save_json(SEEN_PATH, seen)
    if summaries_updated:
        save_json(SUMMARIES_PATH, summaries_cache)

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

    # Add web articles
    for article in web_articles:
        all_digest_items.append({
            "type": "web",
            "title": article["title"],
            "link": article["link"],
            "score": article["score"],
            "semantic_score": article.get("semantic_score"),
            "embedding_id": article.get("id"),
            "source": article.get("source", ""),
            "summary": article.get("summary", ""),
            "show_name": None
        })

    # Deduplicate by link across all item types (news, podcast, video, web)
    # Also check against DB links from previous digests
    try:
        from web.db_writer import get_seen_links_from_db
        seen_links = get_seen_links_from_db(days=14)
        print(f"Loaded {len(seen_links)} links from recent digests for cross-day dedup")
    except Exception:
        seen_links = set()
    deduped_items = []
    for item in all_digest_items:
        if item["link"] not in seen_links:
            seen_links.add(item["link"])
            deduped_items.append(item)
    if len(deduped_items) < len(all_digest_items):
        print(f"Removed {len(all_digest_items) - len(deduped_items)} cross-source duplicates")
    all_digest_items = deduped_items

    # Sort all items by score (descending)
    all_digest_items.sort(key=lambda x: x["score"], reverse=True)

    # Write digest — use Australia/Sydney date so the digest matches the local
    # calendar day even when the server runs in UTC (e.g. 6 AM AEDT = 7 PM UTC
    # the previous day).
    try:
        from zoneinfo import ZoneInfo
        digest_tz = ZoneInfo(settings.scheduler_timezone)
    except Exception:
        digest_tz = timezone.utc
    date_str = datetime.now(digest_tz).strftime("%Y-%m-%d")
    md_path = os.path.join(OUT_DIR, f"digest-{date_str}.md")
    html_path = os.path.join(OUT_DIR, f"digest-{date_str}.html")

    # Write markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# AI News Digest — {date_str}\n\n")
        f.write(f"Sources: {len(sources)} | New items considered: {len(fresh)}")
        if podcast_episodes:
            f.write(f" | Podcasts: {len(podcast_episodes)}")
        if web_articles:
            f.write(f" | Web: {len(web_articles)}")
        f.write(f" | Total items: {len(all_digest_items)}\n\n")

        # Unified list of all items (news + podcasts + web)
        for i, item in enumerate(all_digest_items, 1):
            tag = "MATCH" if item["score"] > 0 else "FALLBACK"
            if item["type"] == "news":
                item_type = "📰"
            elif item["type"] == "podcast":
                item_type = "🎙️"
            elif item["type"] == "web":
                item_type = "🌐"
            else:
                item_type = "📺"

            f.write(f"### {i}. {item_type} [{item['score']}] ({tag}) {item['title']}\n\n")

            if item["type"] in ("news", "web"):
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
        if web_articles:
            f.write(f' · Web: {len(web_articles)}')
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
            elif item["type"] == "web":
                item_type_icon = "🌐"
                item_type_label = "Web"
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
            if item["type"] in ("news", "web"):
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
                summary_label = "Why this matters:" if item["type"] in ("news", "web") else "Summary:"
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
        print(f"Saved digest to database")

        # Run topic clustering on the new digest
        if USE_SEMANTIC_SCORING:
            try:
                from tasks.clustering_tasks import cluster_latest_digest
                print("Running topic clustering...")
                result = cluster_latest_digest()
                if "error" not in result:
                    print(f"  Created {result.get('clusters_created', 0)} topic clusters")
                else:
                    print(f"  Clustering: {result.get('error')}")
            except Exception as ce:
                print(f"  Topic clustering failed: {ce}")

    except Exception as e:
        print(f"Database save failed: {e}")

    # Send email with digest
    try:
        from emailer import send_digest_email
        send_digest_email(html_path, md_path, date_str)
    except Exception as e:
        print(f"Email sending failed: {e}")

    # Update feed source statuses in DB after run
    update_feed_statuses("news", sources)

    return md_path


if __name__ == "__main__":
    path = run_agent()
    print(f"Digest written to: {path}")
