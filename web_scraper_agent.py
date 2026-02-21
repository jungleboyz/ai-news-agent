"""
Web Scraper Agent — Scrapes blog/news listing pages via Firecrawl.
Extracts article links from markdown, scores, and returns items for the digest.
Follows the same pattern as podcast_agent.py and video_agent.py.
"""
import os
import re
import json
import hashlib
import time
from typing import List, Dict, Optional
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

# Semantic scoring imports
from services.embeddings import EmbeddingService
from services.semantic_scorer import SemanticScorer

# Legacy keyword list for fallback scoring
USER_INTERESTS = [
    "genai", "generative ai", "llm", "agent", "agents",
    "openai", "anthropic", "gemini", "mistral", "claude",
    "cursor", "copilot", "aider", "enterprise", "bank",
    "marketing", "automation", "workflow", "funding", "acquisition"
]

# --- Config ---
WEB_SOURCES_FILE = "web_sources.txt"
OUT_DIR = "out"
WEB_SEEN_PATH = os.path.join(OUT_DIR, "web_seen.json")
WEB_SUMMARIES_PATH = os.path.join(OUT_DIR, "web_summaries.json")
MAX_LINKS_PER_PAGE = 15
LISTING_CONTENT_LENGTH = 8000
RATE_LIMIT_DELAY = 2.0  # seconds between listing page scrapes

# Enable/disable semantic scoring
USE_SEMANTIC_SCORING = True

# Service singletons
_semantic_scorer: Optional[SemanticScorer] = None
_embedding_service: Optional[EmbeddingService] = None

# Quota exceeded flag
_quota_exceeded: bool = False

# Skip patterns for link extraction (navigation, footer, meta links)
_SKIP_PATTERNS = [
    '/tag/', '/category/', '/about', '/contact', '/privacy',
    '/login', '/search', '/feed', '/rss', '/page/', '/author/',
    '/careers', '/subscribe', '/newsletter', '/topics/', '/series/',
    '/events/', '/webinar', '/terms', '/sitemap', '/cookie',
    '/legal', '/press-release', '/annual-report',
    'javascript:', 'mailto:', '#',
]

# Regex for markdown links: [Title](URL)
MARKDOWN_LINK_RE = re.compile(r'\[([^\]]{10,200})\]\((https?://[^\)\s]+)\)')
RELATIVE_LINK_RE = re.compile(r'\[([^\]]{10,200})\]\((/[^\)\s]+)\)')


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def get_semantic_scorer() -> SemanticScorer:
    global _semantic_scorer
    if _semantic_scorer is None:
        _semantic_scorer = SemanticScorer(embedding_service=get_embedding_service())
    return _semantic_scorer


def ensure_out_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def load_web_sources(path: str = WEB_SOURCES_FILE) -> List[str]:
    """Load web listing URLs from DB (source_type='web'), falling back to web_sources.txt."""
    try:
        from web.database import SessionLocal
        from web.models import FeedSource
        with SessionLocal() as session:
            urls = session.query(FeedSource.feed_url).filter(
                FeedSource.source_type == "web",
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


def load_web_seen() -> Dict[str, float]:
    if not os.path.exists(WEB_SEEN_PATH):
        return {}
    with open(WEB_SEEN_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_web_seen(seen: Dict[str, float]) -> None:
    with open(WEB_SEEN_PATH, "w", encoding="utf-8") as f:
        json.dump(seen, f, indent=2)


def load_web_summaries() -> Dict[str, str]:
    if not os.path.exists(WEB_SUMMARIES_PATH):
        return {}
    with open(WEB_SUMMARIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_web_summaries(summaries: Dict[str, str]) -> None:
    with open(WEB_SUMMARIES_PATH, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)


def make_id(title: str, link: str) -> str:
    raw = f"{title}|{link}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def score_item_keywords(title: str, summary: str = "") -> int:
    text = norm(f"{title} {summary}")
    return sum(2 for kw in USER_INTERESTS if kw in text)


def score_item(title: str, summary: str = "") -> int:
    global _quota_exceeded
    if not USE_SEMANTIC_SCORING or _quota_exceeded:
        return score_item_keywords(title, summary)
    try:
        scorer = get_semantic_scorer()
        semantic_score = scorer.score_text(f"{title} {summary}".strip())
        return scorer.score_to_int(semantic_score, scale=10)
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
            if not _quota_exceeded:
                print("  ⚠ OpenAI API quota/rate limit hit - switching to keyword scoring")
                _quota_exceeded = True
        else:
            print(f"  ⚠ Semantic scoring failed, using keywords: {e}")
        return score_item_keywords(title, summary)


# --- Listing page scraping and link extraction ---

def scrape_listing_page(url: str) -> Optional[str]:
    """Scrape a listing page via Firecrawl, return markdown."""
    from services.firecrawl_service import get_firecrawl_service
    fc = get_firecrawl_service()
    if not fc.available:
        return None
    return fc.scrape_article(url, max_length=LISTING_CONTENT_LENGTH)


def _is_same_domain(url: str, base_netloc: str) -> bool:
    try:
        candidate = urlparse(url).netloc.lower().lstrip("www.")
        base = base_netloc.lower().lstrip("www.")
        return candidate == base or candidate.endswith("." + base)
    except Exception:
        return False


def _should_skip(url: str) -> bool:
    url_lower = url.lower()
    return any(p in url_lower for p in _SKIP_PATTERNS)


def extract_article_links(markdown: str, listing_url: str) -> List[dict]:
    """Extract candidate article links from listing-page markdown.

    Parses markdown for [Title](URL) patterns, filters to same-domain
    article links, and skips navigation/category/meta links.
    """
    base_netloc = urlparse(listing_url).netloc
    results = []
    seen_urls = set()

    def try_add(title: str, url: str):
        title = title.strip()
        # Clean URL of trailing markdown artifacts
        url = url.strip().split(" ")[0].rstrip(")")
        if not title or not url:
            return
        if url in seen_urls:
            return
        if _should_skip(url):
            return
        if not _is_same_domain(url, base_netloc):
            return
        seen_urls.add(url)
        results.append({"title": title, "link": url})

    # First pass: absolute URLs
    for m in MARKDOWN_LINK_RE.finditer(markdown):
        try_add(m.group(1), m.group(2))
        if len(results) >= MAX_LINKS_PER_PAGE:
            break

    # Second pass: relative URLs (converted to absolute)
    if len(results) < MAX_LINKS_PER_PAGE:
        for m in RELATIVE_LINK_RE.finditer(markdown):
            abs_url = urljoin(listing_url, m.group(2))
            try_add(m.group(1), abs_url)
            if len(results) >= MAX_LINKS_PER_PAGE:
                break

    return results


# --- Main entry point ---

def run_web_scraper_agent() -> List[dict]:
    """
    Scrape web listing pages, extract article links, score and return items.
    Returns a list of article dicts for inclusion in the digest.
    """
    from summarizer import summarize_article, generate_fallback_summary

    ensure_out_dir()
    seen = load_web_seen()
    summaries_cache = load_web_summaries()
    now = time.time()

    # Merge DB-backed seen hashes to survive container restarts
    try:
        from web.db_writer import get_seen_hashes_from_db
        db_hashes = get_seen_hashes_from_db(days=30, item_type="web")
        if db_hashes:
            merged = 0
            for h in db_hashes:
                if h not in seen:
                    seen[h] = now
                    merged += 1
            if merged:
                print(f"🔄 Loaded {merged} previously seen web hashes from database")
    except Exception as e:
        print(f"  ⚠ DB dedup check skipped: {e}")

    sources = load_web_sources()
    if not sources:
        print("⚠ No web sources found in web_sources.txt")
        return []

    print(f"🌐 Processing {len(sources)} web sources via Firecrawl...")

    all_items = []
    for i, listing_url in enumerate(sources):
        # Scrape listing page
        markdown = scrape_listing_page(listing_url)
        if not markdown:
            print(f"  ✗ {listing_url}: scrape failed")
            continue

        # Extract article links
        articles = extract_article_links(markdown, listing_url)
        if articles:
            print(f"  ✓ {listing_url}: {len(articles)} articles")
        else:
            print(f"  ✗ {listing_url}: 0 articles found")
            continue

        for article in articles:
            article["source"] = listing_url
            article["id"] = make_id(article["title"], article["link"])
            all_items.append(article)

        # Rate limit between scrapes
        if i < len(sources) - 1:
            time.sleep(RATE_LIMIT_DELAY)

    print(f"📋 Total web articles found: {len(all_items)}")

    # Remove already-seen items
    fresh = [it for it in all_items if it["id"] not in seen]
    print(f"🆕 Fresh web articles: {len(fresh)} (seen: {len(all_items) - len(fresh)})")

    if not fresh:
        return []

    # Score items
    print(f"📊 Scoring {len(fresh)} web articles...")
    for it in fresh:
        it["score"] = score_item(it["title"])

    # Sort by score desc
    fresh.sort(key=lambda x: x["score"], reverse=True)

    # Summarize articles using Firecrawl + Claude
    picked = []
    summaries_updated = False

    # Batch pre-fetch article content via Firecrawl
    candidate_urls = [it["link"] for it in fresh if it["link"] not in summaries_cache]
    if candidate_urls:
        try:
            from services.firecrawl_service import get_firecrawl_service
            from summarizer import set_article_cache
            fc = get_firecrawl_service()
            if fc.available:
                print(f"🔥 Batch pre-fetching {len(candidate_urls)} web articles via Firecrawl...")
                prefetched = fc.batch_scrape(candidate_urls)
                if prefetched:
                    set_article_cache(prefetched)
                    print(f"  ✓ Pre-fetched {len(prefetched)} articles")
        except Exception as e:
            print(f"  ⚠ Batch pre-fetch failed: {e}")

    for it in fresh:
        article_url = it["link"]
        if article_url in summaries_cache:
            it["summary"] = summaries_cache[article_url]
        else:
            summary = summarize_article(article_url, it["title"])
            if summary is None:
                it["summary"] = generate_fallback_summary(article_url, it["title"])
            else:
                it["summary"] = summary
            summaries_cache[article_url] = it["summary"]
            summaries_updated = True

        picked.append(it)
        seen[it["id"]] = now

    # Save caches
    save_web_seen(seen)
    if summaries_updated:
        save_web_summaries(summaries_cache)

    # Update feed source statuses in DB
    _update_feed_statuses(sources)

    print(f"✓ Web scraper complete: {len(picked)} articles")
    return picked


def _update_feed_statuses(listing_urls: list):
    """Update FeedSource last_fetched after a web scraper run."""
    try:
        from web.database import SessionLocal
        from web.models import FeedSource
        with SessionLocal() as session:
            feeds = session.query(FeedSource).filter(
                FeedSource.source_type == "web",
                FeedSource.status.in_(["active", "error"]),
            ).all()
            for feed in feeds:
                if feed.feed_url in listing_urls:
                    feed.last_fetched = datetime.now(timezone.utc)
                    feed.status = "active"
                    feed.error_message = None
            session.commit()
    except Exception:
        pass  # Non-critical


if __name__ == "__main__":
    articles = run_web_scraper_agent()
    print(f"\nFound {len(articles)} web articles")
    for i, a in enumerate(articles[:10], 1):
        print(f"  {i}. [{a['score']}] {a['title'][:60]}...")
        print(f"     {a['link']}")
