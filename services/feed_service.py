"""Parallel feed fetching and consolidated feed status updates.

Replaces the sequential feed loops and duplicated _update_feed_statuses()
across agent.py, podcast_agent.py, video_agent.py, and web_scraper_agent.py.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Callable, List


def fetch_feeds_parallel(
    feed_urls: List[str],
    parse_fn: Callable[[str], list],
    max_workers: int = 10,
    timeout: int = 15,
) -> dict:
    """Fetch multiple feeds concurrently.

    Args:
        feed_urls: List of feed URLs to fetch.
        parse_fn: Callable that takes a URL and returns a list of items.
        max_workers: Max concurrent threads.
        timeout: Per-future timeout in seconds (not used for individual fetches,
                 feedparser handles its own timeout).

    Returns:
        Dict mapping feed_url -> list of items returned by parse_fn.
    """
    results: dict = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(parse_fn, url): url for url in feed_urls
        }
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                items = future.result(timeout=timeout)
                results[url] = items or []
            except Exception as e:
                print(f"  Feed fetch failed for {url}: {e}")
                results[url] = []

    return results


def update_feed_statuses(source_type: str, feed_urls: list) -> None:
    """Update FeedSource last_fetched and status after an agent run.

    Consolidated from the 4 identical _update_feed_statuses() implementations.
    Always uses UTC timestamps.

    Args:
        source_type: One of "news", "podcast", "video", "web".
        feed_urls: List of feed URLs that were processed.
    """
    try:
        from web.database import SessionLocal
        from web.models import FeedSource

        with SessionLocal() as session:
            feeds = session.query(FeedSource).filter(
                FeedSource.source_type == source_type,
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
