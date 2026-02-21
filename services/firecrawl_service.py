"""Firecrawl SDK wrapper for article scraping and web search."""
from typing import Optional

from config import settings


# Try to import firecrawl
try:
    from firecrawl import FirecrawlApp
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False

# Max chars to return (matches existing summarizer input limit)
MAX_CONTENT_LENGTH = 4000

# Singleton instance
_instance: Optional["FirecrawlService"] = None


class FirecrawlService:
    """Wrapper around Firecrawl SDK for scraping and search."""

    def __init__(self):
        api_key = settings.firecrawl_api_key
        if not api_key or not FIRECRAWL_AVAILABLE:
            self._client = None
        else:
            self._client = FirecrawlApp(api_key=api_key)
        self._quota_exhausted = False

    @property
    def available(self) -> bool:
        return self._client is not None and not self._quota_exhausted

    def _extract_markdown(self, doc) -> str:
        """Extract markdown from a scrape result (handles dict or object)."""
        if isinstance(doc, dict):
            return doc.get("markdown", "") or ""
        return getattr(doc, "markdown", "") or ""

    def _extract_metadata(self, doc) -> dict:
        """Extract metadata from a scrape result (handles dict or object)."""
        if isinstance(doc, dict):
            return doc.get("metadata", {}) or {}
        meta = getattr(doc, "metadata", None)
        if meta is None:
            return {}
        if isinstance(meta, dict):
            return meta
        # Pydantic model — convert to dict
        return meta.dict() if hasattr(meta, "dict") else {}

    def scrape_article(self, url: str, max_length: int = MAX_CONTENT_LENGTH) -> Optional[str]:
        """Scrape a single URL and return markdown content.

        Returns None if Firecrawl is not configured or the call fails.
        """
        if not self._client:
            return None
        try:
            doc = self._client.scrape(url, formats=["markdown"])
            markdown = self._extract_markdown(doc)
            if markdown:
                return markdown[:max_length]
            return None
        except Exception as e:
            error_msg = str(e)
            if "Payment Required" in error_msg or "Insufficient credits" in error_msg:
                if not self._quota_exhausted:
                    print(f"  ⚠ Firecrawl credits exhausted — skipping remaining scrapes")
                    self._quota_exhausted = True
                return None
            print(f"  ⚠ Firecrawl scrape failed for {url}: {e}")
            return None

    def batch_scrape(self, urls: list[str]) -> dict[str, str]:
        """Scrape multiple URLs and return {url: markdown} dict.

        URLs that fail are omitted from the result.
        """
        if not self._client or not urls or self._quota_exhausted:
            return {}

        results = {}
        try:
            docs = self._client.batch_scrape(urls, formats=["markdown"])
            for doc in docs:
                meta = self._extract_metadata(doc)
                source_url = meta.get("sourceURL", "") or meta.get("url", "")
                markdown = self._extract_markdown(doc)
                if source_url and markdown:
                    results[source_url] = markdown[:MAX_CONTENT_LENGTH]
        except Exception as e:
            error_msg = str(e)
            if "Payment Required" in error_msg or "Insufficient credits" in error_msg:
                if not self._quota_exhausted:
                    print(f"  ⚠ Firecrawl credits exhausted — skipping remaining scrapes")
                    self._quota_exhausted = True
                return results
            print(f"  ⚠ Firecrawl batch scrape failed: {e}")
            # Fall back to individual scrapes
            for url in urls:
                text = self.scrape_article(url)
                if text:
                    results[url] = text
                if self._quota_exhausted:
                    break

        return results

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Search the web and return results with content.

        Returns list of {url, title, markdown} dicts.
        """
        if not self._client:
            return []
        try:
            # Pass only the query — limit via kwargs may not be supported
            # in all SDK versions
            search_data = self._client.search(query)
            results = []
            # Handle both list and object responses across SDK versions
            items = search_data
            if hasattr(search_data, "data"):
                items = search_data.data or []
            elif hasattr(search_data, "web"):
                items = search_data.web or []
            elif not isinstance(search_data, list):
                items = []
            for item in items:
                if isinstance(item, dict):
                    url = item.get("url", "")
                    title = item.get("title", "")
                    markdown = (item.get("markdown", "") or "")[:MAX_CONTENT_LENGTH]
                else:
                    url = getattr(item, "url", "")
                    title = getattr(item, "title", "")
                    markdown = (getattr(item, "markdown", "") or "")[:MAX_CONTENT_LENGTH]
                if url:
                    results.append({"url": url, "title": title, "markdown": markdown})
                if len(results) >= limit:
                    break
            return results
        except Exception as e:
            print(f"  ⚠ Firecrawl search failed for '{query}': {e}")
            return []


def get_firecrawl_service() -> FirecrawlService:
    """Get or create the FirecrawlService singleton."""
    global _instance
    if _instance is None:
        _instance = FirecrawlService()
    return _instance
