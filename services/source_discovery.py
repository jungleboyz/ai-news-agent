"""Source discovery service for HackerNews and Reddit."""
import asyncio
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlparse

import httpx


# AI-related keywords for filtering
AI_KEYWORDS = [
    "ai", "artificial intelligence", "machine learning", "ml", "deep learning",
    "neural network", "gpt", "llm", "large language model", "chatgpt", "openai",
    "anthropic", "claude", "gemini", "mistral", "llama", "transformer",
    "diffusion", "stable diffusion", "midjourney", "dall-e", "imagen",
    "nlp", "natural language", "computer vision", "robotics", "automation",
    "agi", "alignment", "safety", "rlhf", "fine-tuning", "embedding",
    "vector", "rag", "retrieval", "agent", "copilot", "cursor", "coding assistant",
]

# Compile regex for efficient matching
AI_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(kw) for kw in AI_KEYWORDS) + r')\b',
    re.IGNORECASE
)


@dataclass
class DiscoveredItem:
    """A discovered item from HN or Reddit."""
    title: str
    url: str
    source: str  # "hackernews" or "reddit"
    source_id: str  # HN story ID or Reddit post ID
    score: int
    comments: int
    created_at: datetime
    subreddit: Optional[str] = None  # For Reddit items
    domain: Optional[str] = None
    ai_relevance: float = 0.0  # 0-1 score based on keyword matches


class HackerNewsScraper:
    """Scraper for HackerNews using the official Firebase API."""

    BASE_URL = "https://hacker-news.firebaseio.com/v0"

    def __init__(self, min_score: int = 10, max_items: int = 100):
        self.min_score = min_score
        self.max_items = max_items
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        await self.client.aclose()

    async def _get_item(self, item_id: int) -> Optional[dict]:
        """Fetch a single HN item."""
        try:
            response = await self.client.get(f"{self.BASE_URL}/item/{item_id}.json")
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None

    async def _get_story_ids(self, endpoint: str = "topstories") -> list[int]:
        """Get story IDs from an endpoint (topstories, newstories, beststories)."""
        try:
            response = await self.client.get(f"{self.BASE_URL}/{endpoint}.json")
            if response.status_code == 200:
                return response.json()[:self.max_items * 2]  # Fetch extra for filtering
        except Exception:
            pass
        return []

    def _calculate_ai_relevance(self, title: str, url: str = "") -> float:
        """Calculate AI relevance score based on keyword matches."""
        text = f"{title} {url}".lower()
        matches = AI_PATTERN.findall(text)
        if not matches:
            return 0.0
        # Score based on number of unique keywords found
        unique_matches = len(set(m.lower() for m in matches))
        return min(1.0, unique_matches * 0.25)

    async def fetch_top_ai_stories(self) -> list[DiscoveredItem]:
        """Fetch top stories filtered for AI relevance."""
        story_ids = await self._get_story_ids("topstories")

        # Fetch stories in parallel batches
        items = []
        batch_size = 20

        for i in range(0, len(story_ids), batch_size):
            batch_ids = story_ids[i:i + batch_size]
            tasks = [self._get_item(sid) for sid in batch_ids]
            results = await asyncio.gather(*tasks)

            for story in results:
                if not story or story.get("type") != "story":
                    continue
                if story.get("score", 0) < self.min_score:
                    continue
                if not story.get("url"):  # Skip Ask HN, Show HN without URLs
                    continue

                title = story.get("title", "")
                url = story.get("url", "")
                relevance = self._calculate_ai_relevance(title, url)

                if relevance > 0:
                    domain = urlparse(url).netloc if url else None
                    items.append(DiscoveredItem(
                        title=title,
                        url=url,
                        source="hackernews",
                        source_id=str(story["id"]),
                        score=story.get("score", 0),
                        comments=story.get("descendants", 0),
                        created_at=datetime.fromtimestamp(story.get("time", 0)),
                        domain=domain,
                        ai_relevance=relevance,
                    ))

            if len(items) >= self.max_items:
                break

            # Rate limiting
            await asyncio.sleep(0.1)

        # Sort by relevance * score
        items.sort(key=lambda x: x.ai_relevance * x.score, reverse=True)
        return items[:self.max_items]

    async def fetch_new_ai_stories(self) -> list[DiscoveredItem]:
        """Fetch new stories filtered for AI relevance."""
        story_ids = await self._get_story_ids("newstories")

        items = []
        batch_size = 20

        for i in range(0, len(story_ids), batch_size):
            batch_ids = story_ids[i:i + batch_size]
            tasks = [self._get_item(sid) for sid in batch_ids]
            results = await asyncio.gather(*tasks)

            for story in results:
                if not story or story.get("type") != "story":
                    continue
                if not story.get("url"):
                    continue

                title = story.get("title", "")
                url = story.get("url", "")
                relevance = self._calculate_ai_relevance(title, url)

                if relevance > 0:
                    domain = urlparse(url).netloc if url else None
                    items.append(DiscoveredItem(
                        title=title,
                        url=url,
                        source="hackernews",
                        source_id=str(story["id"]),
                        score=story.get("score", 0),
                        comments=story.get("descendants", 0),
                        created_at=datetime.fromtimestamp(story.get("time", 0)),
                        domain=domain,
                        ai_relevance=relevance,
                    ))

            if len(items) >= self.max_items:
                break

            await asyncio.sleep(0.1)

        items.sort(key=lambda x: x.ai_relevance, reverse=True)
        return items[:self.max_items]


class RedditScraper:
    """Scraper for Reddit using the JSON API (no auth required)."""

    BASE_URL = "https://www.reddit.com"

    # AI-focused subreddits
    SUBREDDITS = [
        "MachineLearning",
        "artificial",
        "LocalLLaMA",
        "singularity",
        "OpenAI",
        "StableDiffusion",
        "ChatGPT",
        "ClaudeAI",
        "Anthropic",
        "deeplearning",
        "learnmachinelearning",
        "MLQuestions",
        "LanguageTechnology",
        "ArtificialInteligence",
    ]

    def __init__(self, min_score: int = 20, max_items: int = 50):
        self.min_score = min_score
        self.max_items = max_items
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": "AI-News-Agent/1.0 (Source Discovery Bot)",
            }
        )

    async def close(self):
        await self.client.aclose()

    def _calculate_ai_relevance(self, title: str, selftext: str = "") -> float:
        """Calculate AI relevance score."""
        text = f"{title} {selftext}".lower()
        matches = AI_PATTERN.findall(text)
        if not matches:
            # Posts from AI subreddits are inherently relevant
            return 0.5
        unique_matches = len(set(m.lower() for m in matches))
        return min(1.0, 0.5 + unique_matches * 0.15)

    async def _fetch_subreddit(self, subreddit: str, sort: str = "hot", limit: int = 25) -> list[dict]:
        """Fetch posts from a subreddit."""
        try:
            url = f"{self.BASE_URL}/r/{subreddit}/{sort}.json?limit={limit}"
            response = await self.client.get(url)
            if response.status_code == 200:
                data = response.json()
                return data.get("data", {}).get("children", [])
        except Exception:
            pass
        return []

    async def fetch_ai_posts(self, sort: str = "hot") -> list[DiscoveredItem]:
        """Fetch posts from AI subreddits."""
        items = []

        for subreddit in self.SUBREDDITS:
            posts = await self._fetch_subreddit(subreddit, sort, limit=25)

            for post_data in posts:
                post = post_data.get("data", {})

                # Skip stickied posts, self posts without content
                if post.get("stickied"):
                    continue

                score = post.get("score", 0)
                if score < self.min_score:
                    continue

                title = post.get("title", "")
                selftext = post.get("selftext", "")
                url = post.get("url", "")

                # For self posts, use the Reddit URL
                if post.get("is_self"):
                    url = f"https://reddit.com{post.get('permalink', '')}"

                relevance = self._calculate_ai_relevance(title, selftext)
                domain = urlparse(url).netloc if url else None

                created_utc = post.get("created_utc", 0)

                items.append(DiscoveredItem(
                    title=title,
                    url=url,
                    source="reddit",
                    source_id=post.get("id", ""),
                    score=score,
                    comments=post.get("num_comments", 0),
                    created_at=datetime.fromtimestamp(created_utc) if created_utc else datetime.now(),
                    subreddit=subreddit,
                    domain=domain,
                    ai_relevance=relevance,
                ))

            # Rate limiting - Reddit is sensitive to rapid requests
            await asyncio.sleep(1.0)

        # Sort by relevance * log(score) to balance popularity and relevance
        import math
        items.sort(key=lambda x: x.ai_relevance * math.log(max(x.score, 1) + 1), reverse=True)
        return items[:self.max_items]


class SourceDiscoveryService:
    """Main service for discovering new AI content sources."""

    def __init__(self):
        self.hn_scraper = HackerNewsScraper()
        self.reddit_scraper = RedditScraper()

    async def close(self):
        await self.hn_scraper.close()
        await self.reddit_scraper.close()

    async def discover_all(self) -> dict[str, list[DiscoveredItem]]:
        """Run all scrapers and return discovered items."""
        hn_top, hn_new, reddit = await asyncio.gather(
            self.hn_scraper.fetch_top_ai_stories(),
            self.hn_scraper.fetch_new_ai_stories(),
            self.reddit_scraper.fetch_ai_posts(),
        )

        return {
            "hackernews_top": hn_top,
            "hackernews_new": hn_new,
            "reddit": reddit,
        }

    async def discover_hackernews(self) -> list[DiscoveredItem]:
        """Discover AI stories from HackerNews."""
        top = await self.hn_scraper.fetch_top_ai_stories()
        new = await self.hn_scraper.fetch_new_ai_stories()

        # Dedupe by URL
        seen_urls = set()
        combined = []
        for item in top + new:
            if item.url not in seen_urls:
                seen_urls.add(item.url)
                combined.append(item)

        return combined

    async def discover_reddit(self) -> list[DiscoveredItem]:
        """Discover AI posts from Reddit."""
        return await self.reddit_scraper.fetch_ai_posts()

    def extract_domains(self, items: list[DiscoveredItem]) -> dict[str, int]:
        """Extract and count domains from discovered items."""
        domain_counts = {}
        for item in items:
            if item.domain:
                domain = item.domain.replace("www.", "")
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        return dict(sorted(domain_counts.items(), key=lambda x: x[1], reverse=True))


# Synchronous wrapper for non-async contexts
def run_discovery() -> dict[str, list[DiscoveredItem]]:
    """Run source discovery synchronously."""
    async def _run():
        service = SourceDiscoveryService()
        try:
            return await service.discover_all()
        finally:
            await service.close()

    return asyncio.run(_run())


if __name__ == "__main__":
    # Test the scrapers
    import json

    async def main():
        service = SourceDiscoveryService()
        try:
            print("Discovering AI content from HackerNews and Reddit...")
            results = await service.discover_all()

            for source, items in results.items():
                print(f"\n=== {source.upper()} ({len(items)} items) ===")
                for item in items[:5]:
                    print(f"  [{item.score}] {item.title[:60]}...")
                    print(f"       URL: {item.url[:60]}...")
                    print(f"       Relevance: {item.ai_relevance:.2f}")

            # Show top domains
            all_items = [item for items in results.values() for item in items]
            domains = service.extract_domains(all_items)
            print(f"\n=== TOP DOMAINS ===")
            for domain, count in list(domains.items())[:20]:
                print(f"  {domain}: {count}")

        finally:
            await service.close()

    asyncio.run(main())
