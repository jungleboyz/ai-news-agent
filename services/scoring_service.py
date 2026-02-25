"""Unified scoring service used by all agents.

Consolidates norm(), keyword scoring, semantic scoring, batch scoring,
and single-with-embedding scoring that were duplicated across
agent.py, podcast_agent.py, video_agent.py, and web_scraper_agent.py.
"""
import re
from typing import List, Optional, Tuple

from services.service_registry import (
    get_embedding_service,
    get_semantic_scorer,
    is_quota_exceeded,
    set_quota_exceeded,
)

# Enable/disable semantic scoring (set to False to use keyword scoring)
USE_SEMANTIC_SCORING = True

# Superset of keywords used across all agents
AI_KEYWORDS = [
    # From agent.py / podcast_agent.py / web_scraper_agent.py
    "genai", "generative ai", "llm", "agent", "agents",
    "openai", "anthropic", "gemini", "mistral", "claude",
    "cursor", "copilot", "aider", "enterprise", "bank",
    "marketing", "automation", "workflow", "funding", "acquisition",
    # From video_agent.py (additional)
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "neural network", "gpt", "chatgpt", "large language model",
    "transformer", "diffusion", "stable diffusion", "midjourney", "dall-e",
    "ai agent", "rag", "retrieval", "embedding",
    "fine-tuning", "prompt engineering", "inference", "training",
    "computer vision", "nlp", "natural language", "speech recognition",
    "text-to-speech", "text-to-image", "multimodal", "foundation model",
]


def norm(text: str) -> str:
    """Lowercase + collapse whitespace so matching is consistent."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def score_keywords(title: str, description: str = "", transcript: str = "") -> int:
    """Keyword-based scoring (superset of all agent keyword lists).

    Title matches get higher weight (3 vs 1) consistent with video_agent behaviour.
    """
    title_lower = norm(title)
    rest = norm(f"{description} {transcript}")
    score = 0
    for kw in AI_KEYWORDS:
        if kw in title_lower:
            score += 3
        elif kw in rest:
            score += 1
    return score


def score_semantic(title: str, description: str = "", transcript: str = "") -> int:
    """Score text using semantic scoring, falling back to keywords on failure."""
    if not USE_SEMANTIC_SCORING or is_quota_exceeded():
        return score_keywords(title, description, transcript)

    try:
        scorer = get_semantic_scorer()
        text = f"{title} {description} {transcript}".strip()
        semantic_score = scorer.score_text(text)
        return scorer.score_to_int(semantic_score, scale=10)
    except Exception as e:
        _handle_scoring_error(e)
        return score_keywords(title, description, transcript)


def score_single_with_embedding(
    title: str, description: str = "", transcript: str = ""
) -> Tuple[int, Optional[float], Optional[list]]:
    """Score a single item and return (int_score, float_score, embedding).

    Used by podcast_agent and video_agent which need the embedding back
    for storage in ChromaDB.
    """
    if not USE_SEMANTIC_SCORING or is_quota_exceeded():
        return score_keywords(title, description, transcript), None, None

    try:
        embedding_service = get_embedding_service()
        scorer = get_semantic_scorer()

        text = f"{title} {description} {transcript}".strip()
        embedding = embedding_service.get_embedding(text)
        semantic_score = scorer.score_item(embedding)
        int_score = scorer.score_to_int(semantic_score, scale=10)

        return int_score, semantic_score, embedding

    except Exception as e:
        _handle_scoring_error(e)
        return score_keywords(title, description, transcript), None, None


def score_items_batch(items: List[dict]) -> List[dict]:
    """Score multiple items using batch embedding for efficiency.

    Args:
        items: List of items with 'title', 'summary', and 'id' keys.

    Returns:
        Items with 'score', 'semantic_score', and 'embedding' added.
    """
    total = len(items)

    if not USE_SEMANTIC_SCORING or not items or is_quota_exceeded():
        print(f"  Scoring {total} items with keyword matching...")
        for i, item in enumerate(items):
            item["score"] = score_keywords(item["title"], item.get("summary", ""))
            item["semantic_score"] = None
            item["embedding"] = None
            if (i + 1) % 200 == 0:
                print(f"    Progress: {i + 1}/{total} items scored")
        print(f"  Keyword scoring complete")
        return items

    try:
        print(f"  Generating embeddings for {total} items...")
        print(f"    (This may take a moment - calling OpenAI API)")

        embedding_service = get_embedding_service()
        scorer = get_semantic_scorer()

        texts = [f"{it['title']} {it.get('summary', '')}".strip() for it in items]
        embeddings = embedding_service.batch_embed(texts)

        print(f"  Embeddings received, scoring items...")

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
                item["score"] = score_keywords(item["title"], item.get("summary", ""))
                item["semantic_score"] = None
                item["embedding"] = None
                fallback_count += 1

        print(f"  Semantic scoring complete: {scored_count} semantic, {fallback_count} keyword fallback")
        return items

    except Exception as e:
        _handle_scoring_error(e)

        print(f"  Scoring {total} items with keyword matching...")
        for i, item in enumerate(items):
            item["score"] = score_keywords(item["title"], item.get("summary", ""))
            item["semantic_score"] = None
            item["embedding"] = None
            if (i + 1) % 200 == 0:
                print(f"    Progress: {i + 1}/{total} items scored")
        print(f"  Keyword scoring complete")
        return items


def _handle_scoring_error(e: Exception) -> None:
    """Check if error is quota/rate-related and flip the global flag."""
    error_msg = str(e)
    if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
        if not is_quota_exceeded():
            print(f"  OpenAI API quota/rate limit hit - switching to keyword scoring")
            set_quota_exceeded(True)
    else:
        print(f"  Semantic scoring failed, using keywords: {e}")
