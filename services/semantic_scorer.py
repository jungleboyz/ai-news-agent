"""Semantic scoring using cosine similarity with user interests."""

import numpy as np
from typing import Optional

from .embeddings import EmbeddingService


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity score between -1 and 1.
    """
    a_arr = np.array(a)
    b_arr = np.array(b)

    dot_product = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


class SemanticScorer:
    """Score items based on semantic similarity to user interests."""

    DEFAULT_INTERESTS = [
        "generative AI and large language models",
        "AI agents and autonomous systems",
        "OpenAI, Anthropic, Google Gemini, and Mistral developments",
        "AI coding assistants like Cursor, Copilot, and Aider",
        "enterprise AI adoption and automation",
        "AI startups, funding rounds, and acquisitions",
        "AI in marketing, banking, and workflow automation",
    ]

    def __init__(
        self,
        user_interests: Optional[list[str]] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """Initialize the semantic scorer.

        Args:
            user_interests: List of interest descriptions. Uses defaults if not provided.
            embedding_service: EmbeddingService instance. Creates one if not provided.
        """
        self.interests = user_interests or self.DEFAULT_INTERESTS
        self.embedding_service = embedding_service or EmbeddingService()
        self._interest_embedding: Optional[list[float]] = None

    @property
    def interest_embedding(self) -> list[float]:
        """Get or generate the combined interest embedding."""
        if self._interest_embedding is None:
            self._interest_embedding = self.generate_interest_embedding()
        return self._interest_embedding

    def generate_interest_embedding(self) -> list[float]:
        """Generate a combined embedding representing all user interests.

        Creates a single embedding by averaging embeddings of all interest descriptions.

        Returns:
            Combined interest embedding vector.
        """
        if not self.interests:
            raise ValueError("No interests defined for semantic scoring")

        # Embed all interests
        embeddings = self.embedding_service.batch_embed(self.interests)

        # Filter out any empty embeddings
        valid_embeddings = [e for e in embeddings if e]

        if not valid_embeddings:
            raise ValueError("Failed to generate embeddings for interests")

        # Average the embeddings
        avg_embedding = np.mean(valid_embeddings, axis=0)

        # Normalize to unit vector for cosine similarity
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        return avg_embedding.tolist()

    def score_item(
        self,
        item_embedding: list[float],
        min_score: float = 0.0,
        max_score: float = 1.0,
    ) -> float:
        """Score an item based on semantic similarity to user interests.

        Args:
            item_embedding: The item's embedding vector.
            min_score: Minimum score to return.
            max_score: Maximum score to return.

        Returns:
            Similarity score between min_score and max_score.
        """
        if not item_embedding:
            return min_score

        similarity = cosine_similarity(self.interest_embedding, item_embedding)

        # Clamp to valid range
        return max(min_score, min(max_score, similarity))

    def score_text(self, text: str) -> float:
        """Score text content by generating its embedding first.

        Args:
            text: Text content to score.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        if not text or not text.strip():
            return 0.0

        embedding = self.embedding_service.get_embedding(text)
        return self.score_item(embedding)

    def score_items_batch(
        self,
        embeddings: list[list[float]],
    ) -> list[float]:
        """Score multiple items at once.

        Args:
            embeddings: List of item embedding vectors.

        Returns:
            List of similarity scores.
        """
        return [self.score_item(emb) for emb in embeddings]

    def is_relevant(self, score: float, threshold: float = 0.3) -> bool:
        """Determine if a score indicates relevance.

        Args:
            score: The semantic similarity score.
            threshold: Minimum score to consider relevant.

        Returns:
            True if the item is considered relevant.
        """
        return score >= threshold

    def score_to_int(self, score: float, scale: int = 10) -> int:
        """Convert a float score to an integer for backward compatibility.

        Args:
            score: Float score between 0.0 and 1.0.
            scale: Scale factor (e.g., 10 means 0-10 range).

        Returns:
            Integer score.
        """
        return int(round(score * scale))
