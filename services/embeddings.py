"""OpenAI embedding service with retry logic."""

import os
from typing import Optional

import tiktoken
from openai import OpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Constants
MODEL = "text-embedding-3-small"
DIMENSIONS = 1536
MAX_TOKENS = 8191  # Model limit for text-embedding-3-small


class EmbeddingService:
    """Service for generating embeddings using OpenAI's text-embedding-3-small model."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the embedding service.

        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = MODEL
        self.dimensions = DIMENSIONS
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy-load the tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = tiktoken.encoding_for_model(self.model)
        return self._tokenizer

    def truncate_text(self, text: str, max_tokens: int = MAX_TOKENS) -> str:
        """Truncate text to fit within token limit.

        Args:
            text: Text to truncate.
            max_tokens: Maximum number of tokens allowed.

        Returns:
            Truncated text that fits within token limit.
        """
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    def get_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        # Truncate if needed
        text = self.truncate_text(text.strip())

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
        )
        return response.data[0].embedding

    def batch_embed(self, texts: list[str], batch_size: int = 2048) -> list[list[float]]:
        """Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts per API call (max 2048).

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            RateLimitError: If API quota/rate limit is exceeded (fails fast, no retry).
            Exception: Other API errors after 3 retry attempts.
        """
        if not texts:
            return []

        # Truncate and filter empty texts
        processed_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                processed_texts.append(self.truncate_text(text.strip()))
                valid_indices.append(i)

        if not processed_texts:
            return [[] for _ in texts]

        all_embeddings = []

        # Process in batches
        for i in range(0, len(processed_texts), batch_size):
            batch = processed_texts[i:i + batch_size]
            try:
                response = self._call_embedding_api(batch)
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except RateLimitError:
                # Fail fast on rate limit/quota - no point retrying
                raise

        # Map back to original indices, filling in empty lists for invalid texts
        result = [[] for _ in texts]
        for idx, embedding in zip(valid_indices, all_embeddings):
            result[idx] = embedding

        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    def _call_embedding_api(self, texts: list[str]):
        """Call the embedding API with retry logic for transient errors.

        RateLimitError is not retried as quota issues won't resolve with retries.
        """
        try:
            return self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
            )
        except RateLimitError:
            # Don't retry rate limit errors - they won't succeed
            raise

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text.

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens.
        """
        return len(self.tokenizer.encode(text))
