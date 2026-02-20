"""Topic clustering service using KMeans and Claude for label generation."""

import os
import uuid
from typing import Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class TopicClusterer:
    """Cluster items by topic using embeddings and generate AI-powered labels."""

    def __init__(
        self,
        min_clusters: int = 2,
        max_clusters: int = 10,
        min_cluster_size: int = 2,
        anthropic_api_key: Optional[str] = None,
    ):
        """Initialize the topic clusterer.

        Args:
            min_clusters: Minimum number of clusters to consider.
            max_clusters: Maximum number of clusters to consider.
            min_cluster_size: Minimum items per cluster.
            anthropic_api_key: API key for Claude (uses ANTHROPIC_API_KEY env var if not provided).
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size

        if ANTHROPIC_AVAILABLE:
            api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
            else:
                self.client = None
        else:
            self.client = None

    def find_optimal_k(self, embeddings: np.ndarray) -> int:
        """Find optimal number of clusters using silhouette score.

        Args:
            embeddings: Array of embeddings (n_samples, n_features).

        Returns:
            Optimal number of clusters.
        """
        n_samples = len(embeddings)
        if n_samples < self.min_clusters + 1:
            return max(1, n_samples // self.min_cluster_size)

        max_k = min(self.max_clusters, n_samples // self.min_cluster_size)
        if max_k < self.min_clusters:
            return self.min_clusters

        best_k = self.min_clusters
        best_score = -1

        for k in range(self.min_clusters, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue

        return best_k

    def cluster_embeddings(
        self,
        embeddings: list[list[float]],
        n_clusters: Optional[int] = None,
    ) -> tuple[list[int], dict[int, np.ndarray]]:
        """Cluster embeddings using KMeans.

        Args:
            embeddings: List of embedding vectors.
            n_clusters: Number of clusters (auto-detected if None).

        Returns:
            Tuple of (cluster_labels, centroids_dict).
        """
        if not embeddings:
            return [], {}

        embeddings_array = np.array(embeddings)

        if n_clusters is None:
            n_clusters = self.find_optimal_k(embeddings_array)

        # Ensure we don't have more clusters than samples
        n_clusters = min(n_clusters, len(embeddings))

        if n_clusters <= 1:
            return [0] * len(embeddings), {0: np.mean(embeddings_array, axis=0)}

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_array)

        centroids = {i: kmeans.cluster_centers_[i] for i in range(n_clusters)}

        return labels.tolist(), centroids

    def compute_confidence(
        self,
        embedding: np.ndarray,
        centroid: np.ndarray,
    ) -> float:
        """Compute confidence score for cluster assignment.

        Args:
            embedding: Item embedding vector.
            centroid: Cluster centroid vector.

        Returns:
            Confidence score (0.0-1.0).
        """
        # Cosine similarity
        dot_product = np.dot(embedding, centroid)
        norm_a = np.linalg.norm(embedding)
        norm_b = np.linalg.norm(centroid)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)
        # Clamp to 0-1 range, cast to plain float for DB compatibility
        return float(max(0.0, min(1.0, (similarity + 1) / 2)))

    def generate_cluster_label(self, items: list[dict]) -> str:
        """Generate a topic label using Claude.

        Args:
            items: List of items with 'title' keys.

        Returns:
            Generated topic label (3-5 words).
        """
        if not items:
            return "Miscellaneous"

        titles = [item.get("title", "")[:100] for item in items[:10]]
        titles_text = "\n".join(f"- {t}" for t in titles if t)

        if not self.client:
            # Fallback: extract common words
            return self._extract_keywords(titles)

        try:
            message = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=50,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Given these news article titles about a common topic:
{titles_text}

Generate a concise topic label (3-5 words) that captures the theme.
Respond with ONLY the label, nothing else.""",
                    }
                ],
            )
            label = message.content[0].text.strip()
            # Clean up the label
            label = label.strip('"\'')
            return label[:255] if label else "General News"
        except Exception as e:
            print(f"  ⚠ Claude label generation failed: {e}")
            return self._extract_keywords(titles)

    def generate_cluster_summary(self, items: list[dict]) -> str:
        """Generate a cross-source synthesis summary using Claude.

        Args:
            items: List of items with 'title' and 'summary' keys.

        Returns:
            Synthesized summary (2-3 sentences).
        """
        if not items:
            return ""

        summaries = []
        for item in items[:5]:
            title = item.get("title", "")[:100]
            summary = item.get("summary", "")[:200]
            source = item.get("source", "Unknown")
            if title:
                summaries.append(f"[{source}] {title}: {summary}")

        if not summaries:
            return ""

        summaries_text = "\n".join(summaries)

        if not self.client:
            # Fallback: return first summary
            return items[0].get("summary", "")[:500] if items else ""

        try:
            message = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                messages=[
                    {
                        "role": "user",
                        "content": f"""These articles from different sources cover the same topic:
{summaries_text}

Write a 2-3 sentence synthesis highlighting key insights and any differing perspectives.
Respond with ONLY the synthesis, nothing else.""",
                    }
                ],
            )
            return message.content[0].text.strip()[:1000]
        except Exception as e:
            print(f"  ⚠ Claude summary generation failed: {e}")
            return items[0].get("summary", "")[:500] if items else ""

    def _extract_keywords(self, titles: list[str]) -> str:
        """Extract common keywords as fallback label.

        Args:
            titles: List of article titles.

        Returns:
            Keyword-based label.
        """
        # Simple word frequency approach
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "each", "few",
            "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "just",
            "and", "but", "if", "or", "because", "until", "while", "this",
            "that", "these", "those", "what", "which", "who", "whom",
            "new", "says", "said", "its", "it", "about", "over", "out",
        }

        word_counts = {}
        for title in titles:
            words = title.lower().split()
            for word in words:
                word = "".join(c for c in word if c.isalnum())
                if word and len(word) > 2 and word not in stop_words:
                    word_counts[word] = word_counts.get(word, 0) + 1

        if not word_counts:
            return "General News"

        # Get top 3 words
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        label = " ".join(word.capitalize() for word, _ in top_words)
        return label or "General News"

    def generate_cluster_id(self) -> str:
        """Generate a unique cluster ID.

        Returns:
            UUID string for cluster identification.
        """
        return str(uuid.uuid4())[:8]

    def cluster_items(
        self,
        items: list[dict],
        embeddings: list[list[float]],
        n_clusters: Optional[int] = None,
    ) -> list[dict]:
        """Cluster items and enrich with cluster information.

        Args:
            items: List of item dicts with 'id', 'title', 'summary'.
            embeddings: Corresponding embeddings for each item.
            n_clusters: Number of clusters (auto-detected if None).

        Returns:
            List of cluster dicts with 'cluster_id', 'label', 'summary', 'items'.
        """
        if not items or not embeddings or len(items) != len(embeddings):
            return []

        # Run clustering
        labels, centroids = self.cluster_embeddings(embeddings, n_clusters)

        # Group items by cluster
        clusters_dict = {}
        for i, (item, label) in enumerate(zip(items, labels)):
            if label not in clusters_dict:
                clusters_dict[label] = {
                    "cluster_id": self.generate_cluster_id(),
                    "items": [],
                    "embeddings": [],
                }
            clusters_dict[label]["items"].append(item)
            clusters_dict[label]["embeddings"].append(embeddings[i])

        # Generate labels and summaries for each cluster
        clusters = []
        for label_idx, cluster_data in clusters_dict.items():
            cluster_items = cluster_data["items"]
            cluster_embeddings = cluster_data["embeddings"]
            centroid = centroids.get(label_idx, np.mean(cluster_embeddings, axis=0))

            # Compute confidence for each item
            for i, item in enumerate(cluster_items):
                item["cluster_id"] = cluster_data["cluster_id"]
                item["cluster_confidence"] = self.compute_confidence(
                    np.array(cluster_embeddings[i]), centroid
                )

            # Generate label and summary
            cluster_label = self.generate_cluster_label(cluster_items)
            cluster_summary = self.generate_cluster_summary(cluster_items)

            # Update items with label
            for item in cluster_items:
                item["cluster_label"] = cluster_label

            # Calculate average score
            scores = [item.get("score", 0) or 0 for item in cluster_items]
            avg_score = sum(scores) / len(scores) if scores else 0

            clusters.append({
                "cluster_id": cluster_data["cluster_id"],
                "label": cluster_label,
                "summary": cluster_summary,
                "item_count": len(cluster_items),
                "avg_score": avg_score,
                "items": cluster_items,
            })

        # Sort clusters by item count (largest first)
        clusters.sort(key=lambda x: x["item_count"], reverse=True)

        return clusters
