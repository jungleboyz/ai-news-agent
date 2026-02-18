"""Personalization service for learning user preferences."""
import json
import uuid
from datetime import datetime
from typing import Optional

import numpy as np

from services.embeddings import EmbeddingService
from services.vector_store import VectorStore


# Action weights for preference learning
ACTION_WEIGHTS = {
    "click": 1.0,
    "save": 3.0,
    "skip": -0.5,
    "hide": -2.0,
}


class PersonalizationService:
    """Service for tracking interactions and computing personalized scores."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        # Initialize services lazily to handle missing API keys gracefully
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._services_initialized = False

    def _init_services(self):
        """Lazily initialize embedding services."""
        if self._services_initialized:
            return

        try:
            if self._embedding_service is None:
                self._embedding_service = EmbeddingService()
            if self._vector_store is None:
                self._vector_store = VectorStore()
        except Exception:
            # Services unavailable (no API key, etc.)
            pass

        self._services_initialized = True

    @property
    def embedding_service(self) -> Optional[EmbeddingService]:
        self._init_services()
        return self._embedding_service

    @property
    def vector_store(self) -> Optional[VectorStore]:
        self._init_services()
        return self._vector_store

    def generate_user_id(self) -> str:
        """Generate a new unique user ID for cookie-based identification."""
        return str(uuid.uuid4())

    def get_or_create_user(self, db, user_id: str) -> "UserProfile":
        """Get existing user or create new one."""
        from web.models import UserProfile

        user = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if not user:
            user = UserProfile(user_id=user_id)
            db.add(user)
            db.commit()
            db.refresh(user)
        return user

    def track_interaction(self, db, user_id: str, item_id: int, action: str) -> None:
        """Record a user interaction with an item."""
        from web.models import Interaction, UserProfile

        if action not in ACTION_WEIGHTS:
            raise ValueError(f"Invalid action: {action}. Must be one of {list(ACTION_WEIGHTS.keys())}")

        # Ensure user exists
        self.get_or_create_user(db, user_id)

        # Create interaction record
        interaction = Interaction(
            user_id=user_id,
            item_id=item_id,
            action=action,
        )
        db.add(interaction)
        db.commit()

    def get_recent_interactions(self, db, user_id: str, limit: int = 100) -> list:
        """Get recent interactions for a user."""
        from web.models import Interaction

        return (
            db.query(Interaction)
            .filter(Interaction.user_id == user_id)
            .order_by(Interaction.created_at.desc())
            .limit(limit)
            .all()
        )

    def compute_preference_embedding(self, db, user_id: str) -> Optional[list[float]]:
        """
        Generate user preference embedding from interaction history.
        Uses weighted average of embeddings for interacted items.
        """
        from web.models import Item, UserProfile

        if not self.vector_store:
            return None  # Vector store not available

        interactions = self.get_recent_interactions(db, user_id, limit=100)
        if not interactions:
            return None

        weighted_embeddings = []
        total_positive_weight = 0

        for interaction in interactions:
            weight = ACTION_WEIGHTS.get(interaction.action, 0)

            # Get item embedding from ChromaDB
            item = db.query(Item).filter(Item.id == interaction.item_id).first()
            if not item or not item.embedding_id:
                continue

            try:
                # Determine item type for collection lookup
                item_type = item.type if item.type in ["news", "podcast", "video"] else "news"
                result = self.vector_store.get_item(item.embedding_id, item_type)
                if result and result.get("embedding"):
                    embedding = result["embedding"]
                    weighted_embeddings.append((embedding, weight))
                    if weight > 0:
                        total_positive_weight += weight
            except Exception:
                continue

        if not weighted_embeddings or total_positive_weight <= 0:
            return None  # Not enough positive signals

        # Compute weighted average
        preference = np.zeros(len(weighted_embeddings[0][0]))
        total_weight = sum(abs(w) for _, w in weighted_embeddings)

        for emb, w in weighted_embeddings:
            preference += np.array(emb) * w

        if total_weight > 0:
            preference = preference / total_weight

        # Normalize to unit vector
        norm = np.linalg.norm(preference)
        if norm > 0:
            preference = preference / norm

        return preference.tolist()

    def update_user_preference_embedding(self, db, user_id: str) -> Optional[list[float]]:
        """Compute and cache user preference embedding."""
        from web.models import UserProfile

        embedding = self.compute_preference_embedding(db, user_id)
        if embedding:
            user = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
            if user:
                user.preference_embedding = json.dumps(embedding)
                user.embedding_updated_at = datetime.utcnow()
                db.commit()
        return embedding

    def get_preference_embedding(self, db, user_id: str) -> Optional[list[float]]:
        """Get cached preference embedding or compute if stale."""
        from web.models import UserProfile

        user = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if not user:
            return None

        if user.preference_embedding:
            return json.loads(user.preference_embedding)

        # Compute and cache if not available
        return self.update_user_preference_embedding(db, user_id)

    def get_personalized_score(
        self,
        db,
        user_id: str,
        item_embedding: list[float],
        base_semantic_score: float = 0.0,
    ) -> float:
        """
        Compute personalized score combining semantic and preference scores.
        Formula: 0.6 * semantic_score + 0.4 * preference_score
        """
        preference_embedding = self.get_preference_embedding(db, user_id)

        if preference_embedding is None:
            return base_semantic_score

        # Compute cosine similarity with preference embedding
        item_vec = np.array(item_embedding)
        pref_vec = np.array(preference_embedding)

        dot_product = np.dot(item_vec, pref_vec)
        norm_item = np.linalg.norm(item_vec)
        norm_pref = np.linalg.norm(pref_vec)

        if norm_item == 0 or norm_pref == 0:
            preference_score = 0.0
        else:
            preference_score = dot_product / (norm_item * norm_pref)
            # Normalize to 0-1 range (cosine can be -1 to 1)
            preference_score = (preference_score + 1) / 2

        # Blend scores
        return 0.6 * base_semantic_score + 0.4 * preference_score

    def get_recommendations(self, db, user_id: str, limit: int = 20) -> list[dict]:
        """Get personalized item recommendations based on preference embedding."""
        from web.models import Item

        preference_embedding = self.get_preference_embedding(db, user_id)
        if preference_embedding is None or not self.vector_store:
            # Fall back to recent high-scoring items
            items = (
                db.query(Item)
                .filter(Item.score > 0)
                .order_by(Item.id.desc())
                .limit(limit)
                .all()
            )
            return [{"item": item, "score": item.score, "personalized": False} for item in items]

        # Search ChromaDB for similar items across all types
        try:
            results = self.vector_store.search_by_embedding(
                embedding=preference_embedding,
                item_type=None,  # Search all collections
                limit=limit,
            )
        except Exception:
            # Fallback if vector search fails
            items = (
                db.query(Item)
                .filter(Item.score > 0)
                .order_by(Item.id.desc())
                .limit(limit)
                .all()
            )
            return [{"item": item, "score": item.score, "personalized": False} for item in items]

        recommendations = []
        for result in results:
            item_id = result.get("metadata", {}).get("item_id")
            if item_id:
                item = db.query(Item).filter(Item.id == item_id).first()
                if item:
                    recommendations.append({
                        "item": item,
                        "similarity": result.get("similarity", 0),
                        "personalized": True,
                    })

        return recommendations

    # Preset management methods

    def create_preset(
        self,
        db,
        user_id: str,
        name: str,
        interests: list[str],
        activate: bool = False,
    ) -> "PreferencePreset":
        """Create a new preference preset from interest list."""
        from web.models import PreferencePreset

        # Ensure user exists
        self.get_or_create_user(db, user_id)

        # Generate embedding from interests (if service available)
        preset_embedding = None
        if self.embedding_service:
            interest_text = " ".join(interests)
            try:
                preset_embedding = self.embedding_service.get_embedding(interest_text)
            except Exception:
                preset_embedding = None

        # Deactivate other presets if activating this one
        if activate:
            db.query(PreferencePreset).filter(
                PreferencePreset.user_id == user_id
            ).update({"is_active": False})

        preset = PreferencePreset(
            user_id=user_id,
            name=name,
            interests=json.dumps(interests),
            is_active=activate,
            preset_embedding=json.dumps(preset_embedding) if preset_embedding else None,
        )
        db.add(preset)
        db.commit()
        db.refresh(preset)

        return preset

    def activate_preset(self, db, user_id: str, preset_id: int) -> bool:
        """Activate a preset and update user preference embedding."""
        from web.models import PreferencePreset, UserProfile

        # Deactivate all presets for this user
        db.query(PreferencePreset).filter(
            PreferencePreset.user_id == user_id
        ).update({"is_active": False})

        # Activate the selected preset
        preset = (
            db.query(PreferencePreset)
            .filter(PreferencePreset.id == preset_id, PreferencePreset.user_id == user_id)
            .first()
        )
        if not preset:
            return False

        preset.is_active = True

        # Update user preference embedding if preset has one
        if preset.preset_embedding:
            user = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
            if user:
                user.preference_embedding = preset.preset_embedding
                user.embedding_updated_at = datetime.utcnow()

        db.commit()
        return True

    def get_user_presets(self, db, user_id: str) -> list["PreferencePreset"]:
        """Get all presets for a user."""
        from web.models import PreferencePreset

        return (
            db.query(PreferencePreset)
            .filter(PreferencePreset.user_id == user_id)
            .order_by(PreferencePreset.created_at.desc())
            .all()
        )

    def delete_preset(self, db, user_id: str, preset_id: int) -> bool:
        """Delete a preset."""
        from web.models import PreferencePreset

        preset = (
            db.query(PreferencePreset)
            .filter(PreferencePreset.id == preset_id, PreferencePreset.user_id == user_id)
            .first()
        )
        if not preset:
            return False

        db.delete(preset)
        db.commit()
        return True


# Default presets for new users
DEFAULT_PRESETS = [
    {
        "name": "AI Research",
        "interests": [
            "machine learning research papers",
            "academic AI breakthroughs",
            "neural network architectures",
            "AI safety research",
            "arXiv preprints",
        ],
    },
    {
        "name": "AI Products",
        "interests": [
            "AI product launches",
            "AI startups and tools",
            "ChatGPT updates",
            "AI coding assistants",
            "consumer AI applications",
        ],
    },
    {
        "name": "AI Business",
        "interests": [
            "AI enterprise adoption",
            "AI funding and investments",
            "AI acquisitions",
            "AI company earnings",
            "AI market trends",
        ],
    },
    {
        "name": "AI Ethics",
        "interests": [
            "AI safety and alignment",
            "AI policy and regulation",
            "AI bias and fairness",
            "responsible AI development",
            "AI governance",
        ],
    },
    {
        "name": "Coding AI",
        "interests": [
            "AI code generation",
            "GitHub Copilot",
            "AI pair programming",
            "code completion models",
            "AI developer tools",
        ],
    },
]


def create_default_presets_for_user(db, user_id: str, personalization_service: PersonalizationService):
    """Create default presets for a new user."""
    existing = personalization_service.get_user_presets(db, user_id)
    if existing:
        return  # User already has presets

    for preset_data in DEFAULT_PRESETS:
        personalization_service.create_preset(
            db,
            user_id,
            preset_data["name"],
            preset_data["interests"],
            activate=False,
        )
