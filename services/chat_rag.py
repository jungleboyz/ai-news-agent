"""RAG (Retrieval Augmented Generation) and Chat service."""
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Generator, Optional

from anthropic import Anthropic

from services.vector_store import VectorStore


@dataclass
class ChatMessage:
    """A message in the chat conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = None
    sources: list = None  # Referenced items

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.sources is None:
            self.sources = []


class ChatRAGService:
    """Service for RAG-based chat about AI news."""

    SYSTEM_PROMPT = """You are an AI news assistant with access to a database of recent AI news articles and podcast episodes. Your role is to:

1. Answer questions about recent AI news and developments
2. Summarize and explain complex AI topics
3. Compare different perspectives from various sources
4. Highlight connections between related news items

When answering:
- Be concise but informative
- Cite specific sources when referencing news items
- Acknowledge uncertainty when information is limited
- Focus on facts from the provided context

You have access to retrieved news items that are relevant to the user's question. Use them to provide accurate, grounded responses."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = None
        if self.api_key:
            self.client = Anthropic(api_key=self.api_key)

        self._vector_store = vector_store
        self._vector_store_initialized = False

    @property
    def vector_store(self) -> Optional[VectorStore]:
        if not self._vector_store_initialized:
            try:
                if self._vector_store is None:
                    self._vector_store = VectorStore()
            except Exception:
                pass
            self._vector_store_initialized = True
        return self._vector_store

    def retrieve_context(
        self,
        query: str,
        limit: int = 10,
        item_type: Optional[str] = None,
    ) -> list[dict]:
        """Retrieve relevant items from the vector store."""
        if not self.vector_store:
            return []

        try:
            results = self.vector_store.search(
                query_text=query,
                item_type=item_type,
                limit=limit,
            )
            return results
        except Exception:
            return []

    def _format_context(self, items: list[dict]) -> str:
        """Format retrieved items as context for the prompt."""
        if not items:
            return "No relevant news items found in the database."

        formatted = ["RETRIEVED NEWS ITEMS:", "=" * 40]

        for i, item in enumerate(items, 1):
            metadata = item.get("metadata", {})
            formatted.append(f"\n[{i}] {metadata.get('title', 'Untitled')}")
            formatted.append(f"    Type: {metadata.get('type', 'news')}")
            formatted.append(f"    Source: {metadata.get('source', 'Unknown')}")
            formatted.append(f"    Score: {metadata.get('score', 0)}")
            formatted.append(f"    Relevance: {item.get('similarity', 0):.2f}")

            text = item.get("text", "")
            if text:
                # Truncate long text
                if len(text) > 500:
                    text = text[:500] + "..."
                formatted.append(f"    Content: {text}")

        return "\n".join(formatted)

    def chat(
        self,
        message: str,
        conversation_history: list[ChatMessage] = None,
        max_tokens: int = 1000,
    ) -> ChatMessage:
        """
        Process a chat message and return a response.

        Args:
            message: User's message
            conversation_history: Previous messages in the conversation
            max_tokens: Maximum tokens in response

        Returns:
            ChatMessage with the assistant's response and sources
        """
        if not self.client:
            return ChatMessage(
                role="assistant",
                content="Chat is not available. Please configure ANTHROPIC_API_KEY.",
                sources=[],
            )

        # Retrieve relevant context
        context_items = self.retrieve_context(message, limit=8)
        context_text = self._format_context(context_items)

        # Build messages
        messages = []

        # Add conversation history (last 10 messages)
        if conversation_history:
            for msg in conversation_history[-10:]:
                messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        # Add current message with context
        user_message = f"""Context from news database:
{context_text}

User question: {message}

Please answer based on the retrieved context. If the context doesn't contain relevant information, say so."""

        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                system=self.SYSTEM_PROMPT,
                messages=messages,
            )

            # Extract sources from context
            sources = []
            for item in context_items[:5]:  # Top 5 sources
                metadata = item.get("metadata", {})
                sources.append({
                    "title": metadata.get("title", "Untitled"),
                    "source": metadata.get("source", "Unknown"),
                    "type": metadata.get("type", "news"),
                    "similarity": item.get("similarity", 0),
                })

            return ChatMessage(
                role="assistant",
                content=response.content[0].text,
                sources=sources,
            )

        except Exception as e:
            return ChatMessage(
                role="assistant",
                content=f"Error generating response: {str(e)}",
                sources=[],
            )

    def chat_stream(
        self,
        message: str,
        conversation_history: list[ChatMessage] = None,
        max_tokens: int = 1000,
    ) -> Generator[str, None, dict]:
        """
        Stream a chat response.

        Yields text chunks, then returns metadata dict with sources.
        """
        if not self.client:
            yield "Chat is not available. Please configure ANTHROPIC_API_KEY."
            return {"sources": []}

        # Retrieve relevant context
        context_items = self.retrieve_context(message, limit=8)
        context_text = self._format_context(context_items)

        # Build messages
        messages = []

        if conversation_history:
            for msg in conversation_history[-10:]:
                messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        user_message = f"""Context from news database:
{context_text}

User question: {message}

Please answer based on the retrieved context. If the context doesn't contain relevant information, say so."""

        messages.append({"role": "user", "content": user_message})

        try:
            with self.client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                system=self.SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield text

            # Return sources after streaming
            sources = []
            for item in context_items[:5]:
                metadata = item.get("metadata", {})
                sources.append({
                    "title": metadata.get("title", "Untitled"),
                    "source": metadata.get("source", "Unknown"),
                    "type": metadata.get("type", "news"),
                    "similarity": item.get("similarity", 0),
                })

            return {"sources": sources}

        except Exception as e:
            yield f"\n\nError: {str(e)}"
            return {"sources": [], "error": str(e)}

    def get_suggested_questions(self, db) -> list[str]:
        """Get suggested questions based on recent news."""
        from web.models import Item, TopicCluster

        suggestions = [
            "What are the latest AI developments this week?",
            "Summarize the top AI news today",
            "What's happening with OpenAI?",
            "Any news about AI safety or regulation?",
            "What are the trending AI topics?",
        ]

        # Add dynamic suggestions based on recent clusters
        try:
            recent_clusters = db.query(TopicCluster).order_by(
                TopicCluster.created_at.desc()
            ).limit(3).all()

            for cluster in recent_clusters:
                suggestions.append(f"Tell me about {cluster.label}")
        except Exception:
            pass

        return suggestions[:8]


class ConversationManager:
    """Manages chat conversations with history."""

    def __init__(self):
        self.conversations: dict[str, list[ChatMessage]] = {}

    def get_conversation(self, conversation_id: str) -> list[ChatMessage]:
        """Get conversation history."""
        return self.conversations.get(conversation_id, [])

    def add_message(self, conversation_id: str, message: ChatMessage):
        """Add a message to conversation history."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        self.conversations[conversation_id].append(message)

        # Keep only last 50 messages
        if len(self.conversations[conversation_id]) > 50:
            self.conversations[conversation_id] = self.conversations[conversation_id][-50:]

    def clear_conversation(self, conversation_id: str):
        """Clear conversation history."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]

    def get_all_conversation_ids(self) -> list[str]:
        """Get all conversation IDs."""
        return list(self.conversations.keys())


# Global conversation manager
conversation_manager = ConversationManager()
