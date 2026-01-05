"""Conversation memory storage for Deep Research Bot."""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypedDict

logger = logging.getLogger(__name__)


class Message(TypedDict):
    """OpenAI-compatible message format."""
    role: str  # "user" or "assistant"
    content: str


@dataclass
class Conversation:
    """Single user conversation."""
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_user_message(self, content: str) -> None:
        """Add user message to conversation."""
        self.messages.append({"role": "user", "content": content})
        self.updated_at = datetime.now()

    def add_assistant_message(self, content: str) -> None:
        """Add assistant response to conversation."""
        self.messages.append({"role": "assistant", "content": content})
        self.updated_at = datetime.now()

    def get_messages(self) -> list[Message]:
        """Get all messages."""
        return self.messages.copy()

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
        self.updated_at = datetime.now()

    def trim(self, max_messages: int) -> None:
        """Trim conversation to max_messages, keeping most recent."""
        if len(self.messages) > max_messages:
            # Keep system message if present, then most recent messages
            trimmed = self.messages[-max_messages:]
            self.messages = trimmed
            logger.debug(f"Trimmed conversation to {max_messages} messages")


class ConversationMemory:
    """In-memory conversation storage per user."""

    def __init__(self, max_messages_per_user: int = 20):
        """
        Initialize conversation memory.

        Args:
            max_messages_per_user: Maximum messages to keep per user
        """
        self._conversations: dict[int, Conversation] = {}
        self.max_messages = max_messages_per_user

    def get_conversation(self, user_id: int) -> Conversation:
        """Get or create conversation for user."""
        if user_id not in self._conversations:
            self._conversations[user_id] = Conversation()
            logger.debug(f"Created new conversation for user {user_id}")
        return self._conversations[user_id]

    def add_user_message(self, user_id: int, content: str) -> None:
        """Add user message to conversation."""
        conv = self.get_conversation(user_id)
        conv.add_user_message(content)
        conv.trim(self.max_messages)

    def add_assistant_message(self, user_id: int, content: str) -> None:
        """Add assistant response to conversation."""
        conv = self.get_conversation(user_id)
        conv.add_assistant_message(content)
        conv.trim(self.max_messages)

    def get_messages(self, user_id: int) -> list[Message]:
        """Get conversation messages for user."""
        return self.get_conversation(user_id).get_messages()

    def clear_conversation(self, user_id: int) -> bool:
        """Clear conversation for user. Returns True if conversation existed."""
        if user_id in self._conversations:
            self._conversations[user_id].clear()
            logger.info(f"Cleared conversation for user {user_id}")
            return True
        return False

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            "total_users": len(self._conversations),
            "total_messages": sum(
                len(conv.messages) for conv in self._conversations.values()
            ),
        }


# Global memory instance
_memory: ConversationMemory | None = None


def get_memory(max_messages: int = 20) -> ConversationMemory:
    """Get or create global memory instance."""
    global _memory
    if _memory is None:
        _memory = ConversationMemory(max_messages_per_user=max_messages)
    return _memory
