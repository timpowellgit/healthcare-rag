import json
import os
import logging
from typing import Dict, List
from datetime import datetime

from ..models.misc import ConversationEntry

logger = logging.getLogger("MedicalRAG")

class ConversationHistory:
    """Manages conversation history for users."""

    def __init__(self, save_directory: str = "data/conversations"):
        """
        Initialize the conversation history manager.

        Args:
            save_directory: Directory to save conversation histories
        """
        self.save_directory = save_directory
        os.makedirs(save_directory, exist_ok=True)
        self.conversations: Dict[str, List[ConversationEntry]] = {}

    def add_entry(self, user_id: str, query: str, answer: str) -> None:
        """
        Add a new entry to a user's conversation history.

        Args:
            user_id: Unique identifier for the user
            query: The user's question
            answer: The system's answer
        """
        if user_id not in self.conversations:
            self.conversations[user_id] = []

        entry = ConversationEntry(
            timestamp=datetime.now(),
            user_query=query,
            answer=answer,
        )

        self.conversations[user_id].append(entry)
        self._save_conversation(user_id)

    def get_history(self, user_id: str, limit: int = 5) -> List[ConversationEntry]:
        """
        Get recent conversation history for a user.

        Args:
            user_id: Unique identifier for the user
            limit: Maximum number of entries to return

        Returns:
            List of conversation entries, most recent first
        """
        if user_id not in self.conversations:
            self._load_conversation(user_id)

        history = self.conversations.get(user_id, [])

        # Convert to dict format and limit number of entries
        return [
            ConversationEntry(
                timestamp=entry.timestamp,
                user_query=entry.user_query,
                answer=entry.answer,
            )
            for entry in history[-limit:][::-1]  # Most recent first
        ]

    def get_context_from_history(self, user_id: str, limit: int = 3) -> str:
        """
        Generate a conversation context string from recent history.

        Args:
            user_id: Unique identifier for the user
            limit: Maximum number of past exchanges to include

        Returns:
            Formatted conversation context string
        """
        if user_id not in self.conversations:
            self._load_conversation(user_id)

        history = self.conversations.get(user_id, [])

        if not history:
            return ""

        # Get most recent exchanges
        recent = history[-limit:]

        # Format as conversation context
        context = "Previous conversation:\n"
        for entry in recent:
            context += f"User: {entry.user_query}\n"
            context += f"Assistant: {entry.answer}\n\n"

        return context

    def _save_conversation(self, user_id: str) -> None:
        """
        Save a user's conversation history to disk.

        Args:
            user_id: Unique identifier for the user
        """
        file_path = os.path.join(self.save_directory, f"{user_id}.json")

        # Use model_dump() for serialization
        entries = [
            entry.model_dump(mode="json") for entry in self.conversations[user_id]
        ]

        with open(file_path, "w") as f:
            json.dump(entries, f, indent=2)

    def _load_conversation(self, user_id: str) -> None:
        """
        Load a user's conversation history from disk.

        Args:
            user_id: Unique identifier for the user
        """
        file_path = os.path.join(self.save_directory, f"{user_id}.json")

        if not os.path.exists(file_path):
            self.conversations[user_id] = []
            return

        try:
            with open(file_path, "r") as f:
                entries_data = json.load(f)

            # Use model_validate for deserialization
            self.conversations[user_id] = [
                ConversationEntry.model_validate(entry_data)
                for entry_data in entries_data
            ]
        except Exception as e:
            logger.error(f"Error loading conversation for {user_id}: {e}")
            self.conversations[user_id] = [] 