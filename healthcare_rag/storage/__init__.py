"""
Storage components for the healthcare RAG system.

Each module in this package provides functionality for storing and retrieving data:
- history: Conversation history storage and retrieval
"""

from .history import ConversationHistory

__all__ = [
    "ConversationHistory",
]
