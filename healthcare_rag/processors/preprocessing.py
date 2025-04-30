import logging
from typing import List, Optional

from ..models.queries import ClarifiedQuery, DecomposedQuery
from ..models.answers import RelevantHistoryContext
from ..models.misc import ConversationEntry
from .base import BaseProcessor, log_timing

logger = logging.getLogger("MedicalRAG")

class QueryPreprocessor(BaseProcessor):
    """
    Preprocesses user queries to resolve ambiguities using conversation history.
    Detects and clarifies ambiguous references like "it", "they", "this side effect", etc.
    """

    @log_timing
    async def clarify_query_async(
        self, user_query: str, conversation_context: str = ""
    ) -> ClarifiedQuery:
        logger.info(f"Clarifying query: '{user_query}'")

        default_response = ClarifiedQuery(
            original_query=user_query,
            clarified_query=user_query,
            ambiguity_level="clear and specific",
        )

        if not conversation_context:
            logger.debug("No conversation context, skipping clarification")
            return default_response

        result = await self._call_llm(
            prompt_name="clarify_query",
            user_query=user_query,
            conversation_context=conversation_context,
            response_format=ClarifiedQuery,
            default_response=default_response,
        )
        
        return result or default_response

    @log_timing
    async def decompose_query_async(self, user_query: str) -> DecomposedQuery:
        logger.info(f"Decomposing query: '{user_query}'")

        default_response = DecomposedQuery(
            original_query=user_query,
            query_complexity="simple",
            decomposed_query=[user_query],
        )

        result = await self._call_llm(
            prompt_name="decompose_query",
            user_query=user_query,
            response_format=DecomposedQuery,
            default_response=default_response,
        )
        
        return result or default_response


class ConversationContextProcessor(BaseProcessor):
    """Processes the conversation history to extract only relevant context for the current query."""

    @log_timing
    async def extract_relevant_context(
        self, query: str, conversation_history: List[ConversationEntry]
    ) -> RelevantHistoryContext:
        """
        Extract only the relevant parts of conversation history for the current query.

        Args:
            query: The current user query (clarified)
            conversation_history: Full conversation history

        Returns:
            RelevantHistoryContext with the formatted context string and relevant entries
        """
        if not conversation_history:
            return RelevantHistoryContext(
                required_context=False,
                explanation="No conversation history available",
                relevant_snippets="",
            )

        # Format conversation history for the prompt
        history_text = "Previous conversation:\n"
        for i, entry in enumerate(conversation_history):
            history_text += f"[{i+1}] User: {entry.user_query}\n"
            history_text += f"    Assistant: {entry.answer}\n\n"

        # Use the base class _call_llm method instead of direct parser calls
        default_response = RelevantHistoryContext(
            required_context=False,
            explanation="No relevant context found",
            relevant_snippets="",
        )

        response = await self._call_llm(
            prompt_name="context_extraction",
            current_query=query,
            history_text=history_text,
            temperature=0.1,
            response_format=RelevantHistoryContext,
            default_response=default_response,
        )

        if response is None:
            return default_response

        return response 