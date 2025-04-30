import logging
from typing import List, Optional

from ..models.misc import FollowUpQuestions, ConversationEntry
from .base import BaseProcessor, log_timing

logger = logging.getLogger("MedicalRAG")

class FollowUpQuestionsGenerator(BaseProcessor):
    """Generates follow-up questions based on the answer and conversation history."""

    @log_timing
    async def generate_follow_up_questions(
        self,
        query: str,
        answer: str,
        conversation_history: Optional[List[ConversationEntry]] = None,
    ) -> FollowUpQuestions:
        """
        Generate three potential follow-up questions based on the answer and conversation history.

        Args:
            query: The original user query
            answer: The answer provided to the user
            conversation_history: Optional list of previous conversation entries

        Returns:
            A FollowUpQuestions object containing a list of 3 follow-up questions
        """
        logger.info(f"Generating follow-up questions for query: '{query}'")

        # Format conversation history if provided
        history_context = ""
        if conversation_history and len(conversation_history) > 0:
            history_context = "Previous conversation:\n"
            for i, entry in enumerate(conversation_history):
                history_context += f"User: {entry.user_query}\n"
                history_context += f"Assistant: {entry.answer}\n\n"
            logger.debug(
                f"Using {len(conversation_history)} conversation entries for context"
            )

        # Use the base class _call_llm method instead of direct calls to parser_service
        default_response = FollowUpQuestions(questions=[])

        response = await self._call_llm(
            prompt_name="follow_up_questions",
            original_query=query,
            answer=answer,
            history_context=history_context,
            temperature=0.3,  # Higher temperature for more variety
            response_format=FollowUpQuestions,
            default_response=default_response,
        )

        if response is None:
            return default_response

        return response 