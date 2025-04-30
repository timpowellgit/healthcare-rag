import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any

import openai
from weaviate.client import WeaviateAsyncClient

from ..models.misc import ConversationEntry
from ..services.llm import LLMParserService
from ..processors import (
    PromptManager,
    QueryPreprocessor,
    ConversationContextProcessor,
    QueryRouter,
    RetrievalEvaluator,
    AnswerGenerator,
    AnswerValidator,
    FollowUpQuestionsGenerator,
)
from ..processors.preprocessing import QueryPreprocessor as NewQueryPreprocessor

from ..processors.generation import AnswerGenerator as NewAnswerGenerator
from ..processors.validation import AnswerValidator as NewAnswerValidator
from ..storage.history import ConversationHistory

logger = logging.getLogger("MedicalRAG")

class MedicalRAG:
    """
    A medical RAG system that integrates Weaviate, query routing, answer generation,
    and conversation history.
    """

    def __init__(
        self,
        weaviate_client: WeaviateAsyncClient,
        collection_names: List[str],
        llm_model: str = "gpt-4o-mini",
        parser_service: Optional[LLMParserService] = None,
        conversation_history_dir: str = "data/conversations",
        prompts_dir: str = "prompts",
    ):
        """
        Initialize the Medical RAG system with a simplified configuration.

        Args:
            weaviate_client: The Weaviate async client
            collection_names: Names of collections in Weaviate to query
            llm_model: Single model name used for all LLM components
            parser_service: Service for making parsed LLM calls
            conversation_history_dir: Directory to store conversation histories
            prompts_dir: Directory containing Jinja templates for prompts
        """
        # Set up shared resources
        self.async_client = openai.AsyncOpenAI()
        self.prompt_manager = PromptManager(prompts_dir)
        self.parser_service = parser_service or LLMParserService(self.async_client)

        # Store the Weaviate client
        self.weaviate_client = weaviate_client

        # Initialize the router with Weaviate client and collection names
        self.router = QueryRouter(
            weaviate_client=weaviate_client,
            collection_names=collection_names,
            llm_model=llm_model,
            async_client=self.async_client,
        )

        # All BaseProcessor components share the same model and resources
        common_args = {
            "llm_model": llm_model,
            "async_client": self.async_client,
            "prompt_manager": self.prompt_manager,
            "parser_service": self.parser_service,
        }

        # Create validator args with a different model
        validator_args = common_args.copy()
        validator_args["llm_model"] = "gpt-4o"  # Using a more capable model for validation

        # Initialize processing components
        self.generator = AnswerGenerator(**common_args)
        self.preprocessor = QueryPreprocessor(**common_args)
        self.evaluator = RetrievalEvaluator(**common_args)
        self.context_processor = ConversationContextProcessor(**common_args)
        self.follow_up_generator = FollowUpQuestionsGenerator(**common_args)
        self.validator = AnswerValidator(**validator_args)

        # Initialize conversation history
        self.conversation_history = ConversationHistory(conversation_history_dir)

    def get_conversation_history(
        self, user_id: str, limit: int = 5
    ) -> List[ConversationEntry]:
        """
        Get recent conversation history for a user.

        Args:
            user_id: Unique identifier for the user
            limit: Maximum number of entries to return

        Returns:
            List of conversation entries, most recent first
        """
        return self.conversation_history.get_history(user_id, limit)

    async def process_query_simple(
        self, user_id: str, user_query: str, use_history: bool = True
    ) -> Tuple[str, List[str]]:
        """
        Processes a query through the RAG pipeline and returns the answer and suggested follow-up questions.

        Args:
            user_id: Unique identifier for the user
            user_query: The user's query
            use_history: Whether to use conversation history for context

        Returns:
            A tuple containing the answer string and a list of follow-up questions
        """
        logger.info(f"Processing query for user '{user_id}': '{user_query}'")

        # Get conversation context if needed
        conversation_context = ""
        history = []
        if use_history:
            history = self.conversation_history.get_history(user_id)
            if history:
                logger.info(f"Found {len(history)} historical entries for user")
                ctx_result = await self.context_processor.extract_relevant_context(
                    user_query, history
                )
                conversation_context = ctx_result.relevant_snippets
                logger.info(
                    f"Context extraction result: required={ctx_result.required_context}"
                )

        # Route the query to get documents
        retrieval_results = await self.router.route_query_async(user_query)
        
        # Evaluate retrieval and fetch any missing information
        current_results = await self.evaluator.evaluate_retrieval(
            original_query=user_query,
            clarified_query=user_query,
            retrieval_results=retrieval_results,
            router=self.router,
        )

        # Generate the answer
        generation_result = await self.generator.generate_answer_async(
            user_question=user_query,
            retrieval_results=current_results,
            conversation_context=conversation_context or "",
        )
        
        logger.info("Plain text answer and context generated.")

        initial_answer = generation_result.plain_answer
        final_answer = initial_answer  # Default to initial answer
        
        if initial_answer != "I'm sorry, I don't know the answer to that question.":
            # Run structuring and validation
            logger.info("Starting structuring and validation...")
            structured_answer, validated_answer = await self.validator.structure_and_validate_async(
                plain_answer=initial_answer,
                retrieval_results=generation_result.retrieval_results,
                formatted_docs=generation_result.formatted_docs,
                prompt_id_map=generation_result.prompt_id_map,
            )

            # Use validated answer if available and non-empty
            if validated_answer:
                final_answer = validated_answer
                logger.info("Using validated answer with invalid statements removed.")
            elif structured_answer is None:
                logger.warning(
                    "Structuring failed. Falling back to initial unvalidated answer."
                )
                # final_answer remains initial_answer
            else:
                logger.warning(
                    "Validation resulted in an empty answer. Falling back to initial unvalidated answer."
                )
                # final_answer remains initial_answer
        else:
            logger.info("Skipping structuring/validation due to default error answer.")

        # Generate follow-up questions
        follow_ups = await self.follow_up_generator.generate_follow_up_questions(
            query=user_query,
            answer=final_answer,  # Use the final (potentially validated) answer
            conversation_history=history if use_history else None,
        )

        # Save the final (potentially validated) answer to conversation history
        self.conversation_history.add_entry(user_id, user_query, final_answer)

        return final_answer, follow_ups.questions 