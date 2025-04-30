import logging
from typing import Dict, List, Optional, Tuple, Callable, AsyncGenerator
from uuid import uuid4

from ..models.retrieval import QueryResultList
from ..models.answers import AnswerGenerationResult
from .base import BaseProcessor, log_timing

logger = logging.getLogger("MedicalRAG")

class AnswerGenerator(BaseProcessor):
    """Generates answers from retrieved documents."""

    def _format_documents_for_prompt(
        self, retrieval_results: QueryResultList
    ) -> Tuple[str, Dict[str, str]]:
        """
        Format retrieved documents for the prompt and create a mapping between 
        prompt IDs and original document IDs.
        """
        doc_context = ""
        prompt_id_to_original_id_map: Dict[str, str] = {}
        doc_index = 0

        if not retrieval_results or not retrieval_results.results:
            return "", {}

        for result in retrieval_results.results:
            for doc in result.docs:
                original_doc_id = doc.doc_id or f"missing_id_{uuid4()}"
                prompt_doc_id = f"doc_{doc_index + 1}"
                prompt_id_to_original_id_map[prompt_doc_id] = original_doc_id

                doc_context += f"Document ID: [{prompt_doc_id}]\n"
                doc_context += f"Content: {doc.content}\n"
                doc_context += f"Source: {doc.source_name}\n"
                if doc.page_numbers:
                    doc_context += f"Page Numbers: {doc.page_numbers}\n"
                doc_context += "---\n"
                doc_index += 1

        return doc_context.strip(), prompt_id_to_original_id_map

    def _create_result(
        self,
        plain_answer: str,
        retrieval_results: QueryResultList,
        formatted_docs: str = "",
        prompt_id_map: Optional[Dict[str, str]] = None,
        user_question: str = "",
        conversation_context: str = "",
    ) -> AnswerGenerationResult:
        """Helper method to create AnswerGenerationResult with provided values."""
        return AnswerGenerationResult(
            plain_answer=plain_answer,
            retrieval_results=retrieval_results,
            formatted_docs=formatted_docs,
            prompt_id_map=prompt_id_map or {},
            user_question=user_question,
            conversation_context=conversation_context,
        )

    @log_timing
    async def generate_answer_async(
        self,
        user_question: str,
        retrieval_results: QueryResultList,
        conversation_context: str = "",
    ) -> AnswerGenerationResult:
        """
        Generates a plain text answer and packages it with context for validation.
        """
        default_error_answer = "I'm sorry, I don't know the answer to that question."

        # Initialize result with common parameters
        result = AnswerGenerationResult(
            plain_answer=default_error_answer,
            retrieval_results=retrieval_results or QueryResultList(results=[]),
            formatted_docs="",
            prompt_id_map={},
            user_question=user_question,
            conversation_context=conversation_context,
        )

        if not retrieval_results or not retrieval_results.results:
            logger.warning("No retrieval results provided.")
            return result

        # Format documents for the prompt and get the ID map
        formatted_docs, prompt_id_map = self._format_documents_for_prompt(
            retrieval_results
        )

        # Update result with formatted docs and prompt ID map
        result.formatted_docs = formatted_docs
        result.prompt_id_map = prompt_id_map

        if not formatted_docs:
            logger.warning("Formatted documents are empty.")
            result.plain_answer = "I encountered an issue processing the information."
            return result

        logger.info(f"Generating plain text answer for query: '{user_question}'")

        # Generate the plain answer string
        plain_answer = await self._call_llm_completions(
            prompt_name="answer_generation",
            user_question=user_question,
            retrieval_results=formatted_docs,
            conversation_context=conversation_context or "",
            temperature=0.1,
            default_response=default_error_answer,
        )

        # Update result with generated answer
        result.plain_answer = plain_answer

        return result
        
    async def generate_answer_stream(
        self,
        user_question: str,
        retrieval_results: QueryResultList,
        conversation_context: str = "",
        callback: Optional[Callable[[str], None]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream the answer generation process, yielding chunks as they are generated.
        
        Args:
            user_question: The user's query
            retrieval_results: The retrieved documents
            conversation_context: Optional conversation context
            callback: Optional callback function for each chunk
            
        Yields:
            Chunks of the generated answer as they become available
        """
        default_error_answer = "I'm sorry, I don't know the answer to that question."
        
        if not retrieval_results or not retrieval_results.results:
            logger.warning("No retrieval results provided.")
            if callback:
                callback(default_error_answer)
            yield default_error_answer
            return

        # Format documents for the prompt and get the ID map
        formatted_docs, prompt_id_map = self._format_documents_for_prompt(
            retrieval_results
        )
        
        if not formatted_docs:
            logger.warning("Formatted documents are empty.")
            error_msg = "I encountered an issue processing the information."
            if callback:
                callback(error_msg)
            yield error_msg
            return
        
        logger.info(f"Streaming answer generation for query: '{user_question}'")
        
        # Use the async client directly for streaming
        if self.async_client is None:
            logger.error("OpenAI async client is not initialized")
            if callback:
                callback(default_error_answer)
            yield default_error_answer
            return
        
        # Get the messages for the prompt
        messages = self.pm.messages(
            "answer_generation",
            user_question=user_question,
            retrieval_results=formatted_docs,
            conversation_context=conversation_context or "",
        )
        
        # Stream the response chunks
        full_response = ""
        try:
            stream = await self.async_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.1,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    if callback:
                        callback(content)
                    yield content
                    
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            if not full_response:
                if callback:
                    callback(default_error_answer)
                yield default_error_answer 