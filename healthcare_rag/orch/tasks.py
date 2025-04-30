"""
Task wrappers for the healthcare RAG orchestrator.

This module provides thin async wrappers around MedicalRAG processor methods.
"""

import logging
from typing import List, Optional, Tuple

from ..models.queries import ClarifiedQuery, DecomposedQuery, RetrievalEvaluation
from ..models.retrieval import QueryResultList
from ..models.answers import CitedAnswerResult, AnswerGenerationResult, RelevantHistoryContext
from ..models.misc import ConversationEntry, FollowUpQuestions
from ..pipeline.medical_rag import MedicalRAG

logger = logging.getLogger("MedicalRAG")

# Task wrappers that call through to MedicalRAG components
async def clarify_query(
    rag: MedicalRAG, query: str, context: str = ""
) -> ClarifiedQuery:
    """
    Clarify an ambiguous query using conversation context.
    
    Args:
        rag: The MedicalRAG instance
        query: The user's query to clarify
        context: Optional conversation context
        
    Returns:
        A ClarifiedQuery object with the clarified query
    """
    return await rag.preprocessor.clarify_query_async(
        user_query=query, conversation_context=context
    )

async def decompose_query(rag: MedicalRAG, query: str) -> DecomposedQuery:
    """
    Decompose a complex query into simpler subqueries.
    
    Args:
        rag: The MedicalRAG instance
        query: The user's query to decompose
        
    Returns:
        A DecomposedQuery object with the decomposed subqueries
    """
    return await rag.preprocessor.decompose_query_async(query)

async def retrieve_documents(rag: MedicalRAG, query: str) -> QueryResultList:
    """
    Retrieve documents relevant to the query.
    
    Args:
        rag: The MedicalRAG instance
        query: The query to retrieve documents for
        
    Returns:
        A QueryResultList containing retrieved documents
    """
    return await rag.router.route_query_async(query)

async def evaluate_retrieval(
    rag: MedicalRAG, query: str, results: QueryResultList
) -> QueryResultList:
    """
    Evaluate the retrieval results and augment if needed.
    
    Args:
        rag: The MedicalRAG instance
        query: The user's query
        results: The initial retrieval results
        
    Returns:
        A potentially augmented QueryResultList
    """
    return await rag.evaluator.evaluate_retrieval(
        original_query=query,
        clarified_query=query,
        retrieval_results=results,
        router=rag.router,
    )

async def extract_conversation_context(
    rag: MedicalRAG, query: str, history: List[ConversationEntry]
) -> RelevantHistoryContext:
    """
    Extract relevant context from conversation history.
    
    Args:
        rag: The MedicalRAG instance
        query: The user's query
        history: List of conversation entries
        
    Returns:
        A RelevantHistoryContext object
    """
    return await rag.context_processor.extract_relevant_context(
        query=query, conversation_history=history
    )

async def generate_answer(
    rag: MedicalRAG, results: QueryResultList, query: str, summary: RelevantHistoryContext
) -> AnswerGenerationResult:
    """
    Generate an answer from the retrieved documents and conversation context.
    
    Args:
        rag: The MedicalRAG instance
        results: The retrieval results
        query: The user's query
        summary: The context from conversation history
        
    Returns:
        An AnswerGenerationResult object
    """
    return await rag.generator.generate_answer_async(
        user_question=query,
        retrieval_results=results,
        conversation_context=summary.relevant_snippets or "",
    )

async def validate_answer(
    rag: MedicalRAG, generation_result: AnswerGenerationResult
) -> Tuple[Optional[CitedAnswerResult], Optional[str]]:
    """
    Validate an answer by checking citations.
    
    Args:
        rag: The MedicalRAG instance
        generation_result: The result from answer generation
        
    Returns:
        A tuple of (structured_answer, validated_text)
    """
    return await rag.validator.structure_and_validate_async(
        plain_answer=generation_result.plain_answer,
        retrieval_results=generation_result.retrieval_results,
        formatted_docs=generation_result.formatted_docs,
        prompt_id_map=generation_result.prompt_id_map,
    )

async def generate_follow_ups(
    rag: MedicalRAG, query: str, answer: str, history: List[ConversationEntry]
) -> FollowUpQuestions:
    """
    Generate follow-up questions based on the answer.
    
    Args:
        rag: The MedicalRAG instance
        query: The user's query
        answer: The answer provided to the user
        history: List of conversation entries
        
    Returns:
        A FollowUpQuestions object
    """
    return await rag.follow_up_generator.generate_follow_up_questions(
        query=query,
        answer=answer,
        conversation_history=history,
    ) 