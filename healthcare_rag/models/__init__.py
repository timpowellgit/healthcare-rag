"""
Data models for the healthcare RAG system.

Each module in this package contains Pydantic models related to a specific domain:
- queries: Models for query processing (clarification, decomposition, etc.)
- answers: Models for structuring and validating answers
- retrieval: Models for document retrieval and storage
- misc: Miscellaneous models used across the system
"""

# Import commonly used models for convenience
from .queries import ClarifiedQuery, DecomposedQuery, RetrievalEvaluation
from .retrieval import QueryDocument, QueryResult, QueryResultList
from .answers import (
    Citation, 
    StatementWithCitations, 
    CitedAnswerResult,
    AnswerGenerationResult,
    RelevantHistoryContext
)
from .misc import ConversationEntry, FollowUpQuestions

__all__ = [
    # Query models
    "ClarifiedQuery",
    "DecomposedQuery",
    "RetrievalEvaluation",
    
    # Retrieval models
    "QueryDocument",
    "QueryResult",
    "QueryResultList",
    
    # Answer models
    "Citation",
    "StatementWithCitations",
    "CitedAnswerResult",
    "AnswerGenerationResult",
    "RelevantHistoryContext",
    
    # Misc models
    "ConversationEntry",
    "FollowUpQuestions",
]
