"""
Processor components for the healthcare RAG system.

Each module in this package contains processors responsible for a specific part of the RAG pipeline:
- base: Base processor class and shared utilities
- preprocessing: Query preprocessing (clarification, decomposition)
- retrieval: Document retrieval and evaluation
- generation: Answer generation
- validation: Citation validation
- followups: Follow-up question generation
"""

from .base import BaseProcessor, PromptManager, log_timing
from .preprocessing import QueryPreprocessor, ConversationContextProcessor
from .retrieval import QueryRouter, RetrievalEvaluator
from .generation import AnswerGenerator
from .validation import AnswerValidator
from .followups import FollowUpQuestionsGenerator

__all__ = [
    # Base classes
    "BaseProcessor",
    "PromptManager",
    "log_timing",
    
    # Preprocessing
    "QueryPreprocessor",
    "ConversationContextProcessor",
    
    # Retrieval
    "QueryRouter",
    "RetrievalEvaluator",
    
    # Generation and Validation
    "AnswerGenerator",
    "AnswerValidator",
    
    # Followups
    "FollowUpQuestionsGenerator",
]
