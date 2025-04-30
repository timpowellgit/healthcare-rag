"""
Main pipeline components for the healthcare RAG system.

This package contains the high-level orchestration logic that ties together
all the processors, services, and storage components.
"""

from .medical_rag import MedicalRAG

__all__ = [
    "MedicalRAG",
]
