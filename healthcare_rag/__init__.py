"""
Healthcare RAG - A healthcare-focused retrieval-augmented generation system.

This package provides components for sophisticated RAG workflows focused on healthcare data.
"""

__version__ = "0.1.0"

# Re-export main classes for easy import
from .pipeline.medical_rag import MedicalRAG
from .config import setup_medical_rag

__all__ = [
    "MedicalRAG",
    "setup_medical_rag",
]
