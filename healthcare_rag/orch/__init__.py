"""
Orchestration components for the healthcare RAG system.

This module provides the orchestration layer for RAG pipelines, 
managing component interactions and workflow execution.
"""

# Import the refactored class and the example runner function
from .orchestrator import RefactoredOrchestrator, run_refactored_orchestrator

__all__ = [
    "RefactoredOrchestrator",
    "run_refactored_orchestrator",
]
