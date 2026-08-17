"""
Command-line interface components for the healthcare RAG system.

This package contains the CLI tools for interacting with the RAG system.
"""

from .interactive import main, interactive_main, process_query_with_orchestrator, QueryMonitor

# Ingestion helpers pull in docling/torch/easyocr and are only needed by the
# ingestion CLI. Import them directly from `healthcare_rag.cli.ingestion` when
# you need them, so `python -m healthcare_rag` (chat only) does not force-load
# the PDF processing stack.

__all__ = [
    "main",
    "interactive_main",
    "process_query_with_orchestrator",
    "QueryMonitor",
]
