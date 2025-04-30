from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class QueryDocument(BaseModel):
    """Represents a single document retrieved from a vector database."""
    content: str
    score: float = Field(
        description="The score metric from the query (higher means more relevant)."
    )
    doc_id: str = Field(description="Unique identifier for this document chunk.")
    source_name: str = Field(
        description="The name of the source (e.g., 'Lipitor', 'Metformin')."
    )
    metadata: Optional[Dict[str, Any]] = None
    page_numbers: Optional[List[int]] = None


class QueryResult(BaseModel):
    """Represents search results from a single source."""
    source: str
    query: str
    docs: List[QueryDocument]


class QueryResultList(BaseModel):
    """Container for all query results across multiple sources."""
    results: List[QueryResult]


class ErrorResult(BaseModel):
    """Simple error representation."""
    error: str 