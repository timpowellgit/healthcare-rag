from typing import List, Literal, Optional, Self
from pydantic import BaseModel, Field, model_validator

class ClarifiedQuery(BaseModel):
    """Represents a clarified user query with ambiguity assessment."""
    original_query: str
    ambiguity_level: Literal[
        "clear and specific", "medium ambiguity", "high ambiguity"
    ] = Field(description="The level of ambiguity in the original query.")
    clarified_query: str = Field(
        description="The clarified query that resolves the ambiguity. If there is nothing to clarify, just return the original query."
    )

    @model_validator(mode="after")
    def check_clarified_query(self) -> Self:
        if (
            self.ambiguity_level == "clear and specific"
            and self.clarified_query != self.original_query
        ):
            self.clarified_query = self.original_query
        return self


class DecomposedQuery(BaseModel):
    """Represents a complex query broken down into simpler components."""
    original_query: str
    query_complexity: Literal["simple", "complex"] = Field(
        description="The complexity of the original query."
    )
    decomposed_query: Optional[List[str]] = Field(
        description="The original query decomposed into subqueries. "
        "Only meant for unpacking complex queries. If there is nothing to decompose, "
        "just return the original query."
    )

    @model_validator(mode="after")
    def check_decomposed_query(self) -> Self:
        if (
            self.query_complexity == "simple"
            and self.decomposed_query != self.original_query
        ):
            self.decomposed_query = [self.original_query]
        return self


class RetrievalEvaluation(BaseModel):
    """Evaluates whether retrieved documents are sufficient to answer a query."""
    is_sufficient: bool = Field(
        description="Whether the retrieved information is sufficient to answer the query"
    )
    missing_information: Optional[str] = Field(
        None,
        description="Description of information that's missing from the retrieved documents",
    )
    additional_queries: Optional[List[str]] = Field(
        None,
        description="List of suggested follow-up queries to retrieve missing information",
    ) 