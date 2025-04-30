from typing import List, Literal, Optional, Self
from pydantic import BaseModel, Field, model_validator
from .retrieval import QueryResultList

class Citation(BaseModel):
    """Represents a citation for a particular statement in the answer."""
    doc_id: str = Field(description="The unique identifier of the cited document.")
    source_name: str = Field(
        description="The name of the source (e.g., 'Lipitor', 'Metformin')."
    )
    quote: str = Field(
        description="The verbatim quote from the document that supports the statement."
    )


class StatementWithCitations(BaseModel):
    """A statement with its supporting citations."""
    text: str = Field(
        description="The statement text. Can be a single sentence or a paragraph."
    )
    citations: List[Citation] = Field(
        description="Citations supporting this statement."
    )
    linebreaks: Literal["\\n", "\\n\\n", "\\n\\n\\n", ""] = Field(
        description="The linebreaks or other formatting in the statement. If there are no linebreaks, return an empty string."
    )


class CitedAnswerResult(BaseModel):
    """A complete answer with structured citations."""
    statements: List[StatementWithCitations] = Field(description="Answer to the query")


class AnswerGenerationResult(BaseModel):
    """Contains the generated answer and the context used to generate it."""
    plain_answer: str
    retrieval_results: QueryResultList
    formatted_docs: str
    prompt_id_map: dict[str, str]
    user_question: str
    conversation_context: str


class RelevantHistoryContext(BaseModel):
    """Extracted context from conversation history relevant to the current query."""
    explanation: str = Field(
        description="Explanation of why any of the previous conversation entries "
        "are needed to answer the current query. If none of the previous "
        "conversation entries are needed to answer the current query, "
        "return an empty string."
    )
    required_context: bool = Field(
        description="Whether the context is required to answer the query. "
        "If none of the previous conversation entries are needed "
        "to answer the current query, return False."
    )

    relevant_snippets: Optional[str] = Field(
        description="Formatted context string ready to be used in the answer "
        "generation prompt. If none of the previous conversation entries "
        "are needed to answer the current query, return an empty string."
    )

    @model_validator(mode="after")
    def check_relevant_snippets(self) -> Self:
        if self.required_context == False:
            self.relevant_snippets = ""
        return self 