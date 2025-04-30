from typing import List
from pydantic import BaseModel, Field
from datetime import datetime

class ConversationEntry(BaseModel):
    """Represents a single entry in a conversation history."""
    timestamp: datetime
    user_query: str
    answer: str


class FollowUpQuestions(BaseModel):
    """Suggested follow-up questions based on the previous answer."""
    questions: List[str] = Field(
        description="A list of 3 follow-up questions that would be natural to ask next based on the answer."
    ) 