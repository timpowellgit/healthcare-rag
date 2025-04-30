from langchain.embeddings import OpenAIEmbeddings
from uuid import uuid4
import json
from typing import (
    Dict,
    List,
    Union,
    Any,
    Optional,
    TypedDict,
    Tuple,
    NamedTuple,
    Literal,
    AsyncGenerator,
)
from datetime import datetime
from langchain_core.documents import Document
import os
import openai
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

# Add Weaviate imports
from weaviate.client import WeaviateAsyncClient
from weaviate.connect import ConnectionParams
from weaviate.classes.query import MetadataQuery, HybridFusion

from pydantic import BaseModel, model_validator, Field, SkipValidation
from typing import Self, Type, TypeVar
import asyncio
from itertools import chain  # (used later)
import logging
import time
from pydantic.json_schema import SkipJsonSchema
from fuzzywuzzy import fuzz, process as fuzzy_process
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path
import re  # Make sure this import is present at the top of the file

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MedicalRAG")


# Custom timing decorator for instrumentation
def log_timing(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result

    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return wrapper


# Type definitions
ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


# Type definitions
class LLMParserService:
    """Handles making calls to and parsing responses from OpenAI chat completions."""

    def __init__(self, async_client: openai.AsyncOpenAI):
        self.async_client = async_client

    @log_timing
    async def parse_completion(
        self,
        *,  # Force keyword arguments
        model: str,
        messages: List[ChatCompletionMessageParam],
        response_format: Type[ResponseModel],
        temperature: float,
        default_response: Optional[ResponseModel] = None,
    ) -> Optional[ResponseModel]:
        """
        Calls the OpenAI API using the beta parse helper and handles errors.

        Args:
            model: The model name to use.
            messages: The list of messages for the prompt.
            response_format: The Pydantic model class for parsing the response.
            temperature: The sampling temperature.
            default_response: The default value to return on failure or if parsing yields None.

        Returns:
            The parsed Pydantic model instance or the default_response.
        """
        format_name = getattr(response_format, "__name__", str(response_format))

        try:
            # Use fully qualified name for logging if it's a Pydantic model
            logger.debug(
                f"Calling LLM (model={model}, temp={temperature}, response_format={format_name})"
            )
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "response_format": response_format,
            }
            if model == "o3-mini":
                # remove temperature
                params.pop("temperature")

            # The .parse() method directly returns the parsed Pydantic model or None
            response = await self.async_client.beta.chat.completions.parse(**params)
            parsed_response = response.choices[0].message.parsed

            if parsed_response is None:
                logger.warning(
                    f"LLM response parsing returned None for {format_name}. Returning default."
                )
                return default_response

            return parsed_response
        except Exception as e:
            logger.error(f"Error during LLM call ({format_name}): {e}", exc_info=True)
            return default_response


class ClarifiedQuery(BaseModel):
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


class FollowUpQuestions(BaseModel):
    questions: List[str] = Field(
        description="A list of 3 follow-up questions that would be natural to ask next based on the answer."
    )


class DecomposedQuery(BaseModel):
    original_query: str
    query_complexity: Literal["simple", "complex"] = Field(
        description="The complexity of the original query."
    )
    decomposed_query: Optional[List[str]] = Field(
        description="The original query decomposed into subqueries."
        "Only meant for unpacking complex queries. If there is nothing to decompose,"
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
    source: str
    query: str
    docs: List[QueryDocument]


class QueryResultList(BaseModel):
    results: List[QueryResult]


class ErrorResult(BaseModel):
    error: str


class ConversationEntry(BaseModel):
    """Represents a single entry in a conversation history."""

    timestamp: datetime
    user_query: str
    answer: str


class DocumentRelevanceEvaluation(BaseModel):
    explanation: str = Field(
        description="Explanation of why the document is relevant or irrelevant to the query"
    )
    is_relevant: bool = Field(
        description="Whether the document contains any information relevant whatsoever to answering the query"
    )


# Add this new Pydantic model for context extraction
class RelevantHistoryContext(BaseModel):
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
    linebreaks: Literal["\\n", "\\n\\n", ""] = Field(
        description="The linebreaks or other formatting in the statement. If there are no linebreaks, return an empty string."
    )

class CitedAnswerResult(BaseModel):
    """A complete answer with structured citations."""

    statements: List[StatementWithCitations] = Field(description="answer to the query")


class PromptManager:
    """
    Loads Jinja templates from a directory, renders them with context,
    and returns a list of OpenAI chat message dicts.
    """

    def __init__(self, templates_dir: str | Path = "prompts"):
        self.env = Environment(
            loader=FileSystemLoader(templates_dir),
            autoescape=select_autoescape(enabled_extensions=("j2",)),
        )
        # Cache compiled templates
        self._cache = {}

    def _get_template(self, name: str):
        if name not in self._cache:
            template_path = f"{name}.yaml.j2"
            self._cache[name] = self.env.get_template(template_path)
        return self._cache[name]

    def messages(self, name: str, **context) -> List[ChatCompletionMessageParam]:
        """
        Render a template file and return a list of messages.
        """
        raw = self._get_template(name).render(**context)
        return yaml.safe_load(raw)


class BaseProcessor:
    """Base class for all LLM-based processors to reduce code duplication."""

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        async_client: openai.AsyncOpenAI = openai.AsyncOpenAI(),
        prompt_manager: Optional[PromptManager] = None,
        parser_service: Optional[LLMParserService] = None,
    ):
        self.llm_model = llm_model
        self.async_client = async_client
        self.pm = prompt_manager or PromptManager()
        self.parser_service = parser_service or LLMParserService(async_client)

    async def _call_llm(
        self,
        prompt_name: str,
        temperature: float = 0.1,
        response_format: Type[ResponseModel] = Any,  # type: ignore
        default_response: Optional[ResponseModel] = None,
        **prompt_args,
    ) -> Optional[ResponseModel]:
        """Standardized method for LLM calls using prompt templates."""
        messages = self.pm.messages(prompt_name, **prompt_args)
        return await self.parser_service.parse_completion(
            model=self.llm_model,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
            default_response=default_response,
        )

    async def _call_llm_completions(
        self,
        prompt_name: str,
        temperature: float = 0.1,
        default_response: str = "",
        **prompt_args,
    ) -> str:
        """Standardized method for LLM streaming using prompt templates."""
        messages = self.pm.messages(prompt_name, **prompt_args)
        response = await self.async_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=temperature,
        )
        if response.choices[0].message.content is None:
            return default_response
        return response.choices[0].message.content


class QueryPreprocessor(BaseProcessor):
    """
    Preprocesses user queries to resolve ambiguities using conversation history.
    Detects and clarifies ambiguous references like "it", "they", "this side effect", etc.
    """

    @log_timing
    async def clarify_query_async(
        self, user_query: str, conversation_context: str = ""
    ):
        logger.info(f"Clarifying query: '{user_query}'")

        if not conversation_context:
            logger.debug("No conversation context, skipping clarification")
            return ClarifiedQuery(
                original_query=user_query,
                clarified_query=user_query,
                ambiguity_level="clear and specific",
            )

        return await self._call_llm(
            prompt_name="clarify_query",
            user_query=user_query,
            conversation_context=conversation_context,
            response_format=ClarifiedQuery,
            default_response=ClarifiedQuery(
                original_query=user_query,
                clarified_query=user_query,
                ambiguity_level="clear and specific",
            ),
        )

    @log_timing
    async def decompose_query_async(self, user_query: str):
        logger.info(f"Decomposing query: '{user_query}'")

        return await self._call_llm(
            prompt_name="decompose_query",
            user_query=user_query,
            response_format=DecomposedQuery,
            default_response=DecomposedQuery(
                original_query=user_query,
                query_complexity="simple",
                decomposed_query=[user_query],
            ),
        )


class ConversationHistory:
    """Manages conversation history for users."""

    def __init__(self, save_directory: str = "data/conversations"):
        """
        Initialize the conversation history manager.

        Args:
            save_directory: Directory to save conversation histories
        """
        self.save_directory = save_directory
        os.makedirs(save_directory, exist_ok=True)
        self.conversations: Dict[str, List[ConversationEntry]] = {}

    def add_entry(self, user_id: str, query: str, answer: str) -> None:
        """
        Add a new entry to a user's conversation history.

        Args:
            user_id: Unique identifier for the user
            query: The user's question
            answer: The system's answer
            sources: List of sources used in the answer
        """
        if user_id not in self.conversations:
            self.conversations[user_id] = []

        entry = ConversationEntry(
            timestamp=datetime.now(),
            user_query=query,
            answer=answer,
        )

        self.conversations[user_id].append(entry)
        self._save_conversation(user_id)

    def get_history(self, user_id: str, limit: int = 5) -> List[ConversationEntry]:
        """
        Get recent conversation history for a user.

        Args:
            user_id: Unique identifier for the user
            limit: Maximum number of entries to return

        Returns:
            List of conversation entries, most recent first
        """
        if user_id not in self.conversations:
            self._load_conversation(user_id)

        history = self.conversations.get(user_id, [])

        # Convert to dict format and limit number of entries
        return [
            ConversationEntry(
                timestamp=entry.timestamp,
                user_query=entry.user_query,
                answer=entry.answer,
            )
            for entry in history[-limit:][::-1]  # Most recent first
        ]

    def get_context_from_history(self, user_id: str, limit: int = 3) -> str:
        """
        Generate a conversation context string from recent history.

        Args:
            user_id: Unique identifier for the user
            limit: Maximum number of past exchanges to include

        Returns:
            Formatted conversation context string
        """
        if user_id not in self.conversations:
            self._load_conversation(user_id)

        history = self.conversations.get(user_id, [])

        if not history:
            return ""

        # Get most recent exchanges
        recent = history[-limit:]

        # Format as conversation context
        context = "Previous conversation:\n"
        for entry in recent:
            context += f"User: {entry.user_query}\n"
            context += f"Assistant: {entry.answer}\n\n"

        return context

    def _save_conversation(self, user_id: str) -> None:
        """
        Save a user's conversation history to disk.

        Args:
            user_id: Unique identifier for the user
        """
        file_path = os.path.join(self.save_directory, f"{user_id}.json")

        # Use model_dump() for serialization
        entries = [
            entry.model_dump(mode="json") for entry in self.conversations[user_id]
        ]

        with open(file_path, "w") as f:
            json.dump(entries, f, indent=2)

    def _load_conversation(self, user_id: str) -> None:
        """
        Load a user's conversation history from disk.

        Args:
            user_id: Unique identifier for the user
        """
        file_path = os.path.join(self.save_directory, f"{user_id}.json")

        if not os.path.exists(file_path):
            self.conversations[user_id] = []
            return

        try:
            with open(file_path, "r") as f:
                entries_data = json.load(f)

            # Use model_validate for deserialization
            self.conversations[user_id] = [
                ConversationEntry.model_validate(entry_data)
                for entry_data in entries_data
            ]
        except Exception as e:
            logger.error(f"Error loading conversation for {user_id}: {e}")
            self.conversations[user_id] = []


class QueryRouter:
    """Routes queries to appropriate Weaviate collections based on content."""

    def __init__(
        self,
        weaviate_client: WeaviateAsyncClient,
        collection_names: List[str],
        llm_model: str = "gpt-4o-mini",
        async_client: openai.AsyncOpenAI = openai.AsyncOpenAI(),
    ):
        """
        Initialize the query router.

        Args:
            weaviate_client: Asynchronous Weaviate client
            collection_names: List of Weaviate collection names
            llm_model: Model name for the routing LLM
            async_client: Shared asynchronous OpenAI client
        """
        self.weaviate_client = weaviate_client
        self.collection_names = collection_names
        self.llm_model = llm_model
        self.async_client = async_client

        # Build tools dynamically based on available collections
        self.tools = []
        for collection_name in collection_names:
            self.tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f"query_{collection_name.lower()}",
                        "description": f"Get information about {collection_name} from the Weaviate database",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": f"The query from the user (verbatim) that should pertain to {collection_name}",
                                }
                            },
                            "required": ["query"],
                        },
                    },
                }
            )

    @log_timing
    async def route_query_async(self, user_query: str) -> List[QueryResult]:
        """
        Routes a user query to appropriate Weaviate collections based on content.

        Args:
            user_query: The user's question

        Returns:
            List of query results including document content and relevance scores
        """
        logger.info(f"Routing query: '{user_query}'")
        results = []

        try:
            # Get tool calls from LLM
            tool_calls = await self._get_tool_calls(user_query)

            if not tool_calls:
                logger.warning("No relevant sources identified for query")
                return results

            # Log routing destinations
            route_destinations = [
                self._extract_collection_name(tool_call.function.name)
                for tool_call in tool_calls
            ]
            logger.info(f"Query routed to: {', '.join(route_destinations)}")

            # Process each tool call
            for tool_call in tool_calls:
                result = await self._process_tool_call(tool_call)
                if result:
                    results.append(result)

        except Exception as e:
            logger.error(f"An error occurred in routing: {e}")

        logger.info(
            f"Routing completed, retrieved {sum(len(r.docs) for r in results)} documents"
        )
        return results

    async def _get_tool_calls(self, user_query: str):
        """Get tool calls from LLM for the query."""
        messages: List[ChatCompletionMessageParam] = [
            {"role": "user", "content": user_query}
        ]
        response = await self.async_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            tools=self.tools,
            tool_choice="auto",
        )
        return response.choices[0].message.tool_calls

    def _extract_collection_name(self, function_name: str) -> str:
        """Extract and normalize collection name from function name."""
        return function_name.split("_", 1)[1].capitalize()

    async def _process_tool_call(self, tool_call) -> Optional[QueryResult]:
        """Process a single tool call and return query result."""
        try:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            query = function_args.get("query")

            collection_name = self._extract_collection_name(function_name)

            # Check if collection exists
            if not await self.weaviate_client.collections.exists(collection_name):
                logger.warning(f"Collection not found: {collection_name}")
                return None

            logger.info(f"Routing to {collection_name} collection for query: {query}")

            # Get the Weaviate collection
            collection = self.weaviate_client.collections.get(collection_name)

            # Perform hybrid search
            response = await collection.query.hybrid(
                query=query,
                query_properties=["contextualized"],
                limit=4,
                alpha=0.65,
                fusion_type=HybridFusion.RELATIVE_SCORE,
                return_metadata=MetadataQuery(score=True),
            )

            # Generate QueryDocument objects from the results
            query_docs = self._create_query_documents(response.objects, collection_name)

            return QueryResult(source=collection_name, query=query, docs=query_docs)

        except Exception as e:
            logger.error(f"Error processing tool call: {e}")
            return None

    def _create_query_documents(
        self, search_results, collection_name: str
    ) -> List[QueryDocument]:
        """Create QueryDocument objects from Weaviate search results."""
        query_docs = []

        for obj in search_results:
            # Extract the document content
            content = obj.properties.get("contextualized", "")

            # Extract distance (lower is better)
            score = obj.metadata.score if obj.metadata else 1.0

            # Use Weaviate UUID as document ID
            doc_id = str(obj.uuid)

            page_numbers = obj.properties.get("page_numbers", None)
            # Extract any additional metadata
            metadata = {
                k: v for k, v in obj.properties.items() if k != "contextualized"
            }

            query_docs.append(
                QueryDocument(
                    content=content,
                    score=score,
                    doc_id=doc_id,
                    metadata=metadata,
                    source_name=collection_name,
                    page_numbers=page_numbers,
                )
            )
        return query_docs


class AnswerGenerationResult(BaseModel):
    plain_answer: str
    retrieval_results: QueryResultList
    formatted_docs: str
    prompt_id_map: Dict[str, str]
    user_question: str
    conversation_context: str


class AnswerGenerator(BaseProcessor):
    """Generates answers from retrieved documents."""

    def _format_documents_for_prompt(
        self, retrieval_results: QueryResultList
    ) -> Tuple[str, Dict[str, str]]:
        doc_context = ""
        prompt_id_to_original_id_map: Dict[str, str] = {}
        doc_index = 0

        if not retrieval_results or not retrieval_results.results:
            return "", {}

        for result in retrieval_results.results:
            for doc in result.docs:
                original_doc_id = doc.doc_id or f"missing_id_{uuid4()}"
                prompt_doc_id = f"doc_{doc_index + 1}"
                prompt_id_to_original_id_map[prompt_doc_id] = original_doc_id

                doc_context += f"Document ID: [{prompt_doc_id}]\n"
                doc_context += f"Content: {doc.content}\n"
                doc_context += f"Source: {doc.source_name}\n"
                if doc.page_numbers:
                    doc_context += f"Page Numbers: {doc.page_numbers}\n"
                doc_context += "---\n"
                doc_index += 1

        return doc_context.strip(), prompt_id_to_original_id_map

    def _create_result(
        self,
        plain_answer: str,
        retrieval_results: QueryResultList,
        formatted_docs: str = "",
        prompt_id_map: Optional[Dict[str, str]] = None,
        user_question: str = "",
        conversation_context: str = "",
    ) -> AnswerGenerationResult:
        """Helper method to create AnswerGenerationResult with provided values."""
        return AnswerGenerationResult(
            plain_answer=plain_answer,
            retrieval_results=retrieval_results,
            formatted_docs=formatted_docs,
            prompt_id_map=prompt_id_map or {},
            user_question=user_question,
            conversation_context=conversation_context,
        )

    @log_timing
    async def generate_answer_async(
        self,
        user_question: str,
        retrieval_results: QueryResultList,
        conversation_context: str = "",
    ) -> AnswerGenerationResult:
        """
        Generates a plain text answer and packages it with context for validation.
        """
        default_error_answer = "I'm sorry, I don't know the answer to that question."

        # Initialize result with common parameters
        result = AnswerGenerationResult(
            plain_answer=default_error_answer,
            retrieval_results=retrieval_results or QueryResultList(results=[]),
            formatted_docs="",
            prompt_id_map={},
            user_question=user_question,
            conversation_context=conversation_context,
        )

        if not retrieval_results or not retrieval_results.results:
            logger.warning("No retrieval results provided.")
            return result

        # Format documents for the prompt and get the ID map
        formatted_docs, prompt_id_map = self._format_documents_for_prompt(
            retrieval_results
        )

        # Update result with formatted docs and prompt ID map
        result.formatted_docs = formatted_docs
        result.prompt_id_map = prompt_id_map

        if not formatted_docs:
            logger.warning("Formatted documents are empty.")
            result.plain_answer = "I encountered an issue processing the information."
            return result

        logger.info(f"Generating plain text answer for query: '{user_question}'")

        # Generate the plain answer string
        plain_answer = await self._call_llm_completions(
            prompt_name="answer_generation",
            user_question=user_question,
            retrieval_results=formatted_docs,
            conversation_context=conversation_context or "",
            temperature=0.1,
            default_response=default_error_answer,
        )

        # Update result with generated answer
        result.plain_answer = plain_answer

        return result


class AnswerValidator(BaseProcessor):
    """
    Structures a plain text answer using LLM and validates its citations
    against the original retrieved documents.
    """

    def _find_document_by_id(
        self, doc_id: str, retrieval_results: QueryResultList
    ) -> Optional[QueryDocument]:
        """Finds a QueryDocument within the QueryResultList by its original ID."""
        for result in retrieval_results.results:
            for doc in result.docs:
                if doc.doc_id == doc_id:
                    return doc
        return None

    def _verify_quote(
        self, quote: str, document_content: str, threshold: int = 85
    ) -> bool:
        """Verifies if the quote exists in the document content using fuzzy matching."""
        if not quote or not document_content:
            return False
        if quote in document_content:
            return True
        # Ensure fuzzywuzzy is imported: from fuzzywuzzy import process as fuzzy_process
        match_result = fuzzy_process.extractOne(
            quote, [document_content], score_cutoff=threshold
        )
        return match_result is not None

    def _resolve_citation_ids(
        self, answer: CitedAnswerResult, prompt_id_map: Dict[str, str]
    ) -> CitedAnswerResult:
        """
        Resolves temporary prompt IDs (e.g., 'doc_1') in citations back to
        original document IDs using the provided map.
        """
        for statement in answer.statements:
            for citation in statement.citations:
                prompt_id = citation.doc_id  # Expecting [doc_X] from structuring LLM
                original_id = prompt_id_map.get(prompt_id)  # Use the correct map here
                if original_id:
                    citation.doc_id = original_id
                else:
                    logger.warning(
                        f"Could not resolve prompt ID '{prompt_id}' during validation. Citation will likely fail."
                    )
                    # Keep the invalid prompt_id; validation check below will fail anyway if doc not found
        return answer

    def _validate_citations_and_build_answer(
        self,
        structured_answer,
        retrieval_results,
        original_id_to_prompt_id_map,
        quote_match_threshold=85,
    ):
        """Validates citations and builds final answer with validated statements."""
        if not structured_answer or not structured_answer.statements:
            return self._get_fallback_message()

        validated_statements = []
        stats = {"total_checked": 0, "invalid_count": 0}

        for idx, statement in enumerate(structured_answer.statements):
            processed_statement = self._process_statement(
                statement,
                idx,
                retrieval_results,
                original_id_to_prompt_id_map,
                quote_match_threshold,
                stats,
            )
            if processed_statement:
                validated_statements.append(processed_statement)

        logger.info(
            f"Citation check summary: Total checked: {stats['total_checked']}, Individual citation failures: {stats['invalid_count']}"
        )

        return (
            " ".join(validated_statements)
            if validated_statements
            else self._get_fallback_message()
        )

    def _get_fallback_message(self):
        """Returns standard fallback message when validation fails."""
        return "I'm sorry, I couldn't validate the information to answer your question."

    def _process_statement(
        self,
        statement,
        statement_idx,
        retrieval_results,
        original_id_to_prompt_id_map,
        quote_match_threshold,
        stats,
    ):
        """Processes a single statement and its citations."""
        if not statement.citations:
            logger.debug(
                f"Statement {statement_idx} has no citations, considered valid."
            )
            return self._format_statement(statement.text, [], statement.linebreaks)

        valid_prompt_ids = self._validate_statement_citations(
            statement,
            statement_idx,
            retrieval_results,
            original_id_to_prompt_id_map,
            quote_match_threshold,
            stats,
        )

        # Statement is valid if at least one citation is valid
        if valid_prompt_ids:
            return self._format_statement(statement.text, valid_prompt_ids, statement.linebreaks)

        logger.warning(
            f"Statement {statement_idx} is invalid because all citations failed validation."
        )
        return None

    def _validate_statement_citations(
        self,
        statement,
        statement_idx,
        retrieval_results,
        original_id_to_prompt_id_map,
        quote_match_threshold,
        stats,
    ):
        """Validates all citations for a statement and returns valid prompt IDs."""
        valid_prompt_ids = []

        for citation_idx, citation in enumerate(statement.citations):
            stats["total_checked"] += 1

            # Validate this citation
            original_id = citation.doc_id
            is_valid, prompt_id = self._validate_citation(
                citation,
                citation_idx,
                statement_idx,
                original_id,
                retrieval_results,
                original_id_to_prompt_id_map,
                quote_match_threshold,
            )

            if is_valid and prompt_id:
                valid_prompt_ids.append(prompt_id)
            else:
                stats["invalid_count"] += 1

        return valid_prompt_ids

    def _validate_citation(
        self,
        citation,
        citation_idx,
        statement_idx,
        original_id,
        retrieval_results,
        original_id_to_prompt_id_map,
        quote_match_threshold,
    ):
        """Validates an individual citation and returns (is_valid, prompt_id)."""
        cited_document = self._find_document_by_id(original_id, retrieval_results)

        if not cited_document:
            logger.warning(
                f"Validation failed: Document ID '{original_id}' not found for statement {statement_idx}, citation {citation_idx}."
            )
            return False, None

        is_quote_valid = self._verify_quote(
            citation.quote, cited_document.content, threshold=quote_match_threshold
        )

        if not is_quote_valid:
            logger.warning(
                f"Validation failed: Quote not found in doc ID '{original_id}' for statement {statement_idx}, citation {citation_idx}. Quote: '{citation.quote[:100]}...'"
            )
            return False, None

        # Citation is valid, get corresponding prompt_id
        prompt_id = original_id_to_prompt_id_map.get(original_id)
        if not prompt_id:
            logger.error(
                f"Consistency Error: Validated original ID '{original_id}' not found in original_id_to_prompt_id_map for statement {statement_idx}."
            )
            return True, None  # Citation is valid but can't find prompt_id

        logger.debug(
            f"Statement {statement_idx}, citation {citation_idx} (Original ID: {original_id}, Prompt ID: {prompt_id}) validated successfully."
        )
        return True, prompt_id

    def _format_statement(self, statement_text, valid_prompt_ids, linebreaks):
        """Formats a statement with its valid citation markers."""
        # Remove any old [doc_X] style markers
        cleaned_text = re.sub(r"\[doc_\d+\]", "", statement_text).strip()

        citations_str = ""
        if valid_prompt_ids:
            # Sort prompt IDs naturally and deduplicate
            sorted_prompt_ids = sorted(
                list(set(valid_prompt_ids)), key=lambda x: int(x.split("_")[1])
            )
            citations_str = " ".join([f"[{pid}]" for pid in sorted_prompt_ids])

        # Validate and convert the linebreaks string to actual newline characters
        actual_linebreak = ""
        if linebreaks == "\\n":
            actual_linebreak = "\n"
        elif linebreaks == "\\n\\n":
            actual_linebreak = "\n\n"
        elif linebreaks != "":
            # Log a warning if the value is unexpected but non-empty
            logger.warning(f"Unexpected linebreak value received: {linebreaks!r}. Treating as no linebreak.")
            actual_linebreak = ""
        # If linebreaks is "", actual_linebreak remains ""

        # Construct the final string, adding a space before citations only if needed
        parts = []
        if cleaned_text:
            parts.append(cleaned_text)
        if citations_str:
            parts.append(citations_str)

        # Join parts with a space and append the actual linebreak character(s)
        formatted_statement = " ".join(parts) + actual_linebreak

        return formatted_statement

    @log_timing
    async def structure_and_validate_async(
        self,
        plain_answer: str,
        retrieval_results: QueryResultList,
        formatted_docs: str,
        prompt_id_map: Dict[str, str],
        quote_match_threshold: int = 85,
    ) -> Tuple[Optional[CitedAnswerResult], Optional[str]]:
        """
        Structures the plain answer and validates citations using provided context.

        Returns:
            Tuple containing:
            - The structured answer (CitedAnswerResult) with original IDs resolved.
            - A validated answer string with invalid statements removed and
              validated citations appended using prompt IDs (e.g., '[doc_1]').
        """
        # Input validation
        if (
            not plain_answer
            or not retrieval_results
            or not retrieval_results.results
            or not formatted_docs
            or not prompt_id_map
        ):
            logger.warning("Missing required inputs for structuring and validation.")
            return None, None

        logger.info("Attempting to structure the plain text answer.")

        # Step 1: Structure the plain answer using LLM
        structured_answer = await self._call_llm(
            prompt_name="answer_structuring",
            answer=plain_answer,
            retrieval_results=formatted_docs,
            temperature=0.0,
            response_format=CitedAnswerResult,
            default_response=None,
        )

        if structured_answer is None:
            logger.error("Failed to structure the answer using LLM.")
            return None, None

        logger.info("Answer structured successfully. Proceeding to validation.")

        # Step 2: Resolve prompt IDs to original IDs within the structured answer object
        resolved_structured_answer = self._resolve_citation_ids(
            structured_answer, prompt_id_map
        )

        # Step 2.5: Create the inverse map: original_id -> prompt_id
        # Needed to append the correct marker ([doc_X]) after validation
        original_id_to_prompt_id_map = {v: k for k, v in prompt_id_map.items()}

        # Step 3: Validate citations and build the final validated answer string
        # Pass both the resolved answer and the inverse map
        validated_answer = self._validate_citations_and_build_answer(
            resolved_structured_answer,
            retrieval_results,
            original_id_to_prompt_id_map,
            quote_match_threshold,
        )

        logger.info("Validation and final answer string construction complete.")
        # Return the resolved structured answer (still useful potentially) and the final string
        return resolved_structured_answer, validated_answer


class RetrievalEvaluator(BaseProcessor):
    """Evaluates if retrieved documents contain sufficient information to answer the query."""

    def _prepare_context(
        self, retrieval_results: List[QueryResult]
    ) -> Tuple[str, List[str]]:
        """Extract content and source names from retrieval results."""
        combined_context = ""
        sources = []

        for result in retrieval_results:
            combined_context += "\n\n" + "\n\n".join(
                [doc.content for doc in result.docs]
            )
            sources.append(result.source)

        return combined_context, sources

    async def _fetch_additional_results(
        self, additional_queries: List[str], router: QueryRouter
    ) -> List[QueryResult]:
        """Fetch additional results for the specified queries."""
        if not additional_queries:
            return []

        logger.info(f"Executing {len(additional_queries)} follow-up queries")
        for i, query in enumerate(additional_queries):
            logger.debug(f"  Follow-up query {i+1}: '{query}'")

        try:
            routing_tasks = [
                router.route_query_async(query) for query in additional_queries
            ]
            results_list = await asyncio.gather(*routing_tasks)

            # Flatten the results
            additional_results = []
            for results in results_list:
                additional_results.extend(results)

            return additional_results
        except Exception as e:
            logger.error(f"Error fetching additional results: {e}")
            return []

    @log_timing
    async def evaluate_retrieval(
        self,
        original_query: str,
        clarified_query: str,
        retrieval_results: QueryResultList,
        router: QueryRouter,
    ) -> QueryResultList:
        """Evaluate if retrieval results are sufficient and get additional information if needed."""
        doc_count = sum(len(r.docs) for r in retrieval_results.results)
        logger.info(
            f"Evaluating retrieval results: {doc_count} documents from {len(retrieval_results.results)} sources"
        )

        # Skip evaluation if no results
        if not retrieval_results:
            return retrieval_results

        # Prepare context from retrieval results
        context, sources = self._prepare_context(retrieval_results.results)

        # Use the base class _call_llm method instead of custom _evaluate_with_llm
        evaluation = await self._call_llm(
            prompt_name="retrieval_evaluation",
            original_query=original_query,
            clarified_query=clarified_query,
            retrieved_information=context,
            sources=", ".join(sources),
            temperature=0.1,
            response_format=RetrievalEvaluation,
            default_response=RetrievalEvaluation(
                is_sufficient=True, missing_information="", additional_queries=[]
            ),
        )

        # Process evaluation result
        enhanced_results = retrieval_results  # Create a copy to avoid modifying the original


        if (
            evaluation
            and not evaluation.is_sufficient
            and evaluation.additional_queries
        ):
            logger.info(
                f"Retrieval insufficient. Missing info: {evaluation.missing_information}"
            )

            # Fetch additional results
            additional_results = await self._fetch_additional_results(
                evaluation.additional_queries, router
            )

            if additional_results:
                enhanced_results.results.extend(additional_results)
                logger.info(
                    f"Added {len(additional_results)} additional retrieval results"
                )
        else:
            logger.info("Retrieved information is sufficient or evaluation failed")

        final_doc_count = sum(len(r.docs) for r in enhanced_results.results)
        logger.info(f"Evaluation completed, final document count: {final_doc_count}")

        return enhanced_results


# Add this new class for handling conversation context extraction
class ConversationContextProcessor(BaseProcessor):
    """Processes the conversation history to extract only relevant context for the current query."""

    @log_timing
    async def extract_relevant_context(
        self, query: str, conversation_history: List[ConversationEntry]
    ) -> RelevantHistoryContext:
        """
        Extract only the relevant parts of conversation history for the current query.

        Args:
            query: The current user query (clarified)
            conversation_history: Full conversation history

        Returns:
            RelevantHistoryContext with the formatted context string and relevant entries
        """
        if not conversation_history:
            return RelevantHistoryContext(
                required_context=False,
                explanation="No conversation history available",
                relevant_snippets="",
            )

        # Format conversation history for the prompt
        history_text = "Previous conversation:\n"
        for i, entry in enumerate(conversation_history):
            history_text += f"[{i+1}] User: {entry.user_query}\n"
            history_text += f"    Assistant: {entry.answer}\n\n"

        # Use the base class _call_llm method instead of direct parser calls
        default_response = RelevantHistoryContext(
            required_context=False,
            explanation="No relevant context found",
            relevant_snippets="",
        )

        response = await self._call_llm(
            prompt_name="context_extraction",
            current_query=query,
            history_text=history_text,
            temperature=0.1,
            response_format=RelevantHistoryContext,
            default_response=default_response,
        )

        if response is None:
            return default_response

        return response


class FollowUpQuestionsGenerator(BaseProcessor):
    """Generates follow-up questions based on the answer and conversation history."""

    @log_timing
    async def generate_follow_up_questions(
        self,
        query: str,
        answer: str,
        conversation_history: Optional[List[ConversationEntry]] = None,
    ) -> FollowUpQuestions:
        """
        Generate three potential follow-up questions based on the answer and conversation history.

        Args:
            query: The original user query
            answer: The answer provided to the user
            conversation_history: Optional list of previous conversation entries

        Returns:
            A FollowUpQuestions object containing a list of 3 follow-up questions
        """
        logger.info(f"Generating follow-up questions for query: '{query}'")

        # Format conversation history if provided
        history_context = ""
        if conversation_history and len(conversation_history) > 0:
            history_context = "Previous conversation:\n"
            for i, entry in enumerate(conversation_history):
                history_context += f"User: {entry.user_query}\n"
                history_context += f"Assistant: {entry.answer}\n\n"
            logger.debug(
                f"Using {len(conversation_history)} conversation entries for context"
            )

        # Use the base class _call_llm method instead of direct calls to parser_service
        default_response = FollowUpQuestions(questions=[])

        response = await self._call_llm(
            prompt_name="follow_up_questions",
            original_query=query,
            answer=answer,
            history_context=history_context,
            temperature=0.3,  # Higher temperature for more variety
            response_format=FollowUpQuestions,
            default_response=default_response,
        )

        if response is None:
            return default_response

        return response


class MedicalRAG:
    """
    A medical RAG system that integrates Weaviate, query routing, answer generation,
    and conversation history.
    """

    def __init__(
        self,
        weaviate_client: WeaviateAsyncClient,
        collection_names: List[str],
        llm_model: str = "gpt-4o-mini",
        parser_service: Optional[LLMParserService] = None,
        conversation_history_dir: str = "data/conversations",
        prompts_dir: str = "prompts",
    ):
        """
        Initialize the Medical RAG system with a simplified configuration.

        Args:
            weaviate_client: The Weaviate async client
            collection_names: Names of collections in Weaviate to query
            llm_model: Single model name used for all LLM components
            parser_service: Service for making parsed LLM calls
            conversation_history_dir: Directory to store conversation histories
            prompts_dir: Directory containing Jinja templates for prompts
        """
        # Set up shared resources
        self.async_client = openai.AsyncOpenAI()
        self.prompt_manager = PromptManager(prompts_dir)
        self.parser_service = parser_service or LLMParserService(self.async_client)

        # Store the Weaviate client
        self.weaviate_client = weaviate_client

        # Initialize the router with Weaviate client and collection names
        self.router = QueryRouter(
            weaviate_client=weaviate_client,
            collection_names=collection_names,
            llm_model=llm_model,
            async_client=self.async_client,
        )

        # All BaseProcessor components share the same model and resources
        common_args = {
            "llm_model": llm_model,
            "async_client": self.async_client,
            "prompt_manager": self.prompt_manager,
            "parser_service": self.parser_service,
        }

        # Create validator args with a different model
        validator_args = common_args.copy()
        validator_args["llm_model"] = (
            "gpt-4o"  # Using a more capable model for validation
        )

        self.generator = AnswerGenerator(**common_args)
        self.preprocessor = QueryPreprocessor(**common_args)
        self.evaluator = RetrievalEvaluator(**common_args)
        self.context_processor = ConversationContextProcessor(**common_args)
        self.follow_up_generator = FollowUpQuestionsGenerator(**common_args)
        self.validator = AnswerValidator(**validator_args)

        # Initialize conversation history
        self.conversation_history = ConversationHistory(conversation_history_dir)

    def get_conversation_history(
        self, user_id: str, limit: int = 5
    ) -> List[ConversationEntry]:
        """
        Get recent conversation history for a user.

        Args:
            user_id: Unique identifier for the user
            limit: Maximum number of entries to return

        Returns:
            List of conversation entries, most recent first
        """
        return self.conversation_history.get_history(user_id, limit)

    @log_timing
    async def process_query_simple(
        self, user_id: str, user_query: str, use_history: bool = True
    ) -> Tuple[str, List[str]]:
        """
        Processes query, returns fast text answer & follow-ups. Validation runs async.
        """
        logger.info(f"Processing query for user '{user_id}': '{user_query}'")

        # Get conversation context if needed
        conversation_context = ""
        history = []
        if use_history:
            history = self.conversation_history.get_history(user_id)
            if history:
                logger.info(f"Found {len(history)} historical entries for user")
                ctx_result = await self.context_processor.extract_relevant_context(
                    user_query, history
                )
                conversation_context = ctx_result.relevant_snippets
                logger.info(
                    f"Context extraction result: required={ctx_result.required_context}"
                )

        queries = [user_query]
        results_list = await asyncio.gather(
            *[self.router.route_query_async(q) for q in queries]
        )
        retrieval_results = QueryResultList(
            results=list(chain.from_iterable(results_list))
        )
        current_results = retrieval_results

        generation_result: AnswerGenerationResult = (
            await self.generator.generate_answer_async(
                user_question=user_query,
                retrieval_results=current_results,
                conversation_context=conversation_context or "",
            )
        )
        logger.info("Plain text answer and context generated.")

        initial_answer = generation_result.plain_answer
        final_answer = initial_answer  # Default to initial answer
        print(f"Initial answer: {initial_answer}")
        if initial_answer != "I'm sorry, I don't know the answer to that question.":
            # Run structuring and validation
            logger.info("Starting structuring and validation...")
            structured_answer, validated_answer = (
                await self.validator.structure_and_validate_async(
                    plain_answer=initial_answer,
                    retrieval_results=generation_result.retrieval_results,
                    formatted_docs=generation_result.formatted_docs,
                    prompt_id_map=generation_result.prompt_id_map,
                )
            )

            print(f"Validated answer: {validated_answer}")
            # Use validated answer if available and non-empty
            if validated_answer:
                final_answer = validated_answer
                logger.info("Using validated answer with invalid statements removed.")
            elif structured_answer is None:
                logger.warning(
                    "Structuring failed. Falling back to initial unvalidated answer."
                )
                # final_answer remains initial_answer
            else:
                logger.warning(
                    "Validation resulted in an empty answer. Falling back to initial unvalidated answer."
                )
                # final_answer remains initial_answer
        else:
            logger.info("Skipping structuring/validation due to default error answer.")

        follow_ups = await self.follow_up_generator.generate_follow_up_questions(
            user_query,
            final_answer,  # Use the final (potentially validated) answer
            history if use_history else None,
        )

        # Save the final (potentially validated) answer to conversation history
        self.conversation_history.add_entry(user_id, user_query, final_answer)

        return final_answer, follow_ups.questions


# Application setup and usage
async def setup_medical_rag() -> MedicalRAG:
    """Set up and return a configured MedicalRAG instance."""
    # Define default Weaviate connection parameters
    weaviate_host = os.getenv("WEAVIATE_HOST", "127.0.0.1")
    weaviate_port = int(os.getenv("WEAVIATE_PORT", "8080"))
    weaviate_grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

    # Get OpenAI API key
    openai_key = os.getenv("OPENAI_APIKEY")
    if not openai_key:
        logger.error("OPENAI_APIKEY environment variable not set.")
        raise ValueError("OPENAI_APIKEY environment variable not set.")

    # Set up headers for OpenAI
    headers = {"X-OpenAI-Api-Key": openai_key}

    logger.info(f"Connecting to Weaviate at {weaviate_host}:{weaviate_port}")

    # Create and connect to Weaviate client
    client = WeaviateAsyncClient(
        connection_params=ConnectionParams.from_params(
            http_host=weaviate_host,
            http_port=weaviate_port,
            http_secure=False,
            grpc_host=weaviate_host,
            grpc_port=weaviate_grpc_port,
            grpc_secure=False,
        ),
        additional_headers=headers,
    )

    # Connect to Weaviate
    await client.connect()
    logger.info("Connected to Weaviate successfully")

    # Define collection names to use (previously FAISS store names)
    collection_names = ["Lipitor", "Metformin"]

    # Create and return MedicalRAG instance
    return MedicalRAG(
        weaviate_client=client,
        collection_names=collection_names,
        llm_model="gpt-4o-mini",
    )


# Example usage
if __name__ == "__main__":
    # Configure more verbose logging for the example run
    logger.info("Starting Medical RAG example")

    async def main():
        # Set up the medical RAG system
        medical_rag = await setup_medical_rag()
        logger.info("Medical RAG system initialized")

        # Simulate a conversation with a user
        user_id = "test_user"

        # Simple test query
        test_query = "What is lipitor?"
        logger.info(f"Testing with query: '{test_query}'")

        # Process the query and measure time
        start_time = time.time()
        answer, follow_ups = await medical_rag.process_query_simple(user_id, test_query)
        elapsed = time.time() - start_time

        # Display results
        logger.info(f"Query processed in {elapsed:.2f}s")
        logger.info(f"Answer: {answer}")
        logger.info(f"Follow-ups: {', '.join(follow_ups)}")

        # Close the Weaviate client
        await medical_rag.weaviate_client.close()

    # Run the async main function
    asyncio.run(main())
