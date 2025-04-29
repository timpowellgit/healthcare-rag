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
)
from datetime import datetime
from langchain_core.documents import Document
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import os
import openai
from openai.types.chat import ChatCompletionMessageParam

from pydantic import BaseModel, model_validator, Field
from typing import Self, Type, TypeVar
import asyncio
from itertools import chain  # (used later)
import logging
import time
from pydantic.json_schema import SkipJsonSchema
from fuzzywuzzy import fuzz
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path

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

            # The .parse() method directly returns the parsed Pydantic model or None
            response = await self.async_client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format=response_format,
            )
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
    """Represents a single document retrieved from a vector store."""

    content: str
    relevance_score: float = Field(
        description="The relevance score provided by the similarity search."
    )
    doc_id: str = Field(description="Unique identifier for this document chunk.")
    source_name: str = Field(
        description="The name of the source (e.g., 'Lipitor', 'Metformin')."
    )
    metadata: Optional[Dict[str, Any]] = None


class QueryResult(BaseModel):
    source: str
    query: str
    docs: List[QueryDocument]


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


class CitedAnswerResult(BaseModel):
    """A complete answer with structured citations."""

    statements: List[StatementWithCitations] = Field(
        description="List of statements to answer the query, each with its supporting citations."
    )


class VectorStoreManager:
    """Manages loading and creation of vector stores."""

    def __init__(self, embedding_model: str = "text-embedding-3-large"):
        """
        Initialize the vector store manager.

        Args:
            embedding_model: Model name for embeddings
        """
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

    def load_or_create_store(
        self, save_path: str, data_path: Optional[str] = None
    ) -> FAISS:
        """
        Load an existing vector store or create a new one if it doesn't exist.

        Args:
            save_path: Path to save/load the FAISS index
            data_path: Path to the source data JSON for creation (optional)

        Returns:
            A FAISS vector store
        """
        if os.path.exists(save_path):
            logger.info(f"Loading existing index from {save_path}")
            return FAISS.load_local(
                save_path, self.embeddings, allow_dangerous_deserialization=True
            )
        else:
            if not data_path:
                raise ValueError(
                    "Data path must be provided to create a new vector store"
                )

            logger.info(f"Creating new index at {save_path}")
            # Create initial index structure
            index = faiss.IndexFlatL2(len(self.embeddings.embed_query("test")))
            vector_store = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )

            # Load and process chunks
            chunks = json.load(open(data_path))
            docs = [
                Document(
                    page_content=chunk["contextualized"], metadata=chunk["metadata"]
                )
                for chunk in chunks
            ]
            uuids = [str(uuid4()) for _ in range(len(docs))]

            # Add documents and save
            vector_store.add_documents(documents=docs, ids=uuids)
            vector_store.save_local(save_path)
            logger.info(f"Index saved to {save_path}")

            return vector_store


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
        response_format: Type[ResponseModel] = Any,
        default_response: Optional[ResponseModel] = None,
        **prompt_args
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
    """Routes queries to appropriate vector stores based on content."""

    def __init__(
        self,
        stores: Dict[str, FAISS],
        llm_model: str = "gpt-4o-mini",
        async_client: openai.AsyncOpenAI = openai.AsyncOpenAI(),
    ):
        """
        Initialize the query router.

        Args:
            stores: Dictionary mapping store names to FAISS vector stores
            llm_model: Model name for the routing LLM
            async_client: Shared asynchronous OpenAI client
        """
        self.stores = stores
        self.llm_model = llm_model
        self.async_client = async_client

        # Build tools dynamically based on available stores
        self.tools = []
        for store_name in stores.keys():
            self.tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f"query_{store_name.lower()}",
                        "description": f"Get information about {store_name}",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": f"The query from the user (verbatim) that should pertain to {store_name}",
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
        Routes a user query to appropriate vector stores based on content.

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
                self._extract_store_name(tool_call.function.name)
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

    def _extract_store_name(self, function_name: str) -> str:
        """Extract and normalize store name from function name."""
        return function_name.split("_", 1)[1].capitalize()

    def _get_store(self, store_name: str):
        """Get the correct store based on name, handling case differences."""
        normalized_name = store_name.lower()
        for name, store in self.stores.items():
            if name.lower() == normalized_name:
                return store
        return None

    async def _process_tool_call(self, tool_call) -> Optional[QueryResult]:
        """Process a single tool call and return query result."""
        try:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            query = function_args.get("query")

            store_name = self._extract_store_name(function_name)
            store = self._get_store(store_name)

            if not store:
                logger.warning(f"Store not found for: {store_name}")
                return None

            logger.info(f"Routing to {store_name} vector store for query: {query}")

            # Get search results
            search_results = store.similarity_search_with_score(query, k=3)
            query_docs = self._create_query_documents(search_results, store_name)

            return QueryResult(source=store_name, query=query, docs=query_docs)

        except Exception as e:
            logger.error(f"Error processing tool call: {e}")
            return None

    def _create_query_documents(
        self, search_results, store_name: str
    ) -> List[QueryDocument]:
        """Create QueryDocument objects from search results."""
        query_docs = []

        for i, (doc, score) in enumerate(search_results):
            # Extract or generate doc_id
            doc_id = self._extract_doc_id(doc, store_name, i)

            query_docs.append(
                QueryDocument(
                    content=doc.page_content,
                    relevance_score=score,
                    doc_id=doc_id,
                    metadata=doc.metadata,
                    source_name=store_name,
                )
            )
        return query_docs

    def _extract_doc_id(self, doc, store_name: str, index: int) -> str:
        """Extract document ID from metadata or generate a new one."""
        if hasattr(doc, "metadata") and doc.metadata:
            doc_id = doc.metadata.get("id") or doc.metadata.get("_id")
            if doc_id:
                return str(doc_id)

        return f"{store_name.lower()}_{index}_{uuid4()}"


class AnswerGenerator(BaseProcessor):
    """Generates answers from retrieved documents."""

    def _format_documents_for_prompt(self, retrieval_results: List[QueryResult]) -> str:
        """Format documents from retrieval results into a string for the LLM prompt."""
        doc_context = ""
        for result in retrieval_results:
            for doc in result.docs:
                doc_id = doc.doc_id or f"doc_{uuid4()}"
                doc_context += f"[{doc_id}] Source: {doc.source_name}\n"
                doc_context += f"Content: {doc.content}\n\n"
        return doc_context

    def _validate_citations(
        self, cited_answer: CitedAnswerResult, retrieval_results: List[QueryResult]
    ):
        """Validate citations in the answer against retrieved documents using fuzzy matching."""
        # Create a document lookup map for efficient access
        doc_map = {}
        for result in retrieval_results:
            for doc in result.docs:
                doc_map[doc.doc_id] = doc

        match_threshold = 85  # Adjust as needed - higher is stricter

        for statement in cited_answer.statements:
            for citation in statement.citations:
                # Direct lookup instead of nested loops
                if citation.doc_id in doc_map:
                    doc = doc_map[citation.doc_id]
                    # Use partial_ratio for best substring matching
                    similarity = fuzz.partial_ratio(citation.quote, doc.content)
                    if similarity >= match_threshold:
                        logger.debug(f"Citation validated with {similarity}% match")
                    else:
                        logger.warning(
                            f"Citation failed validation: {similarity}% match below threshold of {match_threshold}%"
                        )
                        logger.debug(f"Quote: '{citation.quote[:50]}...'")
                else:
                    logger.warning(
                        f"Invalid citation: doc_id={citation.doc_id}, source={citation.source_name} - document not found"
                    )

    @log_timing
    async def generate_answer_async(
        self,
        user_question: str,
        retrieval_results: List[QueryResult],
        conversation_context: str = "",
    ) -> CitedAnswerResult:
        """Generates an answer with citations from retrieved documents."""
        if not retrieval_results:
            return CitedAnswerResult(statements=[])
            
        # Format documents for the prompt
        doc_context = self._format_documents_for_prompt(retrieval_results)
        
        logger.info(f"Generating cited answer for query: '{user_question}'")
        default_response = CitedAnswerResult(statements=[])
        
        # Use the base class _call_llm method instead of direct calls
        response = await self._call_llm(
            prompt_name="answer_generation",
            user_question=user_question,
            doc_context=doc_context,
            conversation_context=conversation_context,
            temperature=0.0,  # Low temperature for factual accuracy
            response_format=CitedAnswerResult,
            default_response=default_response,
        )
        
        if response is None:
            return default_response

        # Validate citations against source documents
        self._validate_citations(response, retrieval_results)
        return response


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
        retrieval_results: List[QueryResult],
        router: QueryRouter,
    ) -> List[QueryResult]:
        """Evaluate if retrieval results are sufficient and get additional information if needed."""
        doc_count = sum(len(r.docs) for r in retrieval_results)
        logger.info(
            f"Evaluating retrieval results: {doc_count} documents from {len(retrieval_results)} sources"
        )

        # Skip evaluation if no results
        if not retrieval_results:
            return retrieval_results

        # Prepare context from retrieval results
        context, sources = self._prepare_context(retrieval_results)

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
            )
        )

        # Process evaluation result
        enhanced_results = list(retrieval_results)  # Create a copy to avoid modifying the original

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
                enhanced_results.extend(additional_results)
                logger.info(
                    f"Added {len(additional_results)} additional retrieval results"
                )
        else:
            logger.info("Retrieved information is sufficient or evaluation failed")

        final_doc_count = sum(len(r.docs) for r in enhanced_results)
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
    A medical RAG system that integrates vector stores, query routing, answer generation,
    and conversation history.
    """

    def __init__(
        self,
        vector_stores: Dict[str, FAISS],
        llm_model: str = "gpt-4o-mini",
        parser_service: Optional[LLMParserService] = None,
        conversation_history_dir: str = "data/conversations",
        prompts_dir: str = "prompts",
    ):
        """
        Initialize the Medical RAG system with a simplified configuration.
        
        Args:
            vector_stores: Dictionary mapping store names to FAISS vector stores
            llm_model: Single model name used for all LLM components
            parser_service: Service for making parsed LLM calls
            conversation_history_dir: Directory to store conversation histories
            prompts_dir: Directory containing Jinja templates for prompts
        """
        # Set up shared resources
        self.async_client = openai.AsyncOpenAI()
        self.prompt_manager = PromptManager(prompts_dir)
        self.parser_service = parser_service or LLMParserService(self.async_client)
        
        # Initialize the router (special case as it doesn't use BaseProcessor)
        self.router = QueryRouter(
            vector_stores, 
            llm_model, 
            self.async_client
        )
        
        # All BaseProcessor components share the same model and resources
        common_args = {
            "llm_model": llm_model,
            "async_client": self.async_client,
            "prompt_manager": self.prompt_manager,
            "parser_service": self.parser_service
        }
        
        self.generator = AnswerGenerator(**common_args)
        self.preprocessor = QueryPreprocessor(**common_args)
        self.evaluator = RetrievalEvaluator(**common_args)
        self.context_processor = ConversationContextProcessor(**common_args)
        self.follow_up_generator = FollowUpQuestionsGenerator(**common_args)
        
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
    async def process_query_optimistic_async(
        self, user_id: str, user_query: str, use_history: bool = True
    ):
        """Processes a user query using an optimistic approach with early answers and refinement."""
        logger.info(
            f"Processing query optimistically for user '{user_id}': '{user_query}'"
        )
        logger.info(f"History enabled: {use_history}")

        # Store the current user_id for use in other methods
        self._current_user_id = user_id

        overall_start = time.time()
        history = self.conversation_history.get_history(user_id) if use_history else []
        if history:
            logger.info(f"Found {len(history)} historical entries for user")

        async with asyncio.TaskGroup() as tg:
            # Phase 1: Launch initial parallel tasks
            tasks = await self._launch_initial_tasks(
                tg, user_query, history, use_history
            )

            # Phase 2: Fast path - generate initial answer
            fast_answer, initial_results = await self._generate_fast_answer_path(
                tg, tasks["retrieval"], user_query
            )
            yield fast_answer

            # Phase 3: Process clarification and decomposition results
            final_query, queries, need_new_retrieval = (
                await self._determine_final_query(
                    tasks["clarify"], tasks["decompose"], user_query
                )
            )

            # Phase 4: Get final retrieval results
            final_results, need_new_eval = await self._perform_final_retrieval(
                tg, need_new_retrieval, queries, initial_results, user_query
            )

            # Phase 5: Evaluate retrieval and extract context
            filtered_results, ctx_result = await self._evaluate_and_contextualize(
                tg,
                tasks,
                need_new_eval,
                tasks.get("optimistic_eval"),
                user_query,
                final_query,
                final_results,
            )

            # Phase 6: Generate refined answer
            refined_answer = await self._generate_refined_answer(
                final_query, filtered_results, ctx_result
            )

            # Phase 7: Update history and return final answer
            if refined_answer.answer != fast_answer.answer:
                logger.info("YIELDING REFINED ANSWER")
                yield refined_answer
            else:
                logger.info("Fast answer was optimal, not yielding refined answer")

            # Save to history
            self.conversation_history.add_entry(
                user_id,
                user_query,
                refined_answer.answer,
            )
            logger.info("Conversation history updated")

            # Clear the current user_id
            self._current_user_id = None

            # Log total processing time
            total_time = time.time() - overall_start
            logger.info(f"Query processing completed in {total_time:.2f}s")

    async def _launch_initial_tasks(self, tg, user_query, history, use_history):
        """Launches all initial parallel tasks and returns their task objects."""
        logger.info(
            "PHASE 1: Starting parallel tasks (context, retrieval, clarify, decompose)"
        )
        tasks = {}

        # Start context task if using history
        if use_history and history:
            tasks["context"] = tg.create_task(
                self.context_processor.extract_relevant_context(user_query, history)
            )
            logger.debug("Started context extraction task")

        # Start retrieval, clarify, decompose tasks
        logger.debug("Starting initial retrieval task")
        tasks["retrieval"] = tg.create_task(self.router.route_query_async(user_query))

        logger.debug("Starting clarification task")
        tasks["clarify"] = tg.create_task(
            self.preprocessor.clarify_query_async(user_query)
        )

        logger.debug("Starting decomposition task")
        tasks["decompose"] = tg.create_task(
            self.preprocessor.decompose_query_async(user_query)
        )

        return tasks

    async def _generate_fast_answer_path(self, tg, retrieval_task, user_query):
        """Handles the fast answer path with initial retrieval and optimistic evaluation."""
        logger.info("PHASE 2: Fast path - waiting for initial retrieval")
        initial_results = await retrieval_task
        logger.info(f"Initial retrieval completed")

        # Start optimistic evaluation
        logger.debug("Starting optimistic evaluation task")
        optimistic_eval_task = tg.create_task(
            self.evaluator.evaluate_retrieval(
                user_query, user_query, initial_results, self.router
            )
        )

        # Generate fast answer
        logger.info("Generating fast answer")
        fast_answer_start = time.time()
        fast_answer = await self.generator.generate_answer_async(
            user_query, initial_results, ""
        )
        fast_answer_time = time.time() - fast_answer_start
        logger.info(f"Fast answer generated in {fast_answer_time:.2f}s")

        return fast_answer, initial_results

    async def _determine_final_query(self, clarify_task, decompose_task, user_query):
        """Processes clarification and decomposition to determine the final query."""
        logger.info("PHASE 3: Waiting for clarification & decomposition")
        clarified = await clarify_task
        logger.info(f"Clarification completed")

        final_query = clarified.clarified_query
        if final_query != user_query:
            logger.info(f"Query was clarified: '{user_query}' → '{final_query}'")
            logger.info("Re-decomposing based on clarified query")
            decomposed = await self.preprocessor.decompose_query_async(final_query)
            logger.info(f"Re-decomposition completed")
        else:
            logger.info("Query was not clarified, using original decomposition")
            decomposed = await decompose_task
            logger.info(f"Decomposition completed")

        # Handle queries for retrieval
        queries = (
            decomposed.decomposed_query
            if decomposed.query_complexity == "complex"
            else [final_query]
        )
        if queries is None:
            queries = [final_query]
        queries = list(dict.fromkeys(queries))  # dedupe

        # Determine if we need new retrieval
        need_new_retrieval = queries != [user_query]
        logger.info(f"Need new retrieval: {need_new_retrieval}")

        return final_query, queries, need_new_retrieval

    async def _perform_final_retrieval(
        self, tg, need_new_retrieval, queries, initial_results, user_query
    ):
        """Performs final retrieval based on clarified and decomposed queries if needed."""
        if need_new_retrieval:
            logger.info(f"PHASE 4A: Running {len(queries)} new retrieval queries")
            for i, q in enumerate(queries):
                logger.debug(f"  Retrieval query {i+1}: '{q}'")

            new_lists = await asyncio.gather(
                *[self.router.route_query_async(q) for q in queries]
            )
            final_results = list(chain.from_iterable(new_lists))
            logger.info(f"New retrieval completed")
            need_new_eval = True
        else:
            logger.info("PHASE 4B: No new retrieval needed, using initial results")
            final_results = initial_results
            need_new_eval = False

        return final_results, need_new_eval

    async def _evaluate_and_contextualize(
        self,
        tg,
        tasks,
        need_new_eval,
        optimistic_eval_task,
        user_query,
        final_query,
        final_results,
    ):
        """Handles final evaluation and context extraction."""
        logger.info("PHASE 5: Setting up final evaluation")
        tasks_to_await = []

        # Decide which evaluation task to use
        if need_new_eval:
            logger.info("Need new evaluation due to query/results changes")
            if optimistic_eval_task and not optimistic_eval_task.done():
                logger.info("Cancelling optimistic evaluation task")
                optimistic_eval_task.cancel()

            logger.info("Creating new evaluation task")
            eval_coro = self.evaluator.evaluate_retrieval(
                user_query, final_query, final_results, self.router
            )
            final_eval_task = tg.create_task(eval_coro)
        else:
            logger.info("Reusing optimistic evaluation task")
            final_eval_task = optimistic_eval_task

        # Build list of tasks to await
        if final_eval_task:
            tasks_to_await.append(final_eval_task)
            logger.debug("Added evaluation task to await list")
        if "context" in tasks:
            tasks_to_await.append(tasks["context"])
            logger.debug("Added context task to await list")

        # Await final tasks
        if tasks_to_await:
            logger.info(f"PHASE 6: Awaiting {len(tasks_to_await)} final tasks")
            await asyncio.gather(*tasks_to_await, return_exceptions=True)
            logger.info(f"All final tasks completed")
        else:
            logger.info("PHASE 6: No final tasks to await")

        # Extract results safely
        evaluated_results = final_results  # Default fallback
        if (
            final_eval_task
            and final_eval_task.done()
            and not final_eval_task.cancelled()
        ):
            try:
                evaluated_results = final_eval_task.result()
                logger.info("Successfully obtained evaluation results")
            except Exception as e:
                logger.error(f"Evaluation task failed: {e}")
                logger.info("Falling back to unevaluated results")

        ctx_result_obj = RelevantHistoryContext(
            required_context=False, explanation="", relevant_snippets=""
        )
        if (
            "context" in tasks
            and tasks["context"].done()
            and not tasks["context"].cancelled()
        ):
            try:
                ctx_result_obj = tasks["context"].result()
                has_context = bool(ctx_result_obj.relevant_snippets.strip())
                logger.info(
                    f"Context extraction succeeded. Has relevant context: {has_context}"
                )
            except Exception as e:
                logger.error(f"Context task failed: {e}")
                logger.info("Proceeding without historical context")

        return evaluated_results, ctx_result_obj

    async def _generate_refined_answer(
        self, final_query, filtered_results, ctx_result_obj
    ):
        """Generates the refined answer using evaluated results and context."""
        logger.info("Generating refined answer")
        refined_answer = await self.generator.generate_answer_async(
            final_query, filtered_results, ctx_result_obj.relevant_snippets
        )
        logger.info(f"Refined answer generated")

        # Get conversation history for follow-up question generation
        user_id = getattr(self, "_current_user_id", None)
        history = []
        if user_id:
            history = self.conversation_history.get_history(user_id)

        # Generate follow-up questions
        logger.info("Generating follow-up questions")
        follow_up_questions = (
            await self.follow_up_generator.generate_follow_up_questions(
                final_query, refined_answer.answer, history
            )
        )
        logger.info(f"Generated follow-up questions: {follow_up_questions.questions}")

        # Add follow-up questions to the refined answer
        # refined_answer.follow_up_questions = follow_up_questions.questions

        return refined_answer

    def process_query_all_answers(
        self, user_id: str, user_query: str, use_history: bool = True
    ) -> List[CitedAnswerResult]:
        """Get a list of all answers (fast and refined) from the query"""
        logger.info(f"Collecting all answers for query: '{user_query}'")

        async def _run():
            results = []
            async for ans in self.process_query_optimistic_async(
                user_id, user_query, use_history
            ):
                is_fast = len(results) == 0
                logger.info(f"Received {'fast' if is_fast else 'refined'} answer")
                results.append(ans)
            return results

        start_time = time.time()
        all_results = asyncio.run(_run())
        elapsed = time.time() - start_time

        logger.info(f"Retrieved {len(all_results)} answers in {elapsed:.2f}s")
        return all_results


# Application setup and usage
def setup_medical_rag() -> MedicalRAG:
    """Set up and return a configured MedicalRAG instance."""
    # Initialize vector store manager
    store_manager = VectorStoreManager()

    # Define paths
    lipitor_save_path = "data/faiss_lipitor_index"
    metformin_save_path = "data/faiss_metformin_index"
    lipitor_data_path = "data/chunks_lipitor.json"
    metformin_data_path = "data/chunks_metformin.json"

    # Load or create vector stores
    lipitor_store = store_manager.load_or_create_store(
        lipitor_save_path, lipitor_data_path
    )
    metformin_store = store_manager.load_or_create_store(
        metformin_save_path, metformin_data_path
    )

    # Create vector store dictionary
    vector_stores = {"Lipitor": lipitor_store, "Metformin": metformin_store}

    # Create and return MedicalRAG instance with a single model
    return MedicalRAG(vector_stores, llm_model="gpt-4o-mini")


# Example usage
if __name__ == "__main__":
    # Configure more verbose logging for the example run
    logging.getLogger().setLevel(logging.INFO)
    logger.info("Starting Medical RAG example")

    # Set up the medical RAG system
    medical_rag = setup_medical_rag()
    logger.info("Medical RAG system initialized")

    # Simulate a conversation with a user
    user_id = "user123"

    # First question
    logger.info("\n--- New conversation with user123 ---")
    question1 = "What are the side effects of Lipitor?"
    logger.info(f"Processing question 1: '{question1}'")

    start_time = time.time()
    results1 = medical_rag.process_query_all_answers(user_id, question1)
    elapsed1 = time.time() - start_time

    logger.info(f"\nUser: {question1}")
    logger.info(f"Fast Answer ({elapsed1:.2f}s): {results1[0].answer[:150]}...")

    if len(results1) > 1:
        logger.info(f"Refined Answer ({elapsed1:.2f}s): {results1[1].answer[:150]}...")
        logger.info(f"Answer improved: {results1[1].answer != results1[0].answer}")
    else:
        logger.info("No refined answer needed (fast answer was sufficient)")
