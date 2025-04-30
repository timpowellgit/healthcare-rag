import json
import logging
import asyncio
import openai
from typing import Dict, List, Optional, Union

from ..models.retrieval import QueryDocument, QueryResult, QueryResultList
from ..models.queries import RetrievalEvaluation
from .base import BaseProcessor, log_timing
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam

# For type hints
from weaviate.client import WeaviateAsyncClient
from weaviate.classes.query import MetadataQuery, HybridFusion

logger = logging.getLogger("MedicalRAG")

class QueryRouter:
    """Routes queries to appropriate Weaviate collections based on content."""

    def __init__(
        self,
        weaviate_client: WeaviateAsyncClient,
        collection_names: List[str],
        llm_model: str = "gpt-4o-mini",
        async_client: Optional[openai.AsyncOpenAI] = None,
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
    async def route_query_async(self, user_query: str) -> QueryResultList:
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
                return QueryResultList(results=[])

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
        return QueryResultList(results=results)

    async def _get_tool_calls(self, user_query: str):
        """Get tool calls from LLM for the query."""
        if not self.async_client:
            logger.error("OpenAI async client not initialized")
            return []
            
        messages: List[ChatCompletionMessageParam] = [
            {"role": "user", "content": user_query} # type: ChatCompletionUserMessageParam
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


class RetrievalEvaluator(BaseProcessor):
    """Evaluates if retrieved documents contain sufficient information to answer the query."""

    def _prepare_context(
        self, retrieval_results: List[QueryResult]
    ) -> tuple[str, List[str]]:
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
    ) -> QueryResultList:
        """Fetch additional results for the specified queries."""
        if not additional_queries:
            return QueryResultList(results=[])

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
            for single_query_result_list in results_list:
                if single_query_result_list and single_query_result_list.results:
                    additional_results.extend(single_query_result_list.results)

            return QueryResultList(results=additional_results)
        except Exception as e:
            logger.error(f"Error fetching additional results: {e}")
            return QueryResultList(results=[])

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
        if not retrieval_results or not retrieval_results.results:
            return retrieval_results

        # Prepare context from retrieval results
        context, sources = self._prepare_context(retrieval_results.results)

        # Use the base class _call_llm method 
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
        enhanced_results = retrieval_results  # Start with original results

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

            if additional_results and additional_results.results:
                enhanced_results.results.extend(additional_results.results)
                logger.info(
                    f"Added {len(additional_results.results)} additional retrieval results"
                )
        else:
            logger.info("Retrieved information is sufficient or evaluation failed")

        final_doc_count = sum(len(r.docs) for r in enhanced_results.results)
        logger.info(f"Evaluation completed, final document count: {final_doc_count}")

        return enhanced_results 