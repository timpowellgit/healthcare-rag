from langchain.embeddings import OpenAIEmbeddings
from uuid import uuid4
import json
from typing import Dict, List, Union, Any, Optional, TypedDict, Tuple, NamedTuple, Literal
from datetime import datetime
from langchain_core.documents import Document
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import os
import openai
from pydantic import BaseModel, model_validator, Field
from typing import Self
import asyncio

# Type definitions

class ClarifiedQuery(BaseModel):
    original_query: str
    ambiguity_level: Literal["clear and specific", "medium ambiguity", "high ambiguity"] = Field(
        description="The level of ambiguity in the original query.")
    clarified_query: str = Field(
        description="The clarified query that resolves the ambiguity. If there is nothing to clarify, just return the original query.")

    @model_validator(mode="after")
    def check_clarified_query(self)-> Self:
        if self.ambiguity_level == "clear and specific" and self.clarified_query != self.original_query:
            self.clarified_query = self.original_query
        # TODO:: retrieve hitory here to verify the citation is correct 

        return self
    
class DecomposedQuery(BaseModel):
    original_query: str
    query_complexity: Literal["simple", "complex"] = Field(description="The complexity of the original query.")
    decomposed_query: Optional[List[str]] = Field(description="The original query decomposed into subqueries." 
                                                  "Only meant for unpacking complex queries. If there is nothing to decompose," 
                                                  "just return the original query.")

    @model_validator(mode="after")
    def check_decomposed_query(self)-> Self:
        if self.query_complexity == "simple" and self.decomposed_query != self.original_query:
            self.decomposed_query = [self.original_query]
        return self
    
    
class RetrievalEvaluation(BaseModel):
    is_sufficient: bool = Field(description="Whether the retrieved information is sufficient to answer the query")
    missing_information: Optional[str] = Field(None, description="Description of information that's missing from the retrieved documents")
    additional_queries: Optional[List[str]] = Field(None, description="List of suggested follow-up queries to retrieve missing information")


class QueryDocument(BaseModel):
    """Represents a single document retrieved from a vector store."""
    content: str
    relevance_score: float = Field(description="The relevance score provided by the similarity search.")
    doc_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class QueryResult(BaseModel):
    source: str
    query: str
    docs: List[QueryDocument]

class ErrorResult(BaseModel):
    error: str

class AnswerResult(BaseModel):
    answer: str
    sources: List[str]
    original_query: str
    clarified_query: Optional[str]

class ConversationEntry(BaseModel):
    """Represents a single entry in a conversation history."""
    timestamp: datetime
    user_query: str
    answer: str
    sources: List[str]

class DocumentRelevanceEvaluation(BaseModel):
    explanation: str = Field(description="Explanation of why the document is relevant or irrelevant to the query")
    is_relevant: bool = Field(description="Whether the document contains any information relevant whatsoever to answering the query")


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
    def check_relevant_snippets(self)-> Self:
        if self.required_context == False:
            self.relevant_snippets = ""
        return self
    

    
class VectorStoreManager:
    """Manages loading and creation of vector stores."""
    
    def __init__(self, embedding_model: str = "text-embedding-3-large"):
        """
        Initialize the vector store manager.
        
        Args:
            embedding_model: Model name for embeddings
        """
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
    def load_or_create_store(self, save_path: str, data_path: Optional[str] = None) -> FAISS:
        """
        Load an existing vector store or create a new one if it doesn't exist.
        
        Args:
            save_path: Path to save/load the FAISS index
            data_path: Path to the source data JSON for creation (optional)
            
        Returns:
            A FAISS vector store
        """
        if os.path.exists(save_path):
            print(f"Loading existing index from {save_path}")
            return FAISS.load_local(save_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            if not data_path:
                raise ValueError("Data path must be provided to create a new vector store")
                
            print(f"Creating new index at {save_path}")
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
            docs = [Document(page_content=chunk["contextualized"], metadata=chunk["metadata"]) 
                   for chunk in chunks]
            uuids = [str(uuid4()) for _ in range(len(docs))]
            
            # Add documents and save
            vector_store.add_documents(documents=docs, ids=uuids)
            vector_store.save_local(save_path)
            print(f"Index saved to {save_path}")
            
            return vector_store

class QueryPreprocessor:
    """
    Preprocesses user queries to resolve ambiguities using conversation history.
    Detects and clarifies ambiguous references like "it", "they", "this side effect", etc.
    """
    
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        """
        Initialize the query preprocessor.
        
        Args:
            llm_model: Model name for the clarification LLM
        """
        self.llm_model = llm_model
        self.client = openai.OpenAI()
        
    def clarify_query(self, user_query: str, conversation_context: str = "") -> ClarifiedQuery:
        """
        Clarify an ambiguous query by expanding references based on conversation history.
        
        Args:
            user_query: The raw user query which may contain ambiguous references
            conversation_context: Recent conversation history for context
            
        Returns:
            Dictionary with original and clarified queries
        """
        # If there's no conversation context, no need to clarify
        if not conversation_context:
            return ClarifiedQuery(original_query=user_query, clarified_query=user_query)
            
        prompt = [
            {
                "role": "system", 
                "content": """You are an AI assistant that clarifies ambiguous medical queries.
                Your task is to rewrite potentially ambiguous queries to be more explicit and specific,
                especially when they contain pronouns or implicit references to previous conversation.
                
                For example:
                - "What are its side effects?" → "What are Lipitor's side effects?"
                - "Is it safe during pregnancy?" → "Is metformin safe during pregnancy?"
                - "Are any of these serious?" → "Are any of the side effects of Lipitor serious?"
                
                Only rewrite if needed - if the query is already clear and specific, leave it unchanged.
                Return the original query, the clarified query, and the level of ambiguity."""
            },
            {
                "role": "user", 
                "content": f"Previous conversation:\n{conversation_context}\n\nUser query: {user_query}\n\nClarified query:"
            }
        ]
        
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.llm_model,
                messages=prompt,
                temperature=0.1,
                response_format=ClarifiedQuery
            )
            
            clarified_query = response.choices[0].message.parsed
            print(f"Clarified query: {clarified_query}")
            return clarified_query
            
        except Exception as e:
            print(f"Error during query clarification: {e}")
            # If clarification fails, fall back to the original query
            return ClarifiedQuery(original_query=user_query, ambiguity_level="medium ambiguity", explanation="", clarified_query=user_query, clarification="", citation="")
        
    def decompose_query(self, user_query: str) -> DecomposedQuery:
        """
        Decompose a complex query into simpler subqueries.
        
        Args:
            user_query: The raw user query which may contain complex references
            
        Returns:
            DecomposedQuery object containing original query, query complexity, and decomposed queries
        """
        prompt = [
            {
                "role": "system", 
                "content": """You are an AI assistant that decomposes complex medical queries into simpler subqueries.
                
                For example:
                - "Compare the side effects of Lipitor and Crestor and tell me which one is safer for elderly patients with kidney problems?" → ["What are the side effects of Lipitor?", "What are the side effects of Crestor?", "How do Lipitor and Crestor affect elderly patients?", "How do Lipitor and Crestor affect patients with kidney problems?"]
                - "What's the recommended dosage of metformin for type 2 diabetes, and how should it be adjusted for patients with liver disease?" → ["What is the recommended dosage of metformin for type 2 diabetes?", "How should metformin dosage be adjusted for patients with liver disease?"]
                - "Can you explain how ACE inhibitors work to lower blood pressure and what their long-term effects are on kidney function and cardiovascular health?" → ["How do ACE inhibitors work to lower blood pressure?", "What are the long-term effects of ACE inhibitors on kidney function?", "What are the long-term effects of ACE inhibitors on cardiovascular health?"]
    
                Only decompose if needed - if the query is already simple, leave it unchanged.
                Return the original query, the complexity, and the decomposed queries."""
            },
            {
                "role": "user", 
                "content": f"User query: {user_query}\n\nDecomposed query:" 
            }
        ]
        
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.llm_model,
                messages=prompt,
                temperature=0.1,
                response_format=DecomposedQuery
            )
            
            decomposed_query = response.choices[0].message.parsed
            print(f"Decomposed query: {decomposed_query}")
            return decomposed_query     
        
        except Exception as e:
            print(f"Error during query decomposition: {e}")
            return DecomposedQuery(original_query=user_query, query_complexity="simple", decomposed_query=user_query)
            
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
        
    def add_entry(self, user_id: str, query: str, answer: str, sources: List[str]) -> None:
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
            sources=sources
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
                sources=entry.sources
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
        entries = [entry.model_dump(mode='json') for entry in self.conversations[user_id]]
        
        with open(file_path, 'w') as f:
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
            with open(file_path, 'r') as f:
                entries_data = json.load(f)
                
            # Use model_validate for deserialization
            self.conversations[user_id] = [
                ConversationEntry.model_validate(entry_data)
                for entry_data in entries_data
            ]
        except Exception as e:
            print(f"Error loading conversation for {user_id}: {e}")
            self.conversations[user_id] = []

class QueryRouter:
    """Routes queries to appropriate vector stores based on content."""
    
    def __init__(self, stores: Dict[str, FAISS], llm_model: str = "gpt-4o-mini"):
        """
        Initialize the query router.
        
        Args:
            stores: Dictionary mapping store names to FAISS vector stores
            llm_model: Model name for the routing LLM
        """
        self.stores = stores
        self.llm_model = llm_model
        self.client = openai.OpenAI()
        self.async_client = openai.AsyncOpenAI()
        
        # Build tools dynamically based on available stores
        self.tools = []
        for store_name in stores.keys():
            self.tools.append({
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
            })
    

    async def route_query_async(self, user_query: str) -> List[QueryResult]:
        """
        Async version of route_query to route a user query to the appropriate vector store(s).
        
        Args:
            user_query: The user's question
            
        Returns:
            List of query results including document content and relevance scores
        """
        messages = [{"role": "user", "content": user_query}]
        results = []
        
        try:
            response = await self.async_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
            )
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    query = function_args.get("query")
                    
                    # Extract store name from function name (query_storename)
                    store_name = function_name.split("_", 1)[1].capitalize()
                    
                    if store_name.lower() in [name.lower() for name in self.stores.keys()]:
                        print(f"Routing to {store_name} vector store for query: {query}")
                        store = self.stores[store_name] if store_name in self.stores else self.stores[store_name.lower()]
                        
                        # Use similarity_search_with_score instead of similarity_search
                        search_results = store.similarity_search_with_score(query, k=3)
                        query_docs = [
                            QueryDocument(
                                content=doc.page_content, 
                                relevance_score=score,
                                doc_id=doc.id,
                                metadata=doc.metadata
                            )
                            for doc, score in search_results
                        ]
                        results.append(QueryResult(source=store_name, query=query, docs=query_docs))
            else:
                print("No relevant function identified for the query.")
        except Exception as e:
            print(f"An error occurred in routing: {e}")
                
        return results

class AnswerGenerator:
    """Generates answers from retrieved documents."""
    
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        """
        Initialize the answer generator.
        
        Args:
            llm_model: Model name for the answer generation LLM
        """
        self.llm_model = llm_model
        self.client = openai.OpenAI()
    
    def generate_answer(
        self, 
        user_question: str, 
        retrieval_results: List[QueryResult],
        conversation_context: str = ""
    ) -> AnswerResult:
        """
        Generate a comprehensive answer based on retrieved documents.
        
        Args:
            user_question: The original user question
            retrieval_results: List of query results from different sources
            conversation_context: Optional context from previous conversation
            
        Returns:
            Answer result with answer and sources
        """
        # Check if we have any retrieval results
        if not retrieval_results:
            return AnswerResult(
                answer="I'm sorry, I couldn't find relevant information to answer your question.",
                sources=[],
                original_query=user_question,
                clarified_query=None
            )
                
        # Combine context from all sources
        combined_context = ""
        sources = []
        
        for result in retrieval_results:
            combined_context += "\n\n" + "\n\n".join([doc.content for doc in result.docs])
            sources.append(result.source)
        
        # Create system prompt with conversation history if available
        system_content = f"You are a medical assistant. Use the following context about {', '.join(sources)} to answer the question."
        if conversation_context:
            system_content += f"This information from the conversation history is relevant to the question:\n\n{conversation_context}"
            
        rag_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Context:\n{combined_context.strip()}\n\nQuestion: {user_question}"}
        ]
        
        try:
            rag_response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=rag_messages
            )
            answer = rag_response.choices[0].message.content
            # TODO: Use pydantic, probably use structured output, will have to make changes like citation, requerying, etc.
            return AnswerResult(
                answer=answer,
                sources=sources,
                original_query=user_question,
                clarified_query=None
            )
        except Exception as e:
            return AnswerResult(
                answer=f"I encountered an error while generating your answer: {str(e)}",
                sources=[],
                original_query=user_question,
                clarified_query=None
            )

class RetrievalEvaluator:
    """Evaluates if retrieved documents contain sufficient information to answer the query."""
    
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        """
        Initialize the retrieval evaluator.
        
        Args:
            llm_model: Model name for the evaluation LLM
        """
        self.llm_model = llm_model
        self.client = openai.OpenAI()
        self.async_client = openai.AsyncOpenAI()
    
    async def evaluate_retrieval(
        self, 
        original_query: str, 
        clarified_query: str,
        retrieval_results: List[QueryResult],
        router: QueryRouter
    ) -> List[QueryResult]:
        """
        Evaluate if retrieval results are sufficient and generate additional queries if needed.
        
        Args:
            original_query: The original user query
            clarified_query: The clarified query after preprocessing
            retrieval_results: List of initial query results from different sources
            router: Query router to fetch additional results
            
        Returns:
            Enhanced list of query results with additional information if needed
        """
        # Skip evaluation if no results
        if not retrieval_results:
            return retrieval_results
            
        # Combine context from all sources
        combined_context = ""
        sources = []
        
        for result in retrieval_results:
            combined_context += "\n\n" + "\n\n".join([doc.content for doc in result.docs])
            sources.append(result.source)
        
        # Create evaluation prompt
        eval_prompt = [
            {
                "role": "system", 
                "content": """You are an AI assistant that evaluates if retrieved medical information is sufficient to answer a user's query.
                Analyze the retrieved information and the query to determine:
                1. If the information fully addresses the query
                2. If there are information gaps that need to be filled
                3. What additional queries would retrieve the missing information
                
                For example, if a user asks "How does Lipitor compare to Crestor for elderly patients?" and the retrieved information 
                only discusses Lipitor's effects on elderly patients, you should indicate:
                - The information is not sufficient
                - Information about Crestor's effects on elderly patients is missing
                - Suggest an additional query like "How does Crestor affect elderly patients?"
                
                Be thorough in your analysis and ensure all aspects of the query can be answered.
                """
            },
            {
                "role": "user", 
                "content": f"""
                Original Query: {original_query}
                Clarified Query: {clarified_query}
                
                Retrieved Information:
                {combined_context}
                
                Sources: {', '.join(sources)}
                
                Evaluate if this information is sufficient to fully answer the query.
                If not, specify what information is missing and suggest additional queries to retrieve it.
                """
            }
        ]
        
        try:
            response = await self.async_client.beta.chat.completions.parse(
                model=self.llm_model,
                messages=eval_prompt,
                temperature=0.1,
                response_format=RetrievalEvaluation
            )
            
            evaluation = response.choices[0].message.parsed
            
            # Check if additional queries are needed
            if not evaluation.is_sufficient and evaluation.additional_queries:
                print("Retrieved information is insufficient. Generating additional queries...")
                if evaluation.missing_information:
                    print(f"Missing information: {evaluation.missing_information}")
                
                # Get additional retrieval results asynchronously
                if evaluation.additional_queries:
                    print("Executing follow-up queries asynchronously...")
                    routing_tasks = [router.route_query_async(query) for query in evaluation.additional_queries]
                    results_list = await asyncio.gather(*routing_tasks)
                    
                    # Flatten the results and extend retrieval_results
                    additional_results = []
                    for results in results_list:
                        additional_results.extend(results)
                    
                    retrieval_results.extend(additional_results)
                    print(f"Added {len(additional_results)} additional retrieval results")
            else:
                print("Retrieved information is sufficient to answer the query.")
                
            return retrieval_results
            
        except Exception as e:
            print(f"Error during retrieval evaluation: {e}")
            # If evaluation fails, return original results
            return retrieval_results

    async def filter_irrelevant_documents(
        self,
        query: str,
        retrieval_results: List[QueryResult],
        relevance_threshold: float = 0.8
    ) -> List[QueryResult]:
        """
        Filter out documents that are irrelevant to the query by evaluating each document asynchronously.
        
        Args:
            query: The user query
            retrieval_results: List of query results from different sources
            relevance_threshold: Score threshold above which documents are automatically kept
            
        Returns:
            Filtered list of query results with irrelevant documents removed
        """
        if not retrieval_results:
            return retrieval_results
        
        filtered_results = []
        
        async def evaluate_document(doc: QueryDocument) -> Tuple[QueryDocument, bool]:
            """Evaluate a single document for relevance to the query."""
            eval_prompt = [
                {
                    "role": "system", 
                    "content": """You are an AI assistant that evaluates if a document is relevant to a query.
                    A document is relevant if it contains information that could help answer the query.
                    A document is irrelevant if it has no information related to the query or addresses a completely different topic.
                    Provide a brief explanation and determine if the document is relevant."""
                },
                {
                    "role": "user", 
                    "content": f"Query: {query}\n\nDocument:\n{doc.content}\n\nEvaluate whether this document is relevant to the query."
                }
            ]
            
            try:
                response = await self.async_client.beta.chat.completions.parse(
                    model=self.llm_model,
                    messages=eval_prompt,
                    temperature=0.1,
                    response_format=DocumentRelevanceEvaluation
                )
                
                evaluation = response.choices[0].message.parsed
                return (doc, evaluation.is_relevant)
            except Exception as e:
                print(f"Error evaluating document: {e}")
                return (doc, True)  # Keep document if evaluation fails
        
        for result in retrieval_results:
            # Automatically keep documents with high relevance scores
            high_score_docs = [doc for doc in result.docs if doc.relevance_score > relevance_threshold]
            low_score_docs = [doc for doc in result.docs if doc.relevance_score <= relevance_threshold]
            
            # Create evaluation tasks for low-score documents
            if low_score_docs:
                evaluation_tasks = [evaluate_document(doc) for doc in low_score_docs]
                evaluation_results = await asyncio.gather(*evaluation_tasks)
                
                # Filter based on evaluation results
                relevant_low_score_docs = [doc for doc, is_relevant in evaluation_results if is_relevant]
                all_relevant_docs = high_score_docs + relevant_low_score_docs
            else:
                all_relevant_docs = high_score_docs
            
            # Only add the result if it has relevant documents
            if all_relevant_docs:
                filtered_results.append(QueryResult(
                    source=result.source,
                    query=result.query,
                    docs=all_relevant_docs
                ))
            
        return filtered_results



# Add this new class for handling conversation context extraction
class ConversationContextProcessor:
    """Processes the conversation history to extract only relevant context for the current query."""
    
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        """
        Initialize the conversation context processor.
        
        Args:
            llm_model: Model name for context processing
        """
        self.llm_model = llm_model
        self.client = openai.OpenAI()
        self.async_client = openai.AsyncOpenAI()
    
    async def extract_relevant_context(
        self,
        query: str,
        conversation_history: List[ConversationEntry]
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
                relevant_snippets=""
            )
        
        # Format conversation history for the prompt
        history_text = "Previous conversation:\n"
        for i, entry in enumerate(conversation_history):
            history_text += f"[{i+1}] User: {entry.user_query}\n"
            history_text += f"    Assistant: {entry.answer}\n\n"
        
        # Create the prompt
        context_prompt = [
            {
                "role": "system", 
                "content": """You are an AI assistant that analyzes conversation history to extract only the relevant and required parts needed to answer a current query.
                
                Your task is to identify information from previous exchanges that provides necessary context or background 
                for answering the current query effectively. This includes:
                
                - Prior information the user shared that relates to their current question
                - Previous answers that contain information needed for the current response
                - Any relevant medical topics, medications, or conditions discussed previously
                - User preferences or circumstances mentioned earlier that affect the current answer
                
                Provide:
                1. A boolean value indicating if any parts of the conversation history are relevant and necessary to answer the current query
                2. An explanation of why these entries are relevant and necessary
                3. A formatted context string ready to be used in the answer generation prompt, representing the relevant and necessary snippetsof the conversation history
                
                If no parts of the conversation history are relevant to the current query, return False and an empty formatted context.
                """
            },
            {
                "role": "user", 
                "content": f"Current query: {query}\n\n{history_text}\n\nExtract the relevant context for answering this query:"
            }
        ]
        
        try:
            response = await self.async_client.beta.chat.completions.parse(
                model=self.llm_model,
                messages=context_prompt,
                temperature=0.1,
                response_format=RelevantHistoryContext
            )
            
            return response.choices[0].message.parsed
        
        except Exception as e:
            print(f"Error during context extraction: {e}")
            # Return empty context if extraction fails
            return RelevantHistoryContext(
                required_context=False,
                explanation=f"Error during context extraction: {str(e)}",
                relevant_snippets=""
            )

class MedicalRAG:
    """
    A medical RAG system that integrates vector stores, query routing, answer generation,
    and conversation history.
    """
    
    def __init__(
        self,
        vector_stores: Dict[str, FAISS],
        router_model: str = "gpt-4o-mini",
        generator_model: str = "gpt-4o-mini",
        preprocessor_model: str = "gpt-4o-mini",
        evaluator_model: str = "gpt-4o-mini",
        context_processor_model: str = "gpt-4o-mini",
        conversation_history_dir: str = "data/conversations"
    ):
        """
        Initialize the Medical RAG system.
        
        Args:
            vector_stores: Dictionary mapping store names to FAISS vector stores
            router_model: Model for query routing
            generator_model: Model for answer generation
            preprocessor_model: Model for query preprocessing/clarification
            evaluator_model: Model for retrieval evaluation
            context_processor_model: Model for conversation context processing
            conversation_history_dir: Directory to store conversation histories
        """
        self.router = QueryRouter(vector_stores, router_model)
        self.generator = AnswerGenerator(generator_model)
        self.preprocessor = QueryPreprocessor(preprocessor_model)
        self.evaluator = RetrievalEvaluator(evaluator_model)
        self.context_processor = ConversationContextProcessor(context_processor_model)
        self.conversation_history = ConversationHistory(conversation_history_dir)
    
    
    
    def get_conversation_history(self, user_id: str, limit: int = 5) -> List[ConversationEntry]:
        """
        Get recent conversation history for a user.
        
        Args:
            user_id: Unique identifier for the user
            limit: Maximum number of entries to return
            
        Returns:
            List of conversation entries, most recent first
        """
        return self.conversation_history.get_history(user_id, limit)

    async def process_query_async(self, user_id: str, user_query: str, use_history: bool = True) -> AnswerResult:
        """
        Process a user query from preprocessing to answer generation using async calls.
        
        Args:
            user_id: Unique identifier for the user
            user_query: The user's question
            use_history: Whether to incorporate conversation history
            
        Returns:
            Final result with answer and sources
        """
        # Get full conversation history if needed
        history = []
        if use_history:
            history = self.conversation_history.get_history(user_id)
        
        # Get conversation context from history for preprocessing
        conversation_context = ""
        if use_history and history:
            conversation_context = self.conversation_history.get_context_from_history(user_id)
        
        # Preprocess and clarify the query if needed
        query_info = self.preprocessor.clarify_query(user_query, conversation_context)
        original_query = query_info.original_query
        clarified_query = query_info.clarified_query
        
        # Log the clarification if it happened
        if clarified_query != original_query:
            print(f"Clarified query: '{original_query}' → '{clarified_query}'")
        
        # Decompose the query if needed
        query_decomposition = self.preprocessor.decompose_query(clarified_query)
        decomposed_queries = query_decomposition.decomposed_query
        
        # Route the decomposed queries to appropriate vector stores asynchronously
        retrieval_results = []
        routing_tasks = [self.router.route_query_async(query) for query in decomposed_queries]
        results_list = await asyncio.gather(*routing_tasks)
        
        # Flatten the results
        for results in results_list:
            retrieval_results.extend(results)
        
        # Evaluate retrieval results and get additional information if needed
        enhanced_results = await self.evaluator.evaluate_retrieval(
            original_query,
            clarified_query,
            retrieval_results,
            self.router
        )
        
        # Filter out irrelevant documents
        filtered_results = await self.evaluator.filter_irrelevant_documents(
            clarified_query,
            enhanced_results,
            0.8
        )
        
        # Extract relevant conversation context if history is being used
        relevant_context = ""
        if use_history and history:
            context_result = await self.context_processor.extract_relevant_context(
                clarified_query,
                history
            )
            relevant_context = context_result.relevant_snippets
            print(f"Extracted relevant context: {context_result.explanation}")
        
        # Generate answer with the filtered results and relevant context
        result = self.generator.generate_answer(
            clarified_query, 
            filtered_results,
            relevant_context  # Use extracted relevant context instead of full history
        )
        
        # Add clarification info to result if needed
        if clarified_query != original_query:
            result.clarified_query = clarified_query
        
        # Save to conversation history
        self.conversation_history.add_entry(
            user_id,
            original_query,
            result.answer,
            result.sources
        )
        
        return result
    
    # Add a synchronous wrapper for compatibility
    def process_query(self, user_id: str, user_query: str, use_history: bool = True) -> AnswerResult:
        """Synchronous wrapper for process_query_async"""
        return asyncio.run(self.process_query_async(user_id, user_query, use_history))

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
    lipitor_store = store_manager.load_or_create_store(lipitor_save_path, lipitor_data_path)
    metformin_store = store_manager.load_or_create_store(metformin_save_path, metformin_data_path)
    
    # Create vector store dictionary
    vector_stores = {
        "Lipitor": lipitor_store,
        "Metformin": metformin_store
    }
    
    # Create and return MedicalRAG instance
    return MedicalRAG(vector_stores)

# Example usage
if __name__ == "__main__":
    # Set up the medical RAG system
    medical_rag = setup_medical_rag()
    
    # Simulate a conversation with a user
    user_id = "user123"
    
    # First question
    print("\n--- New conversation with user123 ---")
    question1 = "What are the side effects of Lipitor?"
    print(f"User: {question1}")
    result1 = medical_rag.process_query(user_id, question1)
    print(f"Assistant: {result1.answer}")
    
    # Second question (follow-up with ambiguity)
    question2 = "Are any of these serious?"
    print(f"\nUser: {question2}")
    result2 = medical_rag.process_query(user_id, question2)
    if result2.clarified_query:
        print(f"[Clarified to: {result2.clarified_query}]")
    print(f"Assistant: {result2.answer}")
    
    # Third question (topic change)
    question3 = "How does metformin work?"
    print(f"\nUser: {question3}")
    result3 = medical_rag.process_query(user_id, question3)
    print(f"Assistant: {result3.answer}")
    
    # Fourth question (ambiguous follow-up)
    question4 = "What are its side effects?"
    print(f"\nUser: {question4}")
    result4 = medical_rag.process_query(user_id, question4)
    if result4.clarified_query:
        print(f"[Clarified to: {result4.clarified_query}]")
    print(f"Assistant: {result4.answer}")
    
    # Fifth question (complex query)
    question5 = "Compare the side effects of Lipitor and Crestor and tell me which one is safer for elderly patients with kidney problems?"
    print(f"\nUser: {question5}")
    result5 = medical_rag.process_query(user_id, question5)
    print(f"Assistant: {result5.answer}")

    # Get conversation history
    print("\n--- Conversation History ---")
    history = medical_rag.get_conversation_history(user_id)
    for i, entry in enumerate(history):
        print(f"[{i+1}] User: {entry.user_query}")
        print(f"    Assistant: {entry.answer[:100]}...")  # Show first 100 chars


