# Hybrid RAG Agent with Multi-Stage Answer Validation

This project implements a sophisticated Retrieval-Augmented Generation (RAG) system designed to answer questions grounded in healthcare product monographs (e.g., Lipitor, Metformin). It tackles the challenge of providing accurate, grounded answers quickly, even for complex or ambiguous queries, by employing an **asynchronous orchestration strategy with speculative execution**.

Instead of a rigid sequential pipeline, the system concurrently explores multiple processing paths. Key capabilities include:

*   **Intelligent Query Handling:** Utilizes conversation history for query clarification and decomposes complex questions into simpler sub-queries.
*   **Targeted Hybrid Retrieval:** Employs OpenAI function calling to route queries to the correct Weaviate vector store (Lipitor or Metformin) and retrieves relevant document chunks using Weaviate's hybrid search (BM25 + OpenAI embeddings with `relativeScoreFusion`).
*   **Context Enhancement:** Summarizes relevant snippets from conversation history and evaluates retrieval sufficiency, performing gap-filling via additional sub-queries if necessary.
*   **Validated Answer Synthesis:** Generates a freeform answer incorporating retrieved context and history summary, followed by a rigorous multi-step validation process detailed further below. This validation ensures the final answer is factually grounded in the source documents by checking cited evidence.
*   **Dialogue Promotion:** Suggests relevant follow-up questions based on the interaction.

This orchestrated approach, powered by technologies like Weaviate, OpenAI, and Docling (for document processing), aims to deliver fast, accurate, and context-aware responses for healthcare information retrieval.

---

## Core Pipeline Components

**Clarification & Decomposition:**
*   **Clarification:** Uses conversation history to interpret follow-up questions containing ambiguous references (like pronouns) that depend on previous turns in the dialogue.
*   **Decomposition:** Breaks down complex questions into multiple, focused sub-queries specifically for retrieval. This process operates independently of conversation history.

**Conversation Context Summarization:** Before generating an answer, this component analyzes the current user query and the preceding conversation history. It identifies and extracts key snippets from the history that are relevant for providing context or answering the current question. This summary is then passed along to the answer generation step.

**Document Retrieval (Weaviate Hybrid RAG):** An LLM function call first analyzes the user query to determine the relevant medication, routing the request to the specific Weaviate vector store for either "Lipitor" or "Metformin". The system then retrieves relevant **document chunks** using Weaviate's hybrid search capabilities. The specifics of this hybrid search (combining dense and sparse methods with Relative Score Fusion) are detailed in the "Retrieval Engine Details" section below.

**Retrieval Evaluation & Gap-Filling:** After the initial retrieval, this component assesses whether the collected document chunks contain sufficient information to answer the user's query thoroughly. If the initial context is deemed insufficient, the evaluator generates new, targeted sub-queries to fetch additional relevant document chunks. This augmented set of chunks is then passed on for answer generation.

**Answer Generation:** This component receives the user query (potentially clarified or decomposed) and the final set of retrieved document chunks (potentially augmented by gap-filling). Using this context,and the summarized history, an LLM generates a freeform answer, aiming to include citations pointing back to the source documents.

**Answer Validation:** The initial freeform answer undergoes a rigorous validation process to check for factual grounding and handle potential hallucinations. See the "Detailed Answer Validation and Hallucination Handling" section below for specifics.

**Follow-Up Question Generation:** Based on the final answer and conversation context, the system can also generate relevant follow-up questions to guide the user or explore related topics.

**Conceptual Data Flow:**

The following diagram illustrates a simplified, *conceptual* linear flow, highlighting the sequential dependencies between components if they were executed strictly one after another. This contrasts with the actual concurrent, speculative execution shown in the diagram further below, which aims for faster results when possible.

```mermaid
  %%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#ffcc00', 'edgeLabelBackground':'#ffffff', 'tertiaryColor': '#eee'}}}%%
  flowchart TD
    %% Inputs
    Q("User Query"):::queryStyle
    H("Conversation History"):::summaryStyle

    %% Processing Steps (Strictly Sequential)
    C{Clarification?}:::clarificationStyle
    D{Decomposition?}:::decompositionStyle
    S["History Summary"]:::summaryStyle
    R["Retrieval (uses refined Q)"]:::retrievalStyle
    E["Evaluate Retrieval"]:::retrievalStyle
    A["Answer Generation"]:::gapAnswerStyle
    V{{"Validate Answer"}}:::validateStyle
    F("Follow-up Questions"):::queryStyle

    %% Linear Flow (No superseding)
    Q --> C
    H --> C
    C --> D
    H --> S
    D --> R
    S --> A
    R --> E
    E --> A
    A --> V
    V --> F

    %% Styles
    classDef queryStyle fill:#a3c9a8,stroke:#000,stroke-width:2px;
    classDef summaryStyle fill:#fff2cc,stroke:#000,stroke-width:2px;
    classDef retrievalStyle fill:#f4cccc,stroke:#000,stroke-width:1px;
    classDef validateStyle fill:#d9ead3,stroke:#000,stroke-width:2px;
    classDef clarificationStyle fill:#f6b26b,stroke:#000,stroke-width:1px;
    classDef decompositionStyle fill:#b4a7d6,stroke:#000,stroke-width:2px;
    classDef gapAnswerStyle fill:#ffe599,stroke:#000,stroke-width:2px;
  ```

---

## Speculative Execution & Orchestration

To balance response speed with accuracy, the system employs an asynchronous orchestration strategy that explores multiple processing paths concurrently.

The core idea is **speculative execution**: immediately upon receiving a query, the system starts retrieving documents and working towards an answer based on the *original query* (the "fast path"). Simultaneously, it analyzes the query for potential ambiguity or complexity that might require clarification or decomposition.

*   **If the initial query is clear and simple:** The fast path proceeds quickly, potentially leading to a faster final answer after validation.
*   **If the query needs refinement:** The clarification or decomposition steps produce a better query formulation. The initial speculative path is then halted, and new paths using the refined query take precedence. This ensures accuracy even if it takes slightly longer.

This concurrent approach allows the system to potentially bypass slower refinement steps for simple queries, while still ensuring that complex or ambiguous queries are properly handled before generating a final, validated answer. The orchestrator manages these parallel paths, cancels superseded work, and ultimately selects the best validated answer available, prioritizing those derived from more refined queries when applicable.

**Speculative Execution Flow Diagram:**

The following diagram illustrates this speculative execution flow:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#ffcc00', 'edgeLabelBackground':'#ffffff', 'tertiaryColor': '#eee'}}}%%
flowchart TD
  %% Inputs
  Q(("User Query")):::queryStyle
  Summary(("History Summary")):::summaryStyle
  Q --> Summary

  %% Fast path (boxed)
  subgraph FastPath ["Optimistic Fast Path"]
    class FastPath fastPathStyle
    R0{"Initial Retrieval"}:::retrievalStyle
    FA(("Fast Answer")):::fastAnswerStyle
    R0 --> FA
    Summary --> FA
    FA --> V{{"Validate (Terminal)"}}:::validateStyle
  end

  Q --> R0

  %% Speculative execution
  Q --> C{Clarification}:::clarificationStyle
  Q --> D(("Decomposition")):::decompositionStyle

  %% Supersede fast path and decomposition
  C -.->|supersede| R0
  C -.->|supersede| FA
  C -.->|supersede| D
  D -.->|supersede| R0
  D -.->|supersede| FA

  %% Speculative retrievals
  C --> RC{"Retrieval (Q′)"}:::retrievalStyle
  D --> RS{"Retrieval (subs)"}:::retrievalStyle

  %% Gap-fill and answer
  RC --> G1(("Gap & Answer")):::gapAnswerStyle
  RS --> G2(("Gap & Answer")):::gapAnswerStyle
  Summary --> G1
  Summary --> G2
  G1 --> V
  G2 --> V

  %% Styles
  classDef queryStyle fill:#a3c9a8,stroke:#000,stroke-width:2px;
  classDef summaryStyle fill:#fff2cc,stroke:#000,stroke-width:2px;
  classDef fastPathStyle fill:#6fa8dc,stroke:#000,stroke-width:1px;
  classDef retrievalStyle fill:#f4cccc,stroke:#000,stroke-width:1px;
  classDef fastAnswerStyle fill:#d9edf7,stroke:#000,stroke-width:2px;
  classDef validateStyle fill:#d9ead3,stroke:#000,stroke-width:2px;
  classDef clarificationStyle fill:#f6b26b,stroke:#000,stroke-width:1px;
  classDef decompositionStyle fill:#b4a7d6,stroke:#000,stroke-width:2px;
  classDef gapAnswerStyle fill:#ffe599,stroke:#000,stroke-width:2px;
```

---

## Retrieval Engine Details

**Weaviate Hybrid Retrieval:** Document chunks are indexed in Weaviate collections (e.g., `Lipitor`, `Metformin`). Each chunk includes:
*   An OpenAI embedding vector (dense vector) used for semantic search, compared using cosine similarity.
*   Preparation for Weaviate's sparse keyword indexing via **BM25**. BM25 calculates relevance by considering both the frequency of query terms within a chunk (**Term Frequency**) and how unique those terms are across the entire dataset (**Inverse Document Frequency**). It also includes parameters to **normalize for document length**, preventing longer chunks from having an unfair advantage simply due to size. This results in higher scores for chunks where query terms appear relatively often and are distinctive across the corpus.
*   Associated metadata (source, page numbers, etc.).

Queries utilize Weaviate's `hybrid` search function, specifically configured with:
*   **Fusion Strategy:** `relativeScoreFusion`. This method prepares the scores from the vector search and keyword search for combination. It independently normalizes each set of scores using a min-max scaling approach: within each result set (vector or keyword), the highest score is mapped to 1, the lowest score is mapped to 0, and all other scores are scaled proportionally in between. These normalized scores (ranging from 0 to 1) are then combined based on the alpha parameter. As highlighted in Weaviate's documentation, this approach preserves more information about the relative differences between scores compared to the older rank-based `rankedFusion` method.
*   **Alpha Parameter:** Set to `0.65`. This gives slightly more weight to the vector search results (semantic similarity) compared to the keyword search results (exact matches) in the final ranking.

The system utilizes OpenAI's function calling capability to route the query to the appropriate Weaviate collection(s). Predefined function descriptions, one for each collection (e.g., "query_lipitor", "query_metformin"), are provided to the LLM along with the user query. The LLM analyzes the query and selects the relevant function(s) to call, thereby determining which specific collection(s) (Lipitor or Metformin) should be targeted for the subsequent hybrid search.

---

## Document Processing & Indexing

Raw PDF documents (`docs/lipitor.pdf`, etc.) are processed into indexed chunks suitable for retrieval through the following steps:

1.  **Hybrid Chunking (Docling):** The `docling` library parses PDFs, identifying structural elements (headers, tables, etc.). Its `HybridChunker` leverages this structure for initial hierarchical chunking. It then refines these chunks based on an embedding model's tokenizer, splitting oversized chunks and merging undersized consecutive chunks that share the same structural context (e.g., same heading) to balance semantic coherence and token limits.

2.  **Contextualization:** Crucially, `docling` preserves structural metadata (headers, captions) with each chunk. This "contextualized" text, enriched with its location and role, improves the quality and relevance of the resulting vector embeddings.

3.  **Custom Post-Processing:** Additional logic addresses PDF parsing artifacts:
    *   Rejoins sentence fragments split across chunk boundaries.
    *   Merges fragmented table chunks often caused by page breaks.


---

## Setup & Execution

**Requirements:**
*   Python 3.9+
*   Docker & Docker Compose (for Weaviate)
*   OpenAI API Key (set in a `.env` file: `OPENAI_API_KEY="your_key"`)
*   Python dependencies (install via `pip install -r requirements.txt` after populating the file - see TODO within).

**Prepare Documents & Index Data:**

**Start Weaviate:**
```bash
docker compose up -d
```

**Run Agent (Interactive CLI):**
```bash
python -m healthcare_rag
```

---

## Example Query Flow

Consider the query "What are the side effects of Lipitor?". The orchestrator initiates concurrent tasks: summarizing history, clarifying, decomposing, and retrieving based on the original query. Assuming the query is clear and simple, clarification and decomposition branches won't proceed far. History is summarized. An LLM routes the query to the `Lipitor` collection using a function call. Weaviate performs hybrid retrieval. The results are evaluated; if sufficient, answer generation proceeds using the retrieved context and history summary. A validation LLM checks the answer against the sources. Follow-up questions might be generated. The final validated answer and suggestions are returned. If the initial retrieval was insufficient, the evaluation step would trigger gap-filling sub-queries before answer generation.

---

## Detailed Answer Validation and Hallucination Handling

To ensure the generated answers are factually grounded in the provided documents and to mitigate hallucinations, the system employs a multi-step validation process after the initial answer generation:

1.  **Initial Generation with Attempted Citations:** The first step involves an LLM generating a freeform answer based on the query, retrieved document chunks, and conversation context. This generation prompt encourages the LLM to include citations referencing the source documents.

2.  **Structured Parsing:** The raw, freeform answer (with attempted citations) is then processed using a structured output method (e.g., an LLM call constrained by a specific format or using a tool like Pydantic). This converts the answer into a structured list of individual "statement" objects.

3.  **Statement & Citation Objects:** Each statement object in the list typically contains:
    *   The text of the individual claim or statement being made.
    *   A corresponding "citation" object.
    The citation object itself contains:
    *   An identifier for the specific document chunk referenced.
    *   The exact quote from that document chunk which supposedly supports the statement.

4.  **Quote Verification Loop:** A validation function iterates through each structured statement object. For each statement, it performs a check:
    *   It retrieves the content of the document chunk referenced in the citation object.
    *   It verifies if the quote provided in the citation object can be found within that document chunk's content using **fuzzy matching**. This allows for minor variations and doesn't require an exact string match, succeeding if the match score exceeds a predefined threshold.

5.  **Statement Filtering:** If the quote verification fails for a statement (i.e., the provided quote is not found in the referenced document chunk, indicating a potential hallucination or mis-citation), that specific statement may be removed from the list.

6.  **Final Validated Output:** The final output consists of the remaining, verified statements, ensuring that the answer presented to the user is directly supported by evidence found within the source documents.

---
