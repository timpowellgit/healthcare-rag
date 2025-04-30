import logging
import re
from typing import Dict, List, Optional, Tuple

from fuzzywuzzy import fuzz, process as fuzzy_process

from ..models.retrieval import QueryDocument, QueryResultList
from ..models.answers import CitedAnswerResult, StatementWithCitations, Citation
from .base import BaseProcessor, log_timing

logger = logging.getLogger("MedicalRAG")

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
        # Use fuzzy matching for more flexible matching
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

    def _clean_statement_text(self, statement_text):
        """Removes any old [doc_X] style markers from the statement text."""
        return re.sub(r"\[doc_\d+\]", "", statement_text).strip()

    def _build_citations_string(self, valid_prompt_ids):
        """Builds a formatted string of citation markers from valid prompt IDs."""
        if not valid_prompt_ids:
            return ""
            
        # Sort prompt IDs naturally and deduplicate
        sorted_prompt_ids = sorted(
            list(set(valid_prompt_ids)), key=lambda x: int(x.split("_")[1])
        )
        return " ".join([f"[{pid}]" for pid in sorted_prompt_ids])

    def _convert_linebreaks(self, linebreaks):
        """Converts linebreaks string representation to actual newline characters."""
        if linebreaks in ["\\n", "\n"]:
            return "\n"
        elif linebreaks in ["\\n\\n", "\n\n"]:
            return "\n\n"
        elif linebreaks in ["\\n\\n\\n", "\n\n\n"]:
            return "\n\n\n"
        elif linebreaks != "":
            # Log a warning if the value is unexpected but non-empty
            logger.warning(f"Unexpected linebreak value received: {linebreaks!r}. Treating as no linebreak.")
        return ""

    def _format_statement(self, statement_text, valid_prompt_ids, linebreaks):
        """Formats a statement with its valid citation markers."""
        # Clean the statement text
        cleaned_text = self._clean_statement_text(statement_text)
        
        # Build the citations string
        citations_str = self._build_citations_string(valid_prompt_ids)
        
        # Convert linebreaks string to actual newlines
        actual_linebreak = self._convert_linebreaks(linebreaks)
        
        # Construct the final string
        parts = []
        if cleaned_text:
            parts.append(cleaned_text)
        if citations_str:
            parts.append(citations_str)
        
        # Join parts and append the actual linebreak character(s)
        formatted_statement = " ".join(parts)
        if actual_linebreak:
            formatted_statement += actual_linebreak
        
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