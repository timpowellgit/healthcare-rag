"""
Task Result Handlers for the Refactored Orchestrator.

This module contains the logic for processing the results of individual RAG tasks
(clarify, decompose, retrieve, evaluate, answer, validate) and deciding the next steps.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional, Tuple

from ..models.answers import AnswerGenerationResult, CitedAnswerResult, RelevantHistoryContext
from ..models.queries import ClarifiedQuery, DecomposedQuery
from ..models.retrieval import QueryResultList
from ..models.misc import ConversationEntry, FollowUpQuestions
from ..pipeline.medical_rag import MedicalRAG
from .branch import ProcessingBranch, BranchStatus
from .tasks import (
    clarify_query, decompose_query, retrieve_documents, evaluate_retrieval,
    generate_answer, validate_answer
)

# Use TYPE_CHECKING to avoid circular import issues if Orchestrator needs types from here
if TYPE_CHECKING:
    from .orchestrator import RefactoredOrchestrator

logger = logging.getLogger("MedicalRAG")

class TaskHandler:
    """
    Encapsulates the logic for handling results from completed orchestrator tasks.
    """
    def __init__(self, orchestrator_ref: 'RefactoredOrchestrator'):
        """
        Initialize the handler with a reference to the main orchestrator.
        
        Args:
            orchestrator_ref: A reference to the RefactoredOrchestrator instance.
        """
        self.orchestrator_ref = orchestrator_ref


    async def handle_clarify_result(self, clar_obj: ClarifiedQuery, branch: ProcessingBranch):
        """Handle the result of a clarify_query task."""
        clarified_query = clar_obj.clarified_query
        if clarified_query != branch.query:
            logger.info(f"  Clarification needed: '{branch.query}' -> '{clarified_query}'")
            # Use the wrapper to supersede the current branch
            self.orchestrator_ref._supersede_branch_wrapper(branch.branch_id)
            # Create a new branch for the clarified query
            new_b = self.orchestrator_ref._create_branch(clarified_query, "clarified", branch.branch_id)
            # Launch retrieve and decompose for the new branch
            self.orchestrator_ref._launch_task_wrapper(new_b, "retrieve", retrieve_documents(self.orchestrator_ref.medical_rag, clarified_query))
            self.orchestrator_ref._launch_task_wrapper(new_b, "decompose", decompose_query(self.orchestrator_ref.medical_rag, clarified_query))
        else:
            logger.info("  Clarification resulted in no change.")

    async def handle_decompose_result(self, decomp_obj: DecomposedQuery, branch: ProcessingBranch):
        """Handle the result of a decompose_query task."""
        sub_qs = decomp_obj.decomposed_query or []
        # Check if decomposition actually happened and produced multiple different questions
        if sub_qs and (len(sub_qs) > 1 or sub_qs[0] != branch.query):
            logger.info(f"  Decomposition needed: '{branch.query}' -> {sub_qs}")
            self.orchestrator_ref._supersede_branch_wrapper(branch.branch_id)
            for i, q in enumerate(sub_qs):
                # Create a new branch for each sub-query
                nb = self.orchestrator_ref._create_branch(q, f"decomposed_{i}", branch.branch_id)
                # Launch retrieve for the sub-query branch
                self.orchestrator_ref._launch_task_wrapper(nb, "retrieve", retrieve_documents(self.orchestrator_ref.medical_rag, q))
        else:
            logger.info("  Decomposition resulted in no change or only one identical subquery.")

    async def handle_retrieve_result(self, results: QueryResultList, branch: ProcessingBranch):
        """Handle the result of a retrieve_documents task."""
        doc_count = sum(len(r.docs) for r in results.results) if results and results.results else 0
        logger.info(f"  Retrieved {doc_count} docs for branch {branch.branch_id}")
        branch.retrieved_results = results
        branch.merged_results = results # Initially, merged is the same as retrieved
        
        if results and results.results:
             # Launch evaluation task
             self.orchestrator_ref._launch_task_wrapper(
                 branch, 
                 "evaluate", 
                 evaluate_retrieval(self.orchestrator_ref.medical_rag, branch.query, results)
             )
        else:
            logger.info(f"  Skipping evaluation for branch {branch.branch_id} due to no retrieval results.")
            

    async def handle_evaluate_result(self, merged: QueryResultList, branch: ProcessingBranch):
        """Handle the result of an evaluate_retrieval task."""
        merged_count = sum(len(r.docs) for r in merged.results) if merged and merged.results else 0
        original_count = sum(len(r.docs) for r in branch.retrieved_results.results) if branch.retrieved_results and branch.retrieved_results.results else 0
        logger.info(f"  Evaluation completed for branch {branch.branch_id}. Merged docs: {merged_count} (original: {original_count})")
        branch.merged_results = merged # Store potentially augmented results
        
        # Check if the summary is ready before launching the answer task
        if self.orchestrator_ref.summary_result is not None:
            logger.info(f"  Summary ready, launching answer task for branch {branch.branch_id}")
            self.orchestrator_ref._launch_task_wrapper(
                branch,
                "answer",
                generate_answer(self.orchestrator_ref.medical_rag, merged, branch.query, self.orchestrator_ref.summary_result),
            )
        else:
            logger.info(f"  Summary not ready, delaying answer task for branch {branch.branch_id}")
            # The _trigger_answers_with_summary method will handle launching this later
        
    async def handle_answer_result(self, ans_gen_result: AnswerGenerationResult, branch: ProcessingBranch):
        """Handle the result of a generate_answer task."""
        branch.raw_answer = ans_gen_result
        logger.info(
            f"  Answer generated for Branch {branch.branch_id}, launching validation task."
        )
        # Launch validation task
        self.orchestrator_ref._launch_task_wrapper(branch, "validate", validate_answer(self.orchestrator_ref.medical_rag, ans_gen_result))

        # --- Monitor Hook: Raw Answer --- 
        # Optionally notify monitor about the raw answer
        monitor = self.orchestrator_ref.query_monitor
        if monitor and hasattr(monitor, 'on_answer_task_completed'):
            if hasattr(ans_gen_result, 'plain_answer'):
                 monitor.on_answer_task_completed(branch.branch_id, ans_gen_result)
            else:
                 logger.warning("Monitor hook expected plain_answer in AnswerGenerationResult")
        # ----------------------------------

    async def handle_validate_result(self, validation_result: Tuple[Optional[CitedAnswerResult], Optional[str]], branch: ProcessingBranch):
        """Handle the result of a validate_answer task."""
        structured_answer, validated_str = validation_result

        if validated_str:
            branch.validated_answer_str = validated_str
            branch.structured_validated_answer = structured_answer # Store structured too
            branch.status = BranchStatus.COMPLETED
            # Don't set final_answer_source_branch_id here, let _select_best_answer decide
            logger.info(
                f"*** Validation successful. Branch {branch.branch_id} ({branch.branch_type}) COMPLETED. ***"
            )
            # --- Monitor Hook: Branch Complete --- 
            # Optionally notify monitor about the completed branch
            monitor = self.orchestrator_ref.query_monitor
            if monitor and hasattr(monitor, 'on_branch_completed'):
                 monitor.on_branch_completed(branch.branch_id, validated_str)
            # -------------------------------------
        else:
            logger.warning(
                f"!!! Validation failed or produced empty result for Branch {branch.branch_id}. Marking as FAILED. !!!"
            )
            branch.status = BranchStatus.FAILED
            # Store the structured answer even if validation failed, might be useful for debugging
            branch.structured_validated_answer = structured_answer 