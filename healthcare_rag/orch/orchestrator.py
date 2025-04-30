"""
Main orchestrator logic for the healthcare RAG system.

This module coordinates the different processing steps (clarification, decomposition,
retrieval, generation, validation) using branches and asynchronous tasks.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Coroutine
from dataclasses import dataclass

from ..pipeline.medical_rag import MedicalRAG
from ..models.answers import AnswerGenerationResult, CitedAnswerResult, RelevantHistoryContext
from ..models.queries import ClarifiedQuery, DecomposedQuery
from ..models.retrieval import QueryResultList
from ..models.misc import ConversationEntry, FollowUpQuestions

from .branch import ProcessingBranch, BranchStatus
from .tasks import (
    clarify_query, decompose_query, retrieve_documents, evaluate_retrieval,
    extract_conversation_context, generate_answer, validate_answer, generate_follow_ups
)
from .scheduler import (
    launch_task, wait_for_first_completed, process_task_result,
    get_active_branch_tasks, supersede_branch
)
from .monitor import QueryMonitor # Optional: For progress reporting
from .handlers import TaskHandler # Import the new handler class

logger = logging.getLogger("MedicalRAG")

class RefactoredOrchestrator:
    """
    Orchestrates the RAG pipeline using refactored components.
    
    Manages branches, schedules tasks, handles results, and selects the final answer.
    """
    def __init__(self, medical_rag: MedicalRAG):
        """
        Initialize the orchestrator.
        
        Args:
            medical_rag: The configured MedicalRAG instance.
        """
        self.medical_rag = medical_rag
        self.branches: Dict[str, ProcessingBranch] = {}
        self.active_tasks: Dict[asyncio.Task, str] = {} # Maps Task -> branch_id
        # self.retrieval_cache = {} # Consider if caching is needed in refactored version
        self.summary_result: Optional[RelevantHistoryContext] = None
        self.final_answer_source_branch_id: Optional[str] = None
        self.summary_task: Optional[asyncio.Task] = None
        self.user_id: Optional[str] = None
        self.query_monitor: Optional[QueryMonitor] = None # Optional monitoring hook
        
        # Instantiate the TaskHandler
        self.task_handler_instance = TaskHandler(self)
        
        # Map task names to their handler methods on the TaskHandler instance
        self.task_handlers: Dict[str, Callable[[Any, ProcessingBranch], Coroutine]] = {
            "clarify": self.task_handler_instance.handle_clarify_result,
            "decompose": self.task_handler_instance.handle_decompose_result,
            "retrieve": self.task_handler_instance.handle_retrieve_result,
            "evaluate": self.task_handler_instance.handle_evaluate_result,
            "answer": self.task_handler_instance.handle_answer_result,
            "validate": self.task_handler_instance.handle_validate_result,
        }

    # --- Core Orchestration Methods ---

    async def process_query(
        self, user_query: str, user_id: str, monitor: Optional[QueryMonitor] = None
    ) -> Tuple[Optional[str], Optional[List[str]]]:
        """Main entry point to process a user query."""
        self.user_id = user_id
        self.query_monitor = monitor # Store monitor if provided
        self.branches = {} # Reset state for new query
        self.active_tasks = {}
        self.summary_result = None
        self.summary_task = None
        self.final_answer_source_branch_id = None

        processed_history = self.medical_rag.conversation_history.get_history(user_id)
        logger.info(
            f"Orchestrator: Fetched {len(processed_history)} history entries for user '{user_id}'"
        )

        history_context_str = (
            self.medical_rag.conversation_history.get_context_from_history(user_id)
        )
        if history_context_str:
            logger.info(f"Orchestrator: Using history context string for clarification.")
        else:
            logger.info("Orchestrator: No history context string for clarification.")

        # Launch summary task (runs in parallel)
        self.summary_task = asyncio.create_task(
             extract_conversation_context(self.medical_rag, user_query, processed_history)
        )

        # Create initial branch and launch starting tasks
        initial_branch = self._create_branch(user_query, "initial")
        self._launch_initial_tasks(initial_branch, user_query, history_context_str)

        # Run the main processing loop
        await self._process_tasks_until_completion()

        # Select the best answer from completed branches
        best_answer_str = self._select_best_answer()
        follow_up_questions = None

        if best_answer_str and self.user_id:
            logger.info(f"Orchestrator: Adding result to history for user '{self.user_id}'")
            self.medical_rag.conversation_history.add_entry(
                self.user_id, user_query, best_answer_str
            )

            logger.info(f"Orchestrator: Generating follow-up questions for user '{self.user_id}'")
            try:
                # Use the generate_follow_ups task directly here
                follow_up_result = await generate_follow_ups(
                    self.medical_rag, user_query, best_answer_str, processed_history
                )
                follow_up_questions = follow_up_result.questions
            except Exception as e:
                logger.error(f"Failed to generate follow-up questions: {e}", exc_info=True)
                follow_up_questions = ["Error generating follow-ups."]

        elif self.user_id:
            logger.info(
                f"Orchestrator: No final answer found, not adding to history or generating follow-ups for user '{self.user_id}'"
            )

        return best_answer_str, follow_up_questions

    async def _process_tasks_until_completion(self):
        """Main loop to wait for and process tasks."""
        while True:
            # Use scheduler helper to get tasks from active branches
            current_active_branch_tasks = get_active_branch_tasks(self.active_tasks, self.branches)
            
            # Determine if the summary task is still pending
            summary_is_pending = self.summary_task and not self.summary_task.done()

            # --- Loop Exit Condition ---
            # Exit if there are no active branch tasks AND the summary task is no longer pending
            if not current_active_branch_tasks and not summary_is_pending:
                logger.info("No active branch tasks remaining and summary task is complete or gone. Exiting processing loop.")
                break

            # --- Prepare tasks to wait for ---
            tasks_to_wait = current_active_branch_tasks[:] # Create a copy
            if summary_is_pending and self.summary_task is not None: # Ensure task exists before appending
                tasks_to_wait.append(self.summary_task)

            if not tasks_to_wait:
                 # This should theoretically not be reached due to the exit condition above, but acts as a safeguard.
                 logger.warning("No tasks to wait for, but loop condition didn't exit. Breaking.")
                 break

            # --- Wait for first task completion ---
            logger.debug(f"\nWaiting for {len(tasks_to_wait)} tasks... ({len(current_active_branch_tasks)} branch, {1 if summary_is_pending else 0} summary)")
            # Use scheduler helper to wait
            done, pending = await wait_for_first_completed(tasks_to_wait)
            logger.debug(f"Done tasks: {len(done)}, Pending tasks: {len(pending)}")

            # --- Process completed tasks ---
            await self._process_completed_tasks(done)

            # Small sleep to prevent busy-waiting if tasks complete extremely fast
            await asyncio.sleep(0.01) 

    async def _process_completed_tasks(self, done_tasks: set[asyncio.Task]):
        """Process a set of completed tasks."""
        # Separate summary task handling from branch tasks
        summary_finished = False
        if self.summary_task and self.summary_task in done_tasks:
            logger.info("Summary task completed.")
            try:
                # Await the task to get the result or raise exception
                self.summary_result = await self.summary_task 
                summary_finished = True
            except asyncio.CancelledError:
                logger.warning("Summary task was cancelled.")
                self.summary_result = None 
            except Exception as e:
                logger.error(f"!!! Summary task failed: {e}", exc_info=True)
                self.summary_result = None
            done_tasks.remove(self.summary_task)
            self.summary_task = None # Clear the task reference

        # Process results for completed branch tasks
        for task in done_tasks:
            # Get the associated branch_id and remove task from active tracking
            branch_id = self.active_tasks.pop(task, None)
            if branch_id:
                 # Use the scheduler's helper to process the result,
                 # providing our callback to handle the specific logic
                 await process_task_result(
                     task=task,
                     branch_id=branch_id,
                     branches=self.branches,
                     active_tasks=self.active_tasks,
                     result_handler=self._handle_task_result_callback # Use the callback
                 )
            else:
                 # This might happen if a task completed after its branch was superseded
                 logger.warning(f"Ignoring task result: Task {task} no longer in active_tasks.")

        # If the summary finished in this batch, check if any branches are ready for answers
        if summary_finished:
            self._trigger_answers_with_summary()
        
    async def _handle_task_result_callback(self, task_name: str, result: Any, branch: ProcessingBranch):
        """Callback passed to scheduler.process_task_result to route to specific handlers via the TaskHandler instance."""
        handler_method = self.task_handlers.get(task_name)
        if handler_method:
            if branch.status == BranchStatus.ACTIVE:
                # Call the appropriate method on the TaskHandler instance
                await handler_method(result, branch) 
            else:
                logger.info(f"Skipping handler for '{task_name}' on inactive branch {branch.branch_id}")
        else:
            # This maps to the handler method name (e.g., handle_clarify_result)
            logger.warning(f"No handler method found for task type '{task_name}' in orchestrator task_handlers dict")



    def _create_branch(
        self, query: str, branch_type: str, parent_id: Optional[str] = None
    ) -> ProcessingBranch:
        """Create a new processing branch."""
        branch = ProcessingBranch(
            query=query, branch_type=branch_type, parent_id=parent_id
        )
        self.branches[branch.branch_id] = branch
        return branch

    def _launch_task_wrapper(self, branch: ProcessingBranch, task_name: str, coro: Coroutine):
        """Wrapper to use the scheduler's launch_task function."""
        return launch_task(branch, task_name, coro, self.active_tasks)
        
    def _supersede_branch_wrapper(self, branch_id: str):
        """Wrapper to use the scheduler's supersede_branch function."""
        return supersede_branch(branch_id, self.branches, self.active_tasks)

    def _launch_initial_tasks(
        self, branch: ProcessingBranch, query: str, history_context_str: str
    ):
        """Launch the first set of tasks for the initial branch."""
        self._launch_task_wrapper(branch, "clarify", clarify_query(self.medical_rag, query, history_context_str))
        self._launch_task_wrapper(branch, "decompose", decompose_query(self.medical_rag, query))
        self._launch_task_wrapper(branch, "retrieve", retrieve_documents(self.medical_rag, query))


    # --- Summary and Answer Selection ---
    
    def _trigger_answers_with_summary(self):
        """Check branches and trigger answer task if summary is ready."""
        # This method should only be called *after* self.summary_result is set.
        if self.summary_result is None:
            logger.warning("_trigger_answers_with_summary called but summary_result is None")
            return
            
        logger.debug(f"Checking branches to trigger answer task now that summary is ready...")
        triggered_count = 0
        for b_id, b in self.branches.items():
            # Check conditions: Active branch, has results, answer not started/done, validation not started/done
            if (
                b.status == BranchStatus.ACTIVE
                and b.merged_results is not None
                and "answer" not in b.tasks
                and not b.raw_answer
                and "validate" not in b.tasks
                and not b.validated_answer_str
            ):
                logger.info(f"Triggering answer task for branch {b.branch_id} with summary.")
                self._launch_task_wrapper(
                    b,
                    "answer",
                    generate_answer(self.medical_rag, b.merged_results, b.query, self.summary_result),
                )
                triggered_count += 1

        if triggered_count > 0:
            logger.debug(f"Triggered answer task for {triggered_count} branches.")
        else:
            logger.debug("No branches needed answer task triggering in this check.")
        
    @dataclass(frozen=True)
    class BranchTraits:
        """Binary traits that capture how a branch refined the original query."""
        clarified: bool
        decomposed: bool
        gap_filled: bool # Indicates if evaluation added results

        def to_priority_tuple(self) -> tuple[int, int, int]:
            """Convert traits to a tuple suitable for lexicographic comparison."""
            # Higher number means higher priority
            return (
                int(self.clarified),
                int(self.decomposed),
                int(self.gap_filled),
            )

    def _compute_branch_traits(self, branch: ProcessingBranch) -> "RefactoredOrchestrator.BranchTraits":
        """Derive refinement traits for a given branch."""
        # Check if this branch or its direct parent was a clarification branch
        clarified = branch.branch_type == "clarified" or (
            branch.parent_id is not None
            and branch.parent_id in self.branches
            and self.branches[branch.parent_id].branch_type == "clarified"
        )
        # Check if this branch was created by decomposition
        decomposed = branch.branch_type.startswith("decomposed")
        
        # Check if the number of results increased after evaluation (gap filling)
        # Need careful checks for None values
        gap_filled = False
        if branch.merged_results and branch.retrieved_results:
             merged_count = sum(len(r.docs) for r in branch.merged_results.results if r.docs) if branch.merged_results.results else 0
             retrieved_count = sum(len(r.docs) for r in branch.retrieved_results.results if r.docs) if branch.retrieved_results.results else 0
             gap_filled = merged_count > retrieved_count
             
        return self.BranchTraits(clarified, decomposed, gap_filled)
    # -----------------------------------------------------

    def _select_best_answer(self) -> Optional[str]:
        """Select the best answer from completed branches based on refinement traits."""
        logger.info("\n--- Selecting Best Answer ---")
        
        completed_branches_with_answers = [
            b
            for b in self.branches.values()
            if b.status == BranchStatus.COMPLETED and b.validated_answer_str is not None
        ]

        if not completed_branches_with_answers:
            logger.warning("No completed branch found with a validated answer.")
            # Optional: Check for failed branches with raw answers as a fallback?
            return None

        # Compute traits and find the branch with the highest priority traits
        best_branch = None
        highest_priority = (-1, -1, -1) # Lower than any possible trait tuple

        for branch in completed_branches_with_answers:
            traits = self._compute_branch_traits(branch)
            priority_tuple = traits.to_priority_tuple()
            logger.debug(
                f"  Branch {branch.branch_id[-8:]} ({branch.branch_type}): Status={branch.status.name}, "
                f"Traits={traits}, Priority={priority_tuple}, Validated=Yes"
            )
            if priority_tuple > highest_priority:
                highest_priority = priority_tuple
                best_branch = branch
        
        # Log other branches for comparison (optional but helpful)
        for b_id, b in self.branches.items():
             if b not in completed_branches_with_answers:
                  traits = self._compute_branch_traits(b)
                  validated_status = "Yes" if b.validated_answer_str else "No" 
                  logger.debug(
                       f"  Branch {b_id[-8:]} ({b.branch_type}): Status={b.status.name}, "
                       f"Traits={traits}, Validated={validated_status} (Not considered best)"
                  )

        if best_branch:
             best_traits = self._compute_branch_traits(best_branch)
             logger.info(
                 f"Returning answer from best branch {best_branch.branch_id[-8:]} "
                 f"({best_branch.branch_type}) with traits {best_traits}"
             )
             self.final_answer_source_branch_id = best_branch.branch_id
             return best_branch.validated_answer_str
        else:
             # This case should ideally not happen if completed_branches_with_answers was not empty
             logger.error("Error selecting best answer: No best branch found despite having completed branches.")
             return None 
        


async def run_refactored_orchestrator(
    rag_instance: MedicalRAG, query: str, user_id: str, monitor: Optional[QueryMonitor] = None
) -> Tuple[Optional[str], Optional[List[str]]]:
    orchestrator = RefactoredOrchestrator(rag_instance)
    return await orchestrator.process_query(query, user_id, monitor)
