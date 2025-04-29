import asyncio
import enum
import uuid
import pytest
from unittest.mock import patch, AsyncMock

# +++++ NEW IMPORTS +++++++++++++++++++++++++++++++++++++++++++++++++++
from rag_agent import (
    setup_medical_rag,
    ClarifiedQuery,
    DecomposedQuery,
    QueryResult,
    RelevantHistoryContext,
    CitedAnswerResult,
)

# Create a shared MedicalRAG to service the wrappers
_medical_rag = setup_medical_rag()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Wrappers that forward to the production components
async def clarify(query: str) -> ClarifiedQuery:
    return await _medical_rag.preprocessor.clarify_query_async(
        user_query=query, conversation_context=""
    )  # type: ignore


async def decompose(query: str) -> DecomposedQuery:
    return await _medical_rag.preprocessor.decompose_query_async(query)  # type: ignore


async def retrieve(query: str) -> list[QueryResult]:
    return await _medical_rag.router.route_query_async(query)


async def evaluate_retrieval(
    query: str, results: list[QueryResult]
) -> list[QueryResult]:
    return await _medical_rag.evaluator.evaluate_retrieval(
        original_query=query,
        clarified_query=query,
        retrieval_results=results,
        router=_medical_rag.router,
    )


async def summarize(history: str) -> RelevantHistoryContext:
    # Pass empty list for conversation history in this example
    return await _medical_rag.context_processor.extract_relevant_context(
        query=history, conversation_history=[]
    )


async def answer(
    results: list[QueryResult], query: str, summary: RelevantHistoryContext
) -> CitedAnswerResult:
    return await _medical_rag.generator.generate_answer_async(
        user_question=query,
        retrieval_results=results,
        conversation_context=summary.relevant_snippets or "",
    )


# ---- Processing Branch Class ----


class BranchStatus(enum.Enum):
    ACTIVE = 1
    SUPERSEDED = 2  # Replaced by a more refined branch (e.g., clarified)
    COMPLETED = 3  # Successfully produced an answer
    FAILED = 4  # An error occurred


class ProcessingBranch:
    def __init__(self, query: str, branch_type: str, parent_id: str | None = None):
        self.branch_id: str = str(uuid.uuid4())
        self.query: str = query
        self.branch_type: str = branch_type
        self.parent_id: str | None = parent_id
        self.status: BranchStatus = BranchStatus.ACTIVE
        self.tasks: dict[str, asyncio.Task] = {}
        self.retrieved_results: list[QueryResult] | None = None
        self.merged_results: list[QueryResult] | None = None
        self.final_answer: CitedAnswerResult | None = None
        print(
            f"BRANCH CREATED: {self.branch_id} (Type: {self.branch_type}, Query: '{self.query}')"
        )

    def add_task(self, task_name: str, task: asyncio.Task):
        if self.status == BranchStatus.ACTIVE:
            self.tasks[task_name] = task
            print(f"  TASK ADDED to {self.branch_id}: {task_name}")
        else:
            print(
                f"  TASK IGNORED (Branch {self.branch_id} status: {self.status.name}): {task_name}"
            )
            task.cancel()  # Cancel immediately if branch isn't active

    def cancel_task(self, task_name: str, active_tasks: dict[asyncio.Task, str]):
        if task_name in self.tasks:
            task = self.tasks.pop(task_name)
            if not task.done():
                task.cancel()
                print(f"  TASK CANCELLED in {self.branch_id}: {task_name}")
            # Remove from active_tasks if present
            if task in active_tasks:
                del active_tasks[task]

    def cancel_all_tasks(
        self, active_tasks: dict[asyncio.Task, str]
    ) -> list[asyncio.Task]:
        print(f"BRANCH CANCEL ALL: {self.branch_id}")
        task_names = list(self.tasks.keys())
        cancelled = []
        for task_name in task_names:
            if task_name in self.tasks:
                task = self.tasks[task_name]
                self.cancel_task(task_name, active_tasks)
                cancelled.append(task)
        self.tasks.clear()
        return cancelled

    def get_task_details(
        self, task: asyncio.Task
    ) -> tuple[str | None, asyncio.Task | None]:
        """Find the name and task object matching the completed task."""
        for name, t in self.tasks.items():
            if t is task:
                # Don't remove it here, let the main loop handle removal after processing
                return name, t
        return None, None


# ---- Orchestrator Refactored ----
class RAGOrchestrator:
    def __init__(self):
        self.branches = {}  # Map branch_id -> ProcessingBranch
        self.active_tasks = {}  # Map task object -> branch_id
        self.retrieval_cache = {}  # Map query -> list of docs
        self.summary_result = None
        self.final_answer_source_branch_id = None

    async def process_query(self, user_query: str, history: str) -> CitedAnswerResult | None:
        """Main entry point to process a user query with RAG orchestration."""
        self.summary_task = asyncio.create_task(summarize(history))

        # Create initial branch and launch first tasks
        initial_branch = self._create_branch(user_query, "initial")
        self._launch_initial_tasks(initial_branch, user_query)

        # Process tasks until completion
        await self._process_tasks_until_completion()

        # Select best answer
        return self._select_best_answer()

    def _create_branch(
        self, query: str, branch_type: str, parent_id: str = None
    ) -> ProcessingBranch:
        """Create a new processing branch and add it to the branches dictionary."""
        branch = ProcessingBranch(
            query=query, branch_type=branch_type, parent_id=parent_id
        )
        self.branches[branch.branch_id] = branch
        return branch

    def _launch_task(self, branch: ProcessingBranch, task_name: str, coro):
        """Launch a task and register it with the branch and active tasks."""
        if branch.status != BranchStatus.ACTIVE:
            print(
                f"Skipping task '{task_name}' launch for inactive branch {branch.branch_id}"
            )
            return
        task = asyncio.create_task(coro)
        branch.add_task(task_name, task)
        self.active_tasks[task] = branch.branch_id

    def _launch_initial_tasks(self, branch: ProcessingBranch, query: str):
        """Launch clarify, decompose, retrieve."""
        self._launch_task(branch, "clarify", clarify(query))
        self._launch_task(branch, "decompose", decompose(query))
        self._launch_task(branch, "retrieve", retrieve(query))

    def _supersede_branch(self, branch_id: str):
        """Mark a branch as superseded and cancel its tasks."""
        if branch_id in self.branches:
            branch = self.branches[branch_id]
            if branch.status == BranchStatus.ACTIVE:
                print(f"BRANCH SUPERSEDED: {branch_id} (Type: {branch.branch_type})")
                branch.status = BranchStatus.SUPERSEDED
                cancelled_tasks = branch.cancel_all_tasks(self.active_tasks)
                if cancelled_tasks:
                    asyncio.ensure_future(
                        asyncio.gather(*cancelled_tasks, return_exceptions=True)
                    )

    def _get_active_branch_tasks(self) -> list[asyncio.Task]:
        """Get all active tasks from active branches."""
        return [
            task
            for task, branch_id in self.active_tasks.items()
            if branch_id in self.branches
            and self.branches[branch_id].status == BranchStatus.ACTIVE
            and not task.done()
        ]

    async def _process_tasks_until_completion(self):
        """Process tasks until all are complete or cancelled."""
        while True:
            current_active_tasks = self._get_active_branch_tasks()

            # Determine if we should continue or break
            if not current_active_tasks and not self.summary_task.done():
                # If only summary is left, wait for it specifically
                await asyncio.wait(
                    [self.summary_task], return_when=asyncio.FIRST_COMPLETED
                )
            elif not current_active_tasks and self.summary_task.done():
                # All work finished or cancelled
                print("No active tasks remaining.")
                break

            # Prepare tasks to wait for
            tasks_to_wait = current_active_tasks + (
                [self.summary_task] if not self.summary_task.done() else []
            )
            if not tasks_to_wait:
                print("Breaking loop: No tasks to wait for.")
                break

            # Wait for next task completion
            print(f"\nWaiting for {len(tasks_to_wait)} tasks...")
            done, pending = await asyncio.wait(
                tasks_to_wait, return_when=asyncio.FIRST_COMPLETED
            )
            print(f"Done tasks: {len(done)}, Pending tasks: {len(pending)}")

            # Process completed tasks
            await self._process_completed_tasks(done)

    async def _process_completed_tasks(self, done_tasks):
        """Process each completed task based on its type."""
        # Get summary result if ready
        self.summary_result = (
            await self.summary_task if self.summary_task.done() else None
        )

        for task in done_tasks:
            # Handle summary task specifically
            if task is self.summary_task:
                print("Summary task completed.")
                self.summary_result = task.result()
                continue

            # Process regular task
            await self._process_individual_task(task)

        # Check if summary is ready to trigger pending answers
        if self.summary_result:
            self._trigger_answers_with_summary()

    async def _process_individual_task(self, task):
        """Process a single completed task."""
        # Get branch information
        branch_id = self.active_tasks.pop(task, None)
        if not branch_id or branch_id not in self.branches:
            print(f"Ignoring task result: Task {task} has no associated branch.")
            return

        branch = self.branches[branch_id]
        if branch.status != BranchStatus.ACTIVE:
            print(
                f"Ignoring task result: Branch {branch_id} is not ACTIVE (Status: {branch.status.name})."
            )
            return

        task_name, _ = branch.get_task_details(task)
        if not task_name:
            print(
                f"WARNING: Completed task {task} not found in branch {branch_id}.tasks"
            )
            return

        branch.tasks.pop(task_name)  # Remove completed task from branch
        print(
            f"Processing result for Task '{task_name}' from Branch {branch.branch_id} ({branch.branch_type})"
        )

        try:
            result = task.result()
            # Handle task result based on task type
            await self._handle_task_result(task_name, result, branch)

        except asyncio.CancelledError:
            print(f"Task '{task_name}' in branch {branch_id} was cancelled.")
        except Exception as e:
            print(f"!!! Task '{task_name}' in branch {branch_id} failed: {e}")
            branch.status = BranchStatus.FAILED

    async def _handle_task_result(self, task_name, result, branch):
        """Handle task result based on task type."""
        task_handlers = {
            "clarify": self._handle_clarify_result,
            "decompose": self._handle_decompose_result,
            "retrieve": self._handle_retrieve_result,
            "evaluate": self._handle_evaluate_result,  # NEW
            "answer": self._handle_answer_result,
        }

        handler = task_handlers.get(task_name)
        if handler:
            await handler(result, branch)
        else:
            print(f"No handler for task type '{task_name}'")

    async def _handle_clarify_result(self, clar_obj: ClarifiedQuery, branch):
        clarified_query = clar_obj.clarified_query
        if clarified_query != branch.query:
            self._supersede_branch(branch.branch_id)
            new_b = self._create_branch(clarified_query, "clarified", branch.branch_id)
            self._launch_task(new_b, "retrieve", retrieve(clarified_query))
            self._launch_task(new_b, "decompose", decompose(clarified_query))
        else:
            print("  Clarification resulted in no change.")

    async def _handle_decompose_result(self, decomp_obj: DecomposedQuery, branch):
        sub_qs = decomp_obj.decomposed_query or []
        if sub_qs and sub_qs != [branch.query]:
            branch.cancel_task("retrieve", self.active_tasks)
            branch.cancel_task("answer", self.active_tasks)
            for i, q in enumerate(sub_qs):
                nb = self._create_branch(q, f"decomposed_{i}", branch.branch_id)
                self._launch_task(nb, "retrieve", retrieve(q))

    async def _handle_retrieve_result(self, results: list[QueryResult], branch):
        print(f"  Retrieved {sum(len(r.docs) for r in results)} docs")
        branch.retrieved_results = results
        branch.merged_results = results
        self._launch_task(branch, "evaluate", evaluate_retrieval(branch.query, results))

    async def _handle_evaluate_result(self, merged: list[QueryResult], branch):
        branch.merged_results = merged
        if self.summary_result is not None:
            self._launch_task(
                branch, "answer", answer(merged, branch.query, self.summary_result)
            )

    async def _handle_answer_result(self, ans: CitedAnswerResult, branch):
        branch.final_answer = ans
        branch.status = BranchStatus.COMPLETED
        self.final_answer_source_branch_id = branch.branch_id
        print(
            f"*** Answer obtained from Branch {branch.branch_id} ({branch.branch_type}) ***"
        )

    def _trigger_answers_with_summary(self):
        """Trigger answer tasks for branches that have docs but no answer yet."""
        for b in self.branches.values():
            if (
                b.status == BranchStatus.ACTIVE
                and b.merged_results is not None
                and "answer" not in b.tasks
            ):
                self._launch_task(
                    b, "answer", answer(b.merged_results, b.query, self.summary_result) # type: ignore
                )

    def _select_best_answer(self) -> CitedAnswerResult | None:
        """Select the best answer from completed branches."""
        print("\n--- Orchestration Complete ---")
        for b_id, b in self.branches.items():
            print(
                f"  Branch {b_id[-8:]} ({b.branch_type}): "
                f"Status={b.status.name}, GapDocs={bool(b.merged_results and b.retrieved_results and len(b.merged_results) > len(b.retrieved_results))}"
            )

        completed = [
            b
            for b in self.branches.values()
            if b.status == BranchStatus.COMPLETED and b.final_answer is not None
        ]

        if completed:
            best_branch = max(completed, key=self._refinement_tuple)
            print(
                f"Returning answer from best branch "
                f"{best_branch.branch_id[-8:]} with refinement {self._refinement_tuple(best_branch)}"
            )
            return best_branch.final_answer

        print("No completed branch found with an answer.")
        return None

    def _refinement_tuple(self, branch) -> tuple[int, int, int]:
        """Calculate a refinement tuple for ranking branches."""
        clarified = (
            1
            if (
                branch.branch_type == "clarified"
                or (
                    branch.parent_id
                    and self.branches[branch.parent_id].branch_type == "clarified"
                )
            )
            else 0
        )
        decomposed = 1 if branch.branch_type.startswith("decomposed") else 0
        gap_filled = (
            1
            if (
                branch.merged_results
                and branch.retrieved_results
                and len(branch.merged_results) > len(branch.retrieved_results)
            )
            else 0
        )
        return (clarified, decomposed, gap_filled)


# --- Example Usage ---
async def run_orchestrator(query: str, history: str) -> CitedAnswerResult | None:
    orchestrator = RAGOrchestrator()
    return await orchestrator.process_query(query, history)


# Modify the main section to use the refactored class
if __name__ == "__main__":
    # Example 1: Query likely to be clarified and decomposed
    print("\n--- RUN 1: Clarify & Decompose ---")
    result1 = asyncio.run(
        run_orchestrator("what is lipitor?", "History 1...")
    )
    print("\nFinal Result 1:", result1)

    print("\n" + "=" * 20 + "\n")

    # Example 2: Query less likely to need clarification/decomposition
    print("\n--- RUN 2: Simpler Query ---")
    result2 = asyncio.run(
        run_orchestrator("Tell me about side effects.", "History 2...")
    )
    print("\nFinal Result 2:", result2)
