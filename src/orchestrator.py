import asyncio
import enum
import uuid
import pytest
from unittest.mock import patch, AsyncMock
from pathlib import Path
from dataclasses import dataclass

# +++++ NEW IMPORTS +++++++++++++++++++++++++++++++++++++++++++++++++++
from rag_agent import (
    setup_medical_rag,
    MedicalRAG,
    ClarifiedQuery,
    DecomposedQuery,
    QueryResult,
    RelevantHistoryContext,
    CitedAnswerResult,
    ConversationEntry,
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
    def __init__(self, medical_rag: MedicalRAG):
        self.medical_rag = medical_rag
        self.branches = {}
        self.active_tasks = {}
        self.retrieval_cache = {}
        self.summary_result = None
        self.final_answer_source_branch_id = None
        self.summary_task: asyncio.Task | None = None
        self.user_id: str | None = None

    async def _clarify(self, query: str, context: str = "") -> ClarifiedQuery:
        return await self.medical_rag.preprocessor.clarify_query_async(
            user_query=query, conversation_context=context
        )  # type: ignore

    async def _decompose(self, query: str) -> DecomposedQuery:
        return await self.medical_rag.preprocessor.decompose_query_async(query)  # type: ignore

    async def _retrieve(self, query: str) -> list[QueryResult]:
        return await self.medical_rag.router.route_query_async(query)

    async def _evaluate_retrieval(
        self, query: str, results: list[QueryResult]
    ) -> list[QueryResult]:
        return await self.medical_rag.evaluator.evaluate_retrieval(
            original_query=query,
            clarified_query=query,
            retrieval_results=results,
            router=self.medical_rag.router,
        )

    async def _summarize(self, query: str, history: list[ConversationEntry]) -> RelevantHistoryContext:
        return await self.medical_rag.context_processor.extract_relevant_context(
            query=query, conversation_history=history
        )

    async def _answer(
        self, results: list[QueryResult], query: str, summary: RelevantHistoryContext
    ) -> CitedAnswerResult:
        return await self.medical_rag.generator.generate_answer_async(
            user_question=query,
            retrieval_results=results,
            conversation_context=summary.relevant_snippets or "",
        )

    async def process_query(
        self, user_query: str, user_id: str
    ) -> CitedAnswerResult | None:
        self.user_id = user_id
        processed_history = self.medical_rag.conversation_history.get_history(user_id)
        print(f"Orchestrator: Fetched {len(processed_history)} history entries for user '{user_id}'")

        history_context_str = self.medical_rag.conversation_history.get_context_from_history(user_id)
        if history_context_str:
            print(f"Orchestrator: Using history context string for clarification.")
        else:
            print("Orchestrator: No history context string for clarification.")

        self.summary_task = asyncio.create_task(
            self._summarize(user_query, processed_history)
        )

        initial_branch = self._create_branch(user_query, "initial")
        self._launch_initial_tasks(initial_branch, user_query, history_context_str)

        await self._process_tasks_until_completion()

        best_answer = self._select_best_answer()

        if best_answer and self.user_id:
            print(f"Orchestrator: Adding result to history for user '{self.user_id}'")
            self.medical_rag.conversation_history.add_entry(
                self.user_id, user_query, best_answer.statements
            )
        elif self.user_id:
             print(f"Orchestrator: No final answer found, not adding to history for user '{self.user_id}'")

        return best_answer

    def _create_branch(
        self, query: str, branch_type: str, parent_id: str | None = None
    ) -> ProcessingBranch:
        branch = ProcessingBranch(
            query=query, branch_type=branch_type, parent_id=parent_id
        )
        self.branches[branch.branch_id] = branch
        return branch

    def _launch_task(self, branch: ProcessingBranch, task_name: str, coro):
        if branch.status != BranchStatus.ACTIVE:
            print(
                f"Skipping task '{task_name}' launch for inactive branch {branch.branch_id}"
            )
            return
        task = asyncio.create_task(coro)
        branch.add_task(task_name, task)
        self.active_tasks[task] = branch.branch_id

    def _launch_initial_tasks(self, branch: ProcessingBranch, query: str, history_context_str: str):
        self._launch_task(branch, "clarify", self._clarify(query, history_context_str))
        self._launch_task(branch, "decompose", self._decompose(query))
        self._launch_task(branch, "retrieve", self._retrieve(query))

    def _supersede_branch(self, branch_id: str):
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
        return [
            task
            for task, branch_id in self.active_tasks.items()
            if branch_id in self.branches
            and self.branches[branch_id].status == BranchStatus.ACTIVE
            and not task.done()
        ]

    async def _process_tasks_until_completion(self):
        while True:
            current_active_tasks = self._get_active_branch_tasks()

            summary_done = self.summary_task and self.summary_task.done()

            if not current_active_tasks and not summary_done:
                if self.summary_task:
                    print("\nWaiting only for Summary task...")
                    await asyncio.wait(
                        [self.summary_task], return_when=asyncio.FIRST_COMPLETED
                    )
                else:
                    print("Error: No active tasks and no summary task.")
                    break
            elif not current_active_tasks and summary_done:
                print("No active tasks remaining.")
                break

            tasks_to_wait = current_active_tasks
            if self.summary_task and not summary_done:
                tasks_to_wait.append(self.summary_task)

            if not tasks_to_wait:
                print("Breaking loop: No tasks to wait for.")
                break

            print(f"\nWaiting for {len(tasks_to_wait)} tasks...")
            done, pending = await asyncio.wait(
                tasks_to_wait, return_when=asyncio.FIRST_COMPLETED
            )
            print(f"Done tasks: {len(done)}, Pending tasks: {len(pending)}")

            await self._process_completed_tasks(done)

    async def _process_completed_tasks(self, done_tasks):
        if self.summary_task and self.summary_task in done_tasks:
            print("Summary task completed.")
            try:
                self.summary_result = self.summary_task.result()
            except Exception as e:
                print(f"!!! Summary task failed: {e}")
                self.summary_result = None
            done_tasks.remove(self.summary_task)

        for task in done_tasks:
            await self._process_individual_task(task)

        if self.summary_result is not None:
            self._trigger_answers_with_summary()

    async def _process_individual_task(self, task):
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

        branch.tasks.pop(task_name)
        print(
            f"Processing result for Task '{task_name}' from Branch {branch.branch_id} ({branch.branch_type})"
        )

        try:
            result = task.result()
            await self._handle_task_result(task_name, result, branch)

        except asyncio.CancelledError:
            print(f"Task '{task_name}' in branch {branch_id} was cancelled.")
        except Exception as e:
            print(f"!!! Task '{task_name}' in branch {branch_id} failed: {e}")
            branch.status = BranchStatus.FAILED

    async def _handle_task_result(self, task_name, result, branch):
        task_handlers = {
            "clarify": self._handle_clarify_result,
            "decompose": self._handle_decompose_result,
            "retrieve": self._handle_retrieve_result,
            "evaluate": self._handle_evaluate_result,
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
            print(f"  Clarification needed: '{branch.query}' -> '{clarified_query}'")
            self._supersede_branch(branch.branch_id)
            new_b = self._create_branch(clarified_query, "clarified", branch.branch_id)
            self._launch_task(new_b, "retrieve", self._retrieve(clarified_query))
            self._launch_task(new_b, "decompose", self._decompose(clarified_query))
        else:
            print("  Clarification resulted in no change.")

    async def _handle_decompose_result(self, decomp_obj: DecomposedQuery, branch):
        sub_qs = decomp_obj.decomposed_query or []
        if sub_qs and sub_qs != [branch.query]:
            branch.cancel_task("retrieve", self.active_tasks)
            branch.cancel_task("answer", self.active_tasks)
            for i, q in enumerate(sub_qs):
                nb = self._create_branch(q, f"decomposed_{i}", branch.branch_id)
                self._launch_task(nb, "retrieve", self._retrieve(q))

    async def _handle_retrieve_result(self, results: list[QueryResult], branch):
        print(f"  Retrieved {sum(len(r.docs) for r in results)} docs")
        branch.retrieved_results = results
        branch.merged_results = results
        self._launch_task(
            branch, "evaluate", self._evaluate_retrieval(branch.query, results)
        )

    async def _handle_evaluate_result(self, merged: list[QueryResult], branch):
        branch.merged_results = merged
        if self.summary_result is not None:
            self._launch_task(
                branch,
                "answer",
                self._answer(merged, branch.query, self.summary_result),
            )

    async def _handle_answer_result(self, ans: CitedAnswerResult, branch):
        branch.final_answer = ans
        branch.status = BranchStatus.COMPLETED
        self.final_answer_source_branch_id = branch.branch_id
        print(
            f"*** Answer obtained from Branch {branch.branch_id} ({branch.branch_type}) ***"
        )

    def _trigger_answers_with_summary(self):
        for b in self.branches.values():
            if (
                b.status == BranchStatus.ACTIVE
                and b.merged_results is not None
                and "answer" not in b.tasks
                and self.summary_result is not None
            ):
                print(f"Triggering answer for branch {b.branch_id} with summary.")
                self._launch_task(
                    b,
                    "answer",
                    self._answer(b.merged_results, b.query, self.summary_result),
                )
            elif b.status == BranchStatus.ACTIVE and "answer" not in b.tasks and self.summary_result is None:
                 print(f"Branch {b.branch_id} waiting for summary before triggering answer.")

    # +++++ Branch scoring helpers ++++++++++++++++++++++++++++++++++++++++
    @dataclass(frozen=True)
    class BranchTraits:
        """Binary traits that capture how a branch refined the original query."""
        clarified: bool
        decomposed: bool
        gap_filled: bool

        def to_priority_tuple(self) -> tuple[int, int, int]:
            """Convert traits to a tuple suitable for lexicographic comparison."""
            return (
                int(self.clarified),
                int(self.decomposed),
                int(self.gap_filled),
            )

    def _compute_branch_traits(self, branch: ProcessingBranch) -> "RAGOrchestrator.BranchTraits":
        """Derive refinement traits for a given branch."""
        clarified = (
            branch.branch_type == "clarified"
            or (
                branch.parent_id is not None
                and self.branches[branch.parent_id].branch_type == "clarified"
            )
        )
        decomposed = branch.branch_type.startswith("decomposed")
        gap_filled = (
            branch.merged_results is not None
            and branch.retrieved_results is not None
            and len(branch.merged_results) > len(branch.retrieved_results)
        )
        return self.BranchTraits(clarified, decomposed, gap_filled)

    # --------------------------------------------------------------------
    def _select_best_answer(self) -> CitedAnswerResult | None:
        """Return the answer from the highest-priority completed branch.

        Priority rules (in order):
        1. Branches that result from query clarification.
        2. Branches that come from query decomposition.
        3. Branches whose evaluation step filled documentation gaps.
        """
        print("\n--- Orchestration Complete ---")
        for b_id, b in self.branches.items():
            traits = self._compute_branch_traits(b)
            print(
                f"  Branch {b_id[-8:]} ({b.branch_type}): Status={b.status.name}, "
                f"Traits={traits}"
            )

        completed_branches = [
            b for b in self.branches.values() if b.status == BranchStatus.COMPLETED and b.final_answer is not None
        ]
        if not completed_branches:
            print("No completed branch found with an answer.")
            return None

        # Pick the branch with the highest priority according to traits.
        best_branch, best_traits = max(
            ((b, self._compute_branch_traits(b)) for b in completed_branches),
            key=lambda item: item[1].to_priority_tuple(),
        )

        print(
            f"Returning answer from best branch {best_branch.branch_id[-8:]} "
            f"({best_branch.branch_type}) with traits {best_traits}"
        )
        return best_branch.final_answer

    # --------------------------------------------------------------------


# --- Example Usage ---
async def run_orchestrator(
    rag_instance: MedicalRAG, query: str, user_id: str
) -> CitedAnswerResult | None:
    orchestrator = RAGOrchestrator(rag_instance)
    return await orchestrator.process_query(query, user_id)


# Modify the main section to initialize RAG first
if __name__ == "__main__":

    async def main():
        print("Initializing Medical RAG System...")
        medical_rag_instance = await setup_medical_rag()
        print("Medical RAG System Initialized.")

        test_user_id = "orchestrator_test_user_1"
        history_path = Path(medical_rag_instance.conversation_history.save_directory) / f"{test_user_id}.json"
        if history_path.exists():
             print(f"Removing existing history file: {history_path}")
             history_path.unlink()
        medical_rag_instance.conversation_history._load_conversation(test_user_id)

        try:
            print("\n--- RUN 1: Ask about Lipitor ---")
            result1 = await run_orchestrator(
                medical_rag_instance, "what is lipitor?", test_user_id
            )
            print("\nFinal Result 1:", result1)

            print("\n" + "=" * 20 + "\n")

            print("\n--- RUN 2: Ask about side effects (uses history from Run 1) ---")
            result2 = await run_orchestrator(
                medical_rag_instance, "Tell me about its side effects.", test_user_id
            )
            print("\nFinal Result 2:", result2)

        finally:
            print("\nClosing Weaviate client...")
            await medical_rag_instance.weaviate_client.close()
            print("Weaviate client closed.")

    asyncio.run(main())
