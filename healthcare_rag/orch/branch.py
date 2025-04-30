"""
Branch management for the healthcare RAG orchestrator.

This module defines the data structures for tracking the status of query processing branches.
"""

import asyncio
import enum
import logging
import uuid
from typing import Dict, List, Optional, Any

from ..models.retrieval import QueryResultList
from ..models.answers import AnswerGenerationResult, CitedAnswerResult

logger = logging.getLogger("MedicalRAG")

class BranchStatus(enum.Enum):
    """Status of a processing branch in the RAG orchestrator."""
    ACTIVE = 1
    SUPERSEDED = 2  # Replaced by a more refined branch (e.g., clarified)
    COMPLETED = 3  # Successfully produced an answer
    FAILED = 4  # An error occurred


class ProcessingBranch:
    """
    Tracks a single branch of query processing in the RAG system.
    
    A branch represents a specific query formulation (original, clarified, or decomposed)
    and manages the tasks, status, and results associated with that query.
    """

    def __init__(self, query: str, branch_type: str, parent_id: Optional[str] = None):
        """
        Initialize a new processing branch.
        
        Args:
            query: The query text for this branch
            branch_type: Type of branch (e.g., 'initial', 'clarified', 'decomposed_0')
            parent_id: Optional ID of the parent branch (if this is a refinement)
        """
        self.branch_id: str = str(uuid.uuid4())
        self.query: str = query
        self.branch_type: str = branch_type
        self.parent_id: Optional[str] = parent_id
        self.status: BranchStatus = BranchStatus.ACTIVE
        self.tasks: Dict[str, asyncio.Task] = {}
        self.retrieved_results: Optional[QueryResultList] = None
        self.merged_results: Optional[QueryResultList] = None
        self.raw_answer: Optional[AnswerGenerationResult] = None
        self.validated_answer_str: Optional[str] = None
        self.structured_validated_answer: Optional[CitedAnswerResult] = None
        logger.info(
            f"BRANCH CREATED: {self.branch_id} (Type: {self.branch_type}, Query: '{self.query}')"
        )

    def add_task(self, task_name: str, task: asyncio.Task) -> None:
        """
        Add a task to this branch if the branch is active.
        
        Args:
            task_name: Name of the task (e.g., 'clarify', 'retrieve')
            task: The asyncio.Task to track
        """
        if self.status == BranchStatus.ACTIVE:
            self.tasks[task_name] = task
            logger.debug(f"  TASK ADDED to {self.branch_id}: {task_name}")
        else:
            logger.warning(
                f"  TASK IGNORED (Branch {self.branch_id} status: {self.status.name}): {task_name}"
            )
            task.cancel()

    def cancel_task(self, task_name: str, active_tasks: Dict[asyncio.Task, str]) -> None:
        """
        Cancel a specific task in this branch.
        
        Args:
            task_name: The name of the task to cancel
            active_tasks: Dict mapping tasks to branch IDs to update
        """
        if task_name in self.tasks:
            task = self.tasks.pop(task_name)
            if not task.done():
                task.cancel()
                logger.debug(f"  TASK CANCELLED in {self.branch_id}: {task_name}")
            if task in active_tasks:
                del active_tasks[task]

    def cancel_all_tasks(
        self, active_tasks: Dict[asyncio.Task, str]
    ) -> List[asyncio.Task]:
        """
        Cancel all tasks in this branch.
        
        Args:
            active_tasks: Dict mapping tasks to branch IDs to update
            
        Returns:
            List of cancelled tasks
        """
        logger.info(f"BRANCH CANCEL ALL: {self.branch_id}")
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
    ) -> tuple[Optional[str], Optional[asyncio.Task]]:
        """
        Find the name and task object matching the completed task.
        
        Args:
            task: The task to look up
            
        Returns:
            Tuple of (task_name, task) if found, else (None, None)
        """
        for name, t in self.tasks.items():
            if t is task:
                return name, t
        return None, None 