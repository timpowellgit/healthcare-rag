"""
Task scheduling and management utilities for the healthcare RAG orchestrator.

This module provides utilities for launching, tracking, and cancelling asyncio tasks.
"""

import asyncio
import logging
from typing import Dict, List, Set, Any, Callable, Optional, Coroutine, TypeVar, Tuple

from .branch import ProcessingBranch

logger = logging.getLogger("MedicalRAG")

# Type variable for task results
T = TypeVar('T')

def launch_task(
    branch: ProcessingBranch,
    task_name: str,
    coro: Coroutine,
    active_tasks: Dict[asyncio.Task, str]
) -> Optional[asyncio.Task]:
    """
    Launch a task for a branch and track it in the active tasks dictionary.
    
    Args:
        branch: The ProcessingBranch to associate with the task
        task_name: Name of the task for logging
        coro: The coroutine to run as a task
        active_tasks: Dictionary mapping tasks to branch IDs
        
    Returns:
        The created task if branch is active, None otherwise
    """
    if branch.status != branch.status.ACTIVE:
        logger.warning(
            f"Skipping task '{task_name}' launch for inactive branch {branch.branch_id}"
        )
        return None
    
    task = asyncio.create_task(coro)
    branch.add_task(task_name, task)
    active_tasks[task] = branch.branch_id
    return task

async def wait_for_first_completed(
    tasks: List[asyncio.Task],
) -> Tuple[Set[asyncio.Task], Set[asyncio.Task]]:
    """
    Wait for the first task in the list to complete.
    
    Args:
        tasks: List of tasks to wait for
        
    Returns:
        Tuple of (done_tasks, pending_tasks)
    """
    if not tasks:
        return set(), set()
    
    done, pending = await asyncio.wait(
        tasks, return_when=asyncio.FIRST_COMPLETED
    )
    return done, pending

async def process_task_result(
    task: asyncio.Task,
    branch_id: str,
    branches: Dict[str, ProcessingBranch],
    active_tasks: Dict[asyncio.Task, str],
    result_handler: Callable[[str, Any, ProcessingBranch], Coroutine]
) -> None:
    """
    Process the result of a completed task.
    
    Args:
        task: The completed task
        branch_id: ID of the branch the task belongs to
        branches: Dictionary of all branches
        active_tasks: Dictionary mapping tasks to branch IDs
        result_handler: Callback to handle the task result
    """
    if not branch_id or branch_id not in branches:
        logger.warning(f"Ignoring task result: Task {task} has no associated active branch.")
        return

    branch = branches[branch_id]
    if branch.status != branch.status.ACTIVE:
        logger.info(
            f"Ignoring task result: Branch {branch_id} is not ACTIVE (Status: {branch.status.name})."
        )
        return

    task_name, _ = branch.get_task_details(task)
    if not task_name:
        logger.warning(
            f"WARNING: Completed task {task} not found in branch {branch_id}.tasks"
        )
        return

    if task_name in branch.tasks:
        del branch.tasks[task_name]
    else:
        logger.warning(f"WARNING: Task '{task_name}' already removed from branch {branch_id}.tasks")

    logger.info(
        f"Processing result for Task '{task_name}' from Branch {branch.branch_id} ({branch.branch_type})"
    )

    try:
        result = await task
        await result_handler(task_name, result, branch)

    except asyncio.CancelledError:
        logger.warning(f"Task '{task_name}' in branch {branch_id} was cancelled.")
        if task_name == "validate" and branch.status == branch.status.ACTIVE:
            logger.warning(f"  Branch {branch_id} cannot complete due to cancelled validation.")

    except Exception as e:
        logger.error(f"!!! Task '{task_name}' in branch {branch_id} failed: {e}", exc_info=True)
        branch.status = branch.status.FAILED
        branch.cancel_all_tasks(active_tasks)

def get_active_branch_tasks(
    active_tasks: Dict[asyncio.Task, str], 
    branches: Dict[str, ProcessingBranch]
) -> List[asyncio.Task]:
    """
    Get all tasks associated with active branches.
    
    Args:
        active_tasks: Dictionary mapping tasks to branch IDs
        branches: Dictionary of all branches
        
    Returns:
        List of tasks belonging to active branches
    """
    return [
        task
        for task, branch_id in active_tasks.items()
        if branch_id in branches
        and branches[branch_id].status == branches[branch_id].status.ACTIVE
        and not task.done()
    ]

def supersede_branch(
    branch_id: str,
    branches: Dict[str, ProcessingBranch],
    active_tasks: Dict[asyncio.Task, str]
) -> None:
    """
    Mark a branch as superseded and cancel all its tasks.
    
    Args:
        branch_id: ID of the branch to supersede
        branches: Dictionary of all branches
        active_tasks: Dictionary mapping tasks to branch IDs
    """
    if branch_id in branches:
        branch = branches[branch_id]
        if branch.status == branch.status.ACTIVE:
            logger.info(f"BRANCH SUPERSEDED: {branch_id} (Type: {branch.branch_type})")
            branch.status = branch.status.SUPERSEDED
            cancelled_tasks = branch.cancel_all_tasks(active_tasks)
            if cancelled_tasks:
                asyncio.ensure_future(
                    asyncio.gather(*cancelled_tasks, return_exceptions=True)
                ) 