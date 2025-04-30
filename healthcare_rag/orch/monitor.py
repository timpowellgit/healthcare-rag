"""
Query monitoring for the healthcare RAG orchestrator.

This module provides utilities for tracking and displaying query processing progress.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any

from ..models.answers import AnswerGenerationResult

logger = logging.getLogger(__name__)

class QueryMonitor:
    """Monitor for tracking query processing and displaying live updates."""
    
    def __init__(self):
        """Initialize a new query monitor."""
        self.current_step: str = "initializing"
        self.status_message: str = "Starting query processing..."
        self.steps_completed: List[str] = []
        self.raw_answer: Optional[str] = None
        self.raw_answer_event = asyncio.Event()
        self.final_answer: Optional[str] = None
        self.final_answer_event = asyncio.Event()
        self.follow_up_questions: Optional[List[str]] = None
        self.error: Optional[str] = None

    def update_status(self, step: str, message: Optional[str] = None) -> None:
        """Update the current status of the query processing."""
        self.current_step = step
        if message:
            self.status_message = message
        else:
            self.status_message = f"Processing: {step.replace('_', ' ').title()}..."
        
        self.steps_completed.append(step)
        logger.debug(f"QueryMonitor: {self.status_message}")

    def update_from_workflow_state(self, state: Dict[str, Any]) -> None:
        """Update monitor state from a workflow state dictionary."""
        if 'current_step' in state:
            self.current_step = state['current_step']
        
        if 'raw_answer' in state and state['raw_answer']:
            self.raw_answer = state['raw_answer']
            self.raw_answer_event.set()
        
        if 'final_answer' in state and state['final_answer']:
            self.final_answer = state['final_answer']
            self.final_answer_event.set()
        
        if 'follow_up_questions' in state:
            self.follow_up_questions = state['follow_up_questions']

    def on_workflow_completed(self, state: Dict[str, Any]) -> None:
        """
        Called when the workflow is completed.
        
        Args:
            state: Final workflow state
        """
        self.update_from_workflow_state(state)
        self.final_answer = state.get("validated_answer", state.get("generation_result", {}).get("plain_answer", None))
        follow_ups = state.get("follow_ups", None)
        if follow_ups and hasattr(follow_ups, "questions"):
            self.follow_up_questions = follow_ups.questions
        self.final_answer_event.set()

    def on_generation_complete(self, result: Any) -> None:
        """
        Called when answer generation is complete.
        
        Args:
            result: The generation result
        """
        if hasattr(result, "plain_answer"):
            self.raw_answer = result.plain_answer
            self.raw_answer_event.set()

    def on_branch_completed(self, branch_id: str, answer: str) -> None:
        """
        Called when a branch completes with a validated answer.
        
        Args:
            branch_id: ID of the branch that completed
            answer: The validated answer string
        """
        logger.info(f"Branch {branch_id} completed with a valid answer")
        if not self.final_answer:
            self.final_answer = answer
            self.final_answer_event.set()

    def on_answer_task_completed(self, branch_id: str, result: AnswerGenerationResult) -> None:
        """
        Called when an answer generation task completes.
        
        Args:
            branch_id: ID of the branch
            result: The generation result
        """
        if result and hasattr(result, "plain_answer") and result.plain_answer:
            # Only update if we haven't set a raw answer yet
            if not self.raw_answer:
                self.raw_answer = result.plain_answer
                self.raw_answer_event.set()
                
    async def display_progress(self) -> None:
        """Display a live progress indicator for the query processing."""
        spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        
        steps_display = {
            "initializing": "Initializing...",
            "retrieving": "Retrieving relevant information...",
            "processing": "Processing query...",
            "generating": "Generating answer...",
            "validating": "Validating and structuring answer...",
            "decomposing": "Breaking question into parts...",
            "clarifying": "Clarifying ambiguous terms...",
            "refining": "Refining the answer...",
            "completed": "Completed!",
        }
        
        last_message = ""
        
        while not self.final_answer_event.is_set():
            # Get the current step message or use a default message
            step_key = next((k for k in steps_display if k in self.current_step.lower()), None)
            display_msg = steps_display.get(step_key, self.status_message) if step_key else self.status_message
            
            # Only update if the message changed
            if display_msg != last_message:
                print(f"\r\033[K{display_msg}", end="", flush=True)
                last_message = display_msg
            else:
                # Just update the spinner
                spinner = spinner_chars[i % len(spinner_chars)]
                print(f"\r\033[K{display_msg} {spinner}", end="", flush=True)
            
            i += 1
            await asyncio.sleep(0.1)
        
        # Clear the line when done
        print("\r\033[K", end="", flush=True) 