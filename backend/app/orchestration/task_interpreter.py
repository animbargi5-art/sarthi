"""
SĀRTHI Orchestration — Task Interpreter.
Bridges natural-language instructions to validated TaskUnderstanding instances
via TaskUnderstandingService.  The interpreter is purely cognitive: it never
executes physical actions, never modifies WorldState, and never calls
SimulationAdapter methods.
"""

from typing import Any, Dict, Optional

from backend.app.model.models import TaskUnderstanding
from backend.app.model.task_understanding import TaskUnderstandingService


class TaskInterpreter:
    """
    Gateway that converts a raw natural-language instruction into a validated
    TaskUnderstanding.

    Responsibilities:
      - Delegate to TaskUnderstandingService for semantic parsing.
      - Surface any Pydantic validation failures before they reach the
        orchestration or decision-engine layers.
      - Enforce the architectural rule that model-layer output is NEVER
        executed directly; it is always mediated by the Decision Engine.

    Non-responsibilities (strict boundary):
      - Must NOT call execute_action().
      - Must NOT read or write SimulationAdapter state.
      - Must NOT modify WorldState.
      - Must NOT select or apply physical action primitives.
    """

    def __init__(self, task_understanding_service: TaskUnderstandingService) -> None:
        if not isinstance(task_understanding_service, TaskUnderstandingService):
            raise TypeError(
                f"Expected TaskUnderstandingService instance, "
                f"got {type(task_understanding_service).__name__}"
            )
        self._service = task_understanding_service

    @property
    def service(self) -> TaskUnderstandingService:
        """Read-only access to the underlying TaskUnderstandingService."""
        return self._service

    def interpret(
        self,
        instruction: str,
        world_context: Optional[Dict[str, Any]] = None,
    ) -> TaskUnderstanding:
        """
        Parse a natural-language instruction into a validated TaskUnderstanding.

        Args:
            instruction: Human-readable task description.
            world_context: Optional snapshot of workspace metadata passed to
                           the underlying model provider for contextual grounding.

        Returns:
            Fully validated, frozen TaskUnderstanding instance.

        Raises:
            ValueError: If instruction is empty or whitespace-only.
            TypeError:  If provider returns an unsupported output type.
            pydantic.ValidationError: If provider output fails schema validation.
        """
        # Delegate entirely to service — no physical logic here
        return self._service.understand(
            instruction=instruction,
            world_context=world_context,
        )
