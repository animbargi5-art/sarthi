"""
SĀRTHI Model Integration — Task Understanding Service.
Exposes high-level semantic task understanding backed by strict Pydantic validation.
Isolates cognitive foundation models from physical execution layers.
"""

from typing import Any, Dict, Optional
from pydantic import ValidationError
from backend.app.model.models import TaskUnderstanding
from backend.app.model.provider import ModelProvider


class TaskUnderstandingService:
    """
    Service gateway for extracting structured TaskUnderstanding from natural language instructions.
    Enforces that all model outputs pass Pydantic schema validation before reaching
    the orchestration or decision engine layers.
    """

    def __init__(self, provider: ModelProvider):
        if not isinstance(provider, ModelProvider):
            raise TypeError(f"Expected ModelProvider instance, got {type(provider).__name__}")
        self._provider = provider

    @property
    def provider(self) -> ModelProvider:
        """Read-only access to configured provider adapter."""
        return self._provider

    def understand(
        self,
        instruction: str,
        world_context: Optional[Dict[str, Any]] = None,
    ) -> TaskUnderstanding:
        """
        Interprets a human instruction using the underlying ModelProvider.
        Validates the output through Pydantic to ensure all semantic invariants hold.
        """
        if not instruction or not instruction.strip():
            raise ValueError("Instruction string must be non-empty.")

        # Query provider
        raw_output = self._provider.understand_task(
            instruction=instruction.strip(),
            world_context=world_context,
        )

        # Enforce Pydantic validation regardless of how provider generated the result
        if isinstance(raw_output, TaskUnderstanding):
            # Validate model integrity (re-validates constraints, bounds, types)
            validated = TaskUnderstanding.model_validate(raw_output.model_dump())
        elif isinstance(raw_output, dict):
            validated = TaskUnderstanding.model_validate(raw_output)
        else:
            raise TypeError(
                f"ModelProvider returned invalid type: {type(raw_output).__name__}. "
                f"Expected TaskUnderstanding or dict."
            )

        return validated
