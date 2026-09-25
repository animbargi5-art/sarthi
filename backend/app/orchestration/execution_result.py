"""
SĀRTHI Orchestration — Task Execution Result.
Structured, serializable summary of a complete run_instruction() lifecycle:
  natural language → TaskUnderstanding → Decision Engine → Simulation → Verification.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from backend.app.model.models import TaskUnderstanding
from backend.app.orchestration.loop import TaskRunStatus, TaskStepResult


class VerificationSummary(BaseModel):
    """Per-step verification record."""
    model_config = ConfigDict(frozen=True)

    step_number: int = Field(..., description="Step counter (1-indexed)")
    action_type: str = Field(..., description="Action primitive executed")
    verified: bool = Field(..., description="Whether verification passed")


class TaskExecutionResult(BaseModel):
    """
    Comprehensive, serializable record of a run_instruction() execution.

    Covers every phase:
      - Semantic parsing (TaskUnderstanding)
      - Target resolution
      - Decision Engine selection per step
      - Simulation execution
      - Verification outcomes
      - Recovery tracking
      - Terminal status and failure diagnosis
    """
    model_config = ConfigDict(frozen=True)

    # Task identity
    task_id: str = Field(..., description="Unique task identifier (from TaskUnderstanding)")
    instruction: str = Field(..., description="Original natural-language instruction")

    # Semantic layer
    task_understanding: Optional[TaskUnderstanding] = Field(
        default=None,
        description="Parsed TaskUnderstanding; None on resolution failure before parsing"
    )

    # Completion flag
    completed: bool = Field(..., description="True if task reached COMPLETED status")

    # State versioning
    final_world_state_version: Union[int, str] = Field(
        ..., description="WorldState version at task termination"
    )

    # Action trace
    executed_actions: List[str] = Field(
        default_factory=list,
        description="Ordered list of action type strings actually executed"
    )
    failed_actions: List[str] = Field(
        default_factory=list,
        description="Action type strings that failed (constraint violations, collisions, etc.)"
    )

    # Recovery tracking
    recovery_count: int = Field(
        default=0,
        ge=0,
        description="Number of replanning / recovery steps taken (e.g., REPOSITION after PATH_BLOCKED)"
    )

    # Verification
    verification_results: List[VerificationSummary] = Field(
        default_factory=list,
        description="Per-step verification outcomes"
    )

    # Failure diagnosis
    failure_reason: Optional[str] = Field(
        default=None,
        description="Structured failure reason code or message; None on success"
    )

    # Extended metadata
    run_status: Optional[str] = Field(
        default=None,
        description="Terminal TaskRunStatus value for cross-layer diagnostics"
    )

    def model_dump_json_safe(self) -> Dict[str, Any]:
        """Returns a JSON-serializable dict (nested Pydantic models flattened)."""
        d = self.model_dump()
        # TaskUnderstanding nested model → plain dict already handled by model_dump()
        return d
