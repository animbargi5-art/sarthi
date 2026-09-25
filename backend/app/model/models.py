"""
SĀRTHI Model Integration — Structured Task Understanding Data Models.
Enforces type-safe, validated representations of natural language instructions.
Never allows raw model output to directly actuate robot hardware.
"""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class TaskUnderstanding(BaseModel):
    """
    Structured semantic interpretation of a human instruction.
    Represents goal decomposition, entities, constraints, and success conditions
    produced by a foundation model (e.g., NVIDIA Nemotron).
    """
    model_config = ConfigDict(frozen=True)

    task_id: str = Field(..., description="Unique identifier for the parsed task")
    objective: str = Field(..., description="Normalized high-level mission objective")
    target_object: Optional[str] = Field(default=None, description="Identified manipulable target entity ID or label")
    target_location: Optional[str] = Field(default=None, description="Identified destination zone ID or label")
    required_actions: List[str] = Field(
        default_factory=list,
        description="High-level symbolic action sequence expected for the task"
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Identified operational, kinematic, or environmental constraints"
    )
    success_conditions: List[str] = Field(
        default_factory=list,
        description="Verifiable physical conditions denoting task completion"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Self-assessed model confidence score in [0.0, 1.0]"
    )
    reasoning_summary: str = Field(
        ...,
        description="Structured summary of model deliberation and entity grounding"
    )
