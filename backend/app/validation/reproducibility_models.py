"""
SĀRTHI V3-9 Reproducibility & Repeated-Run Models.

Defines schemas and telemetry structures for the three validation levels:
- Level A: Physics-Only Replay
- Level B: Decision-Pipeline Replay
- Level C: Full Live V3 Repeatability

Strict Invariants:
- Deterministic, serializable, secret-free.
- Explicit failure classification taxonomy.
"""

from datetime import datetime, timezone
from enum import Enum
import math
import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field

from backend.app.validation.models import sanitize_secrets


class FailureClassification(str, Enum):
    """Canonical failure classification taxonomy defined for SĀRTHI V3-9."""
    MODEL_VARIABILITY = "MODEL_VARIABILITY"
    NETWORK_LATENCY = "NETWORK_LATENCY"
    CONTEXT_VARIATION = "CONTEXT_VARIATION"
    BOUNDED_DECISION_VARIATION = "BOUNDED_DECISION_VARIATION"
    DETERMINISTIC_VALIDATION_FAILURE = "DETERMINISTIC_VALIDATION_FAILURE"
    PHYSICS_FAILURE = "PHYSICS_FAILURE"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"
    VERIFICATION_FAILURE = "VERIFICATION_FAILURE"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


class LevelAPhysicsReplayResult(BaseModel):
    """Results from Level A: Physics-Only Replay across multiple runs."""
    model_config = ConfigDict(frozen=True)

    level_name: str = Field(default="LEVEL_A_PHYSICS_REPLAY")
    num_runs: int = Field(default=5)
    identical_action_sequences: bool = Field(default=True)
    identical_world_state_transitions: bool = Field(default=True)
    all_runs_succeeded: bool = Field(default=True)
    mean_placement_error_m: float = Field(default=0.0)
    max_placement_error_m: float = Field(default=0.0)
    std_placement_error_m: float = Field(default=0.0)
    max_euclidean_pose_spread_m: float = Field(default=0.0)
    run_records: List[Dict[str, Any]] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_secrets(self.model_dump())


class LevelBDecisionReplayResult(BaseModel):
    """Results from Level B: Decision-Pipeline Replay across repeated queries."""
    model_config = ConfigDict(frozen=True)

    level_name: str = Field(default="LEVEL_B_DECISION_REPLAY")
    num_runs: int = Field(default=5)
    logical_decision_agreement_rate: float = Field(default=1.0)
    deterministic_validation_agreement_rate: float = Field(default=1.0)
    candidate_injection_validity_rate: float = Field(default=1.0)
    selected_options: List[str] = Field(default_factory=list)
    option_probabilities: List[Dict[str, float]] = Field(default_factory=list)
    deterministic_decisions: List[str] = Field(default_factory=list)
    rejection_reasons: List[List[str]] = Field(default_factory=list)
    run_records: List[Dict[str, Any]] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_secrets(self.model_dump())


class LevelCLiveRunResult(BaseModel):
    """Detailed record for a single run in Level C Full Live Repeatability."""
    model_config = ConfigDict(frozen=True)

    run_id: str
    run_index: int
    task_understanding_success: bool
    task_target_object: str
    task_target_zone: str
    laya_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    candidate_actions: List[str] = Field(default_factory=list)
    deterministic_actions: List[str] = Field(default_factory=list)
    execution_results: List[Dict[str, Any]] = Field(default_factory=list)
    recovery_occurred: bool
    recovery_action: Optional[str]
    final_object_position: Tuple[float, float, float]
    final_placement_error_m: float
    placement_within_tolerance: bool
    final_task_status: str
    completed: bool
    failure_classification: FailureClassification = Field(default=FailureClassification.NONE)
    latency_summary: Dict[str, float] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_secrets(self.model_dump())


class ReproducibilityMetrics(BaseModel):
    """Aggregate repeatability metrics across all repeated runs."""
    model_config = ConfigDict(frozen=True)

    total_runs_attempted: int = Field(default=0)
    total_runs_completed: int = Field(default=0)
    task_success_rate: float = Field(default=0.0)
    recovery_success_rate: float = Field(default=0.0)
    action_sequence_match_rate: float = Field(default=0.0)
    laya_decision_agreement_rate: float = Field(default=0.0)
    deterministic_validation_agreement_rate: float = Field(default=0.0)
    final_placement_error_mean: float = Field(default=0.0)
    final_placement_error_max: float = Field(default=0.0)
    final_placement_error_std: float = Field(default=0.0)
    final_pose_spread_m: float = Field(default=0.0)
    failure_count: int = Field(default=0)
    unexpected_behavior_count: int = Field(default=0)
    failure_classification_counts: Dict[str, int] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_secrets(self.model_dump())


class V39ReproducibilityReport(BaseModel):
    """Unified comprehensive report for SĀRTHI V3-9."""
    model_config = ConfigDict(frozen=True)

    report_id: str
    timestamp_iso: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    hardware_environment: Dict[str, Any] = Field(default_factory=dict)
    level_a_physics: LevelAPhysicsReplayResult
    level_b_decision: LevelBDecisionReplayResult
    level_c_live_runs: List[LevelCLiveRunResult]
    metrics: ReproducibilityMetrics
    known_limitations: List[str] = Field(default_factory=list)
    reproducibility_conclusion: str = Field(default="")

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_secrets(self.model_dump())
