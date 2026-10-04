"""
SĀRTHI V3 — Core Interface & Data Model Specifications.
Enforces type-safe, validated representations for Human Instructions, Physical Situations,
Compact Decision Contexts, Bounded Decision Questions, Jev Decisions, Latency Metrics,
Physics Validation Results, Recovery Events, and Telemetry Events.

Non-Negotiable Invariants:
1. AI models propose or select bounded semantic candidates; deterministic software validates
   physical feasibility; only validated actions reach the robot controller.
2. Jev must never bypass deterministic safety validation.
3. JevDecision is strictly non-authoritative and does not authorize physical execution.
"""

from __future__ import annotations

from enum import Enum
import time
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.decision_engine.models import (
    ActiveConstraint,
    CandidateAction,
    Decision,
    LastActionOutcome,
    Point3D,
    RobotState,
    TargetZone,
    WorldObject,
    WorldState,
)


class HumanInstruction(BaseModel):
    """
    Represents the original natural-language command issued by a human operator.
    Does not duplicate TaskUnderstanding (which represents foundation model parsing).
    """
    model_config = ConfigDict(frozen=True)

    instruction: str = Field(..., min_length=1, description="Natural-language instruction text")
    task_id: str = Field(..., min_length=1, description="Unique task identifier associated with this command")
    timestamp_ns: int = Field(default=0, description="Epoch timestamp in nanoseconds when command was received")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional request metadata")


class PhysicalSituation(BaseModel):
    """
    Represents the current physical situation relevant to decision making.
    References and reuses canonical SĀRTHI state structures without duplication.
    """
    model_config = ConfigDict(frozen=True)

    situation_id: str = Field(..., min_length=1, description="Unique snapshot identifier for this physical situation")
    world_state: WorldState = Field(..., description="Canonical WorldState from physical simulation or sensors")
    active_constraints: List[ActiveConstraint] = Field(default_factory=list, description="Active physical constraints")
    last_action_outcome: LastActionOutcome = Field(default_factory=LastActionOutcome, description="Outcome of preceding action")
    telemetry_summary: Dict[str, Any] = Field(default_factory=dict, description="Filtered telemetry metadata")
    task_context: Dict[str, Any] = Field(default_factory=dict, description="Contextual annotations (e.g. current phase)")
    timestamp_ns: int = Field(default=0, description="Timestamp of physical snapshot in nanoseconds")


class DecisionContext(BaseModel):
    """
    A COMPACT context assembled specifically for a bounded decision query.
    Must NOT contain raw simulation pointers or uncurated state dumps.
    Contains only decision-relevant parameters to ensure low-latency serialization.
    """
    model_config = ConfigDict(frozen=True)

    context_id: str = Field(..., min_length=1, description="Unique identifier for this decision context")
    task_objective: str = Field(..., description="High-level task objective (e.g. 'PICK_AND_PLACE')")
    robot_state_summary: Dict[str, Any] = Field(..., description="Compact summary: position, gripper state, payload")
    target_object_summary: Optional[Dict[str, Any]] = Field(default=None, description="Target object identifier, position, state")
    destination_summary: Optional[Dict[str, Any]] = Field(default=None, description="Target destination coordinates and tolerance")
    active_constraints: List[str] = Field(default_factory=list, description="Identifiers and summaries of active constraints")
    previous_action_outcome: Optional[str] = Field(default=None, description="Outcome description of previous action")
    candidate_actions: List[str] = Field(default_factory=list, description="Available discrete candidate action primitives")
    disturbance_info: Optional[Dict[str, Any]] = Field(default=None, description="Disturbance description if an anomaly occurred")
    timestamp_ns: int = Field(default=0, description="Context assembly timestamp in nanoseconds")

    @classmethod
    def from_situation(
        cls,
        context_id: str,
        situation: PhysicalSituation,
        candidate_actions: List[str],
        task_objective: Optional[str] = None,
        disturbance_info: Optional[Dict[str, Any]] = None,
    ) -> DecisionContext:
        """Constructs a clean, compact DecisionContext from a full PhysicalSituation."""
        ws = situation.world_state
        robot_summary = {
            "position": [round(ws.robot.position.x, 3), round(ws.robot.position.y, 3), round(ws.robot.position.z, 3)],
            "gripper_open": ws.robot.gripper_open,
            "holding_object_id": ws.robot.holding_object_id,
            "payload_mass_kg": ws.robot.payload_mass_kg,
        }

        target_obj = next((obj for obj in ws.objects if obj.is_target), None)
        target_summary = None
        if target_obj:
            target_summary = {
                "id": target_obj.id,
                "name": target_obj.name,
                "position": [round(target_obj.position.x, 3), round(target_obj.position.y, 3), round(target_obj.position.z, 3)],
                "state": target_obj.state.value if hasattr(target_obj.state, "value") else str(target_obj.state),
            }

        dest_summary = {
            "id": ws.target.id,
            "position": [round(ws.target.position.x, 3), round(ws.target.position.y, 3), round(ws.target.position.z, 3)],
            "tolerance_radius_m": ws.target.tolerance_radius_m,
        }

        constraint_summaries = [c.description for c in situation.active_constraints or ws.active_constraints]
        prev_outcome = (
            situation.last_action_outcome.status.value
            if hasattr(situation.last_action_outcome.status, "value")
            else str(situation.last_action_outcome.status)
        )

        return cls(
            context_id=context_id,
            task_objective=task_objective or (ws.task_objective.value if hasattr(ws.task_objective, "value") else str(ws.task_objective)),
            robot_state_summary=robot_summary,
            target_object_summary=target_summary,
            destination_summary=dest_summary,
            active_constraints=constraint_summaries,
            previous_action_outcome=prev_outcome,
            candidate_actions=candidate_actions,
            disturbance_info=disturbance_info,
            timestamp_ns=situation.timestamp_ns or time.time_ns(),
        )


class DecisionQuestionType(str, Enum):
    """Discrete categories of bounded questions that may be queried to a fast decision layer."""
    ACTION_SELECTION = "ACTION_SELECTION"
    RECOVERY_SELECTION = "RECOVERY_SELECTION"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
    TARGET_SELECTION = "TARGET_SELECTION"


class DecisionQuestion(BaseModel):
    """
    A bounded, structured decision query.
    Must contain discrete, enumerated options; does NOT accept arbitrary free-form execution commands.
    """
    model_config = ConfigDict(frozen=True)

    question_id: str = Field(..., min_length=1, description="Unique identifier for this bounded question")
    question_type: DecisionQuestionType = Field(..., description="Category of bounded decision query")
    question_text: str = Field(..., min_length=1, description="Focused question prompt")
    available_options: List[str] = Field(..., description="Bounded discrete options to choose from")
    context: DecisionContext = Field(..., description="Compact context for evaluating the question")

    @field_validator("available_options")
    @classmethod
    def validate_options(cls, v: List[str]) -> List[str]:
        if not v or len(v) < 2:
            raise ValueError("available_options must contain at least 2 distinct choices")
        if len(set(v)) != len(v):
            raise ValueError("available_options must not contain duplicate choices")
        return v


class JevDecision(BaseModel):
    """
    A structured, non-authoritative candidate recommendation produced by the fast decision layer.
    CRITICAL: JevDecision is NOT an authoritative Decision and cannot directly authorize physical execution.
    It must be passed as or mapped into a CandidateAction and evaluated by the SĀRTHI Decision Engine.
    """
    model_config = ConfigDict(frozen=True)

    selected_option: str = Field(..., min_length=1, description="Recommended candidate option from available choices")
    option_probabilities: Dict[str, float] = Field(default_factory=dict, description="Probability or score distribution over options")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Self-assessed confidence score [0.0, 1.0]")
    answer_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Model answer-specific confidence score [0.0, 1.0]")
    rationale: Optional[str] = Field(default=None, description="Brief structured rationale or explanation")
    provider: str = Field(default="jev", description="Decision provider identifier")
    model_version: Optional[str] = Field(default=None, description="Model identifier or version string")
    latency_ms: Optional[float] = Field(default=None, ge=0.0, description="Decision query roundtrip latency in milliseconds")
    timestamp_ns: int = Field(default=0, description="Timestamp in nanoseconds")
    request_id: Optional[str] = Field(default=None, description="Unique correlation request identifier")


class DecisionLatencyRecord(BaseModel):
    """
    Instruments granular latency across all decision, validation, and execution stages.
    Supports optional/unavailable values to avoid fabricating measurements.
    """
    model_config = ConfigDict(frozen=True)

    cycle_id: str = Field(..., min_length=1, description="Unique decision cycle identifier")
    tau_nemotron: Optional[float] = Field(default=None, ge=0.0, description="Task understanding cloud LLM latency in ms")
    tau_context: Optional[float] = Field(default=None, ge=0.0, description="Decision context synthesis latency in ms")
    tau_jev: Optional[float] = Field(default=None, ge=0.0, description="Jev fast bounded decision query latency in ms")
    tau_validation: Optional[float] = Field(default=None, ge=0.0, description="Deterministic Decision Engine constraint evaluation latency in ms")
    tau_ik: Optional[float] = Field(default=None, ge=0.0, description="DLS numerical IK computation latency in ms")
    tau_sim: Optional[float] = Field(default=None, ge=0.0, description="MuJoCo physics stepping duration in ms")
    tau_verify: Optional[float] = Field(default=None, ge=0.0, description="Physical state reader verification latency in ms")
    total_cycle_ms: Optional[float] = Field(default=None, ge=0.0, description="Total elapsed cycle time in ms")
    timestamp_ns: int = Field(default=0, description="Epoch timestamp in nanoseconds")


class PhysicsValidationResult(BaseModel):
    """
    Schema for recording outcomes from the 12-point Physics Validation Framework.
    """
    model_config = ConfigDict(frozen=True)

    validation_id: str = Field(..., min_length=1, description="Unique validation check identifier (e.g. PV-01)")
    test_name: str = Field(..., min_length=1, description="Descriptive name of test dimension (e.g. Timestep Sensitivity)")
    category: str = Field(..., min_length=1, description="Validation category (e.g. KINEMATICS, DYNAMICS, REPEATABILITY)")
    passed: bool = Field(..., description="Binary pass/fail verdict")
    measured_metrics: Dict[str, float] = Field(default_factory=dict, description="Empirical values measured during simulation")
    configured_thresholds: Dict[str, float] = Field(default_factory=dict, description="Validation tolerances or threshold limits")
    violations: List[str] = Field(default_factory=list, description="Descriptions of any detected violations")
    seed_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Simulation seed and reproducibility parameters")
    timestamp_ns: int = Field(default=0, description="Timestamp of validation run in nanoseconds")


class RecoveryEvent(BaseModel):
    """
    Structured audit record documenting a physical disturbance, constraint rejection,
    and subsequent autonomous recovery selection.
    """
    model_config = ConfigDict(frozen=True)

    event_id: str = Field(..., min_length=1, description="Unique recovery event UUID")
    disturbance_type: str = Field(..., min_length=1, description="Classified disturbance type, e.g. 'PATH_BLOCKED'")
    rejected_action: str = Field(..., min_length=1, description="Identifier or name of rejected candidate action")
    selected_recovery: str = Field(..., min_length=1, description="Selected recovery primitive, e.g. 'REPOSITION'")
    reason: str = Field(..., min_length=1, description="Diagnostic justification for rejection and recovery selection")
    before_state_version: Optional[Union[int, str]] = Field(default=None, description="WorldState version prior to recovery")
    after_state_version: Optional[Union[int, str]] = Field(default=None, description="WorldState version following recovery")
    verification_result: Optional[str] = Field(default=None, description="Verification outcome of recovery action")
    recovery_count: int = Field(default=1, ge=1, description="Cumulative count of recoveries executed during task")
    timestamp_ns: int = Field(default=0, description="Timestamp of recovery event in nanoseconds")


class TelemetryEvent(BaseModel):
    """
    Generic structured telemetry event for SĀRTHI V3.
    Enforces automatic sanitization to guarantee no secrets, credentials, or API keys are ever stored.
    """
    model_config = ConfigDict(frozen=True)

    event_id: str = Field(..., min_length=1, description="Unique telemetry event UUID")
    event_type: str = Field(..., min_length=1, description="Event category: 'STATE_UPDATE', 'DECISION', 'RECOVERY', 'LATENCY'")
    timestamp_ns: int = Field(..., description="Timestamp of event in nanoseconds")
    task_id: Optional[str] = Field(default=None, description="Associated task identifier")
    stage: str = Field(..., min_length=1, description="Pipeline execution stage")
    success: bool = Field(default=True, description="True if stage completed successfully")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Sanitized event metadata")
    latency_ms: Optional[float] = Field(default=None, ge=0.0, description="Associated stage duration if applicable")

    @field_validator("metadata")
    @classmethod
    def sanitize_metadata(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures secrets and API keys are NEVER included in telemetry."""
        forbidden_patterns = ["sk-", "nbf_", "Bearer ", "API_KEY", "SECRET", "TOKEN", "PASSWORD"]
        sanitized: Dict[str, Any] = {}
        for key, val in v.items():
            key_upper = str(key).upper()
            if any(forbidden in key_upper for forbidden in ["KEY", "SECRET", "TOKEN", "PASSWORD", "AUTH"]):
                sanitized[key] = "[REDACTED]"
                continue
            val_str = str(val)
            if any(forbidden in val_str for forbidden in forbidden_patterns):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = val
        return sanitized
