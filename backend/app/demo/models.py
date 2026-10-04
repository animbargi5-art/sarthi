"""
SĀRTHI V3-11 — Visual Demonstration & Telemetry Logging Data Models.
Defines structured schemas for judge-readable event streams, demo session modes,
system statuses, and audit records.

Strict Invariants:
1. SarthiDecisionEngine remains the sole physical authority.
2. AI models (Nemotron, Laya) are explicitly labeled: "AI PROPOSAL — NON-AUTHORITATIVE".
3. Decision Engine is explicitly labeled: "DETERMINISTIC PHYSICAL AUTHORITY".
4. Physical execution is explicitly labeled: "PHYSICAL EXECUTION — VALIDATED ACTION ONLY".
5. Replay mode must never claim to be live execution.
6. UI / dashboard layer cannot directly command robot motors or bypass safety.
7. Telemetry must be 100% secret-free.
"""

from __future__ import annotations

from enum import Enum
import re
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DemoMode(str, Enum):
    """Execution mode of the demonstration layer."""
    LIVE = "LIVE"
    REPLAY = "REPLAY"


class DemoEventType(str, Enum):
    """Categorical types for judge-readable event streams."""
    TASK_RECEIVED = "TASK_RECEIVED"
    NEMOTRON_TASK_UNDERSTANDING = "NEMOTRON_TASK_UNDERSTANDING"
    WORLD_STATE_OBSERVED = "WORLD_STATE_OBSERVED"
    LAYA_DECISION = "LAYA_DECISION"
    CANDIDATE_INJECTED = "CANDIDATE_INJECTED"
    DETERMINISTIC_VALIDATION_ACCEPTED = "DETERMINISTIC_VALIDATION_ACCEPTED"
    DETERMINISTIC_VALIDATION_REJECTED = "DETERMINISTIC_VALIDATION_REJECTED"
    ACTION_EXECUTED = "ACTION_EXECUTED"
    ACTION_VERIFIED = "ACTION_VERIFIED"
    DISTURBANCE_DETECTED = "DISTURBANCE_DETECTED"
    MOVE_REJECTED_BLOCKED_PATH = "MOVE_REJECTED_BLOCKED_PATH"
    RECOVERY_OPTIONS_GENERATED = "RECOVERY_OPTIONS_GENERATED"
    LAYA_SELECTED_REPOSITION = "LAYA_SELECTED_REPOSITION"
    REPOSITION_VALIDATED = "REPOSITION_VALIDATED"
    REPOSITION_EXECUTED = "REPOSITION_EXECUTED"
    MOVE_VALIDATED = "MOVE_VALIDATED"
    MOVE_EXECUTED = "MOVE_EXECUTED"
    RELEASE_VERIFIED = "RELEASE_VERIFIED"
    FINAL_PLACEMENT_VERIFIED = "FINAL_PLACEMENT_VERIFIED"
    TASK_COMPLETED = "TASK_COMPLETED"
    FAILURE_FALLBACK_ENGAGED = "FAILURE_FALLBACK_ENGAGED"
    SAFE_STOP_EXECUTED = "SAFE_STOP_EXECUTED"


def _sanitize_data(data: Any) -> Any:
    """Recursively redacts API keys, bearer tokens, and secrets."""
    if isinstance(data, str):
        sanitized = data
        for pattern in ["sk-", "nbf_", "Bearer ", "API_KEY", "SECRET", "TOKEN", "PASSWORD"]:
            if pattern.lower() in sanitized.lower():
                sanitized = re.sub(
                    rf"{re.escape(pattern)}[A-Za-z0-9_\-\.]*",
                    "[REDACTED]",
                    sanitized,
                    flags=re.IGNORECASE,
                )
        return sanitized
    elif isinstance(data, dict):
        scrubbed = {}
        for k, v in data.items():
            if k == "secret_sanitization_verified":
                scrubbed[k] = bool(v)
            elif any(s in k.lower() for s in ["api_key", "secret_key", "access_token", "bearer", "password", "auth_token", "private_key"]):
                scrubbed[k] = "[REDACTED]"
            elif k.lower() in {"key", "secret", "token", "password", "auth"}:
                scrubbed[k] = "[REDACTED]"
            else:
                scrubbed[k] = _sanitize_data(v)
        return scrubbed
    elif isinstance(data, (list, tuple)):
        return [_sanitize_data(item) for item in data]
    return data


class DemoEvent(BaseModel):
    """Single discrete step in the judge-readable demonstration stream."""
    event_index: int
    event_type: DemoEventType
    code: str  # e.g. "[01] TASK_RECEIVED"
    timestamp_iso: str
    label: str
    authority_classification: str
    details: Dict[str, Any] = Field(default_factory=dict)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        return _sanitize_data(self.model_dump())


class SystemStatus(BaseModel):
    """Current readiness and connectivity state of pipeline subsystems."""
    nemotron_status: str = "Connected"
    nemotron_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    laya_status: str = "Connected"
    laya_model: str = "laya-rl-agent (typed-decisions)"
    decision_engine_status: str = "Active (Sole Physical Authority)"
    mujoco_status: str = "Running (Franka Emika Panda 7-DOF)"
    safety_validation_status: str = "Active (ConstraintValidator Authoritative)"


class RecoveryVisualization(BaseModel):
    """Detailed trace of PATH_BLOCKED disturbance and elevated recovery."""
    disturbance_name: str = "PATH_BLOCKED"
    obstacle_id: str = "blocking_barrier_01"
    direct_move_status: str = "REJECTED"
    rejection_reason: str = "BlockedPath: Obstacle intersects trajectory"
    available_recovery_options: List[str] = Field(default_factory=lambda: ["REPOSITION", "STOP"])
    laya_selected_option: str = "REPOSITION"
    laya_authority_label: str = "AI PROPOSAL — NON-AUTHORITATIVE"
    validation_status: str = "ACCEPTED"
    validation_authority_label: str = "DETERMINISTIC PHYSICAL AUTHORITY"
    execution_result: str = "RECOVERY SUCCESS (Elevated Clearance Transit)"


class DemoSessionTelemetry(BaseModel):
    """Top-level serializable record for a demonstration session."""
    session_id: str
    demo_mode: DemoMode
    mode_label: str  # "LIVE DEMONSTRATION" or "REPLAY MODE"
    timestamp_iso: str
    human_command: str
    system_status: SystemStatus = Field(default_factory=SystemStatus)
    events: List[DemoEvent] = Field(default_factory=list)
    recovery_visualization: Optional[RecoveryVisualization] = None
    failure_fallback_demonstration: Optional[Dict[str, Any]] = None
    final_placement: Dict[str, Any] = Field(default_factory=dict)
    task_completed: bool = False
    secret_sanitization_verified: bool = True
    architectural_guarantee: str = (
        "SarthiDecisionEngine is the sole physical authority. "
        "AI models (Nemotron, Laya) propose non-authoritatively. "
        "No AI model directly commands MuJoCo or robot actuators."
    )

    def to_sanitized_dict(self) -> Dict[str, Any]:
        return _sanitize_data(self.model_dump())
