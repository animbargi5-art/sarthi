"""
SĀRTHI V3-10 — Failure & Resiliency Fallback Data Models.
Defines structured schemas for failure taxonomy, detection stages,
fallback strategies, and auditable resiliency records.

Strict Invariants:
1. SarthiDecisionEngine remains the sole physical authority.
2. AI models (Nemotron, Laya) propose or interpret; software validates.
3. Every fallback action must pass deterministic physical validation.
4. No failure may create an AI -> robot direct bypass.
5. All records must be 100% secret-free.
"""

from __future__ import annotations

from enum import Enum
import re
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FailureType(str, Enum):
    """Categorical classification of failures across cognitive, decision, and physical layers."""
    LAYA_TIMEOUT = "LAYA_TIMEOUT"
    LAYA_CONNECTION_FAILURE = "LAYA_CONNECTION_FAILURE"
    MALFORMED_LAYA_RESPONSE = "MALFORMED_LAYA_RESPONSE"
    LAYA_OUT_OF_BOUNDS = "LAYA_OUT_OF_BOUNDS"
    NEMOTRON_TIMEOUT = "NEMOTRON_TIMEOUT"
    NEMOTRON_MALFORMED = "NEMOTRON_MALFORMED"
    DETERMINISTIC_VALIDATION_REJECTION = "DETERMINISTIC_VALIDATION_REJECTION"
    IK_FAILURE = "IK_FAILURE"
    PHYSICS_EXECUTION_FAILURE = "PHYSICS_EXECUTION_FAILURE"
    VERIFICATION_FAILURE = "VERIFICATION_FAILURE"
    OBJECT_GRASP_FAILURE = "OBJECT_GRASP_FAILURE"
    PATH_BLOCKED_DISTURBANCE = "PATH_BLOCKED_DISTURBANCE"
    MULTI_FAILURE_CASCADE = "MULTI_FAILURE_CASCADE"
    NONE = "NONE"


class DetectionStage(str, Enum):
    """Stage in the cyber-physical pipeline where the failure or disturbance is detected."""
    TASK_UNDERSTANDING = "TASK_UNDERSTANDING"
    TARGET_RESOLUTION = "TARGET_RESOLUTION"
    CONTEXT_SYNTHESIS = "CONTEXT_SYNTHESIS"
    DECISION_PROPOSAL = "DECISION_PROPOSAL"
    CANDIDATE_INJECTION = "CANDIDATE_INJECTION"
    DETERMINISTIC_VALIDATION = "DETERMINISTIC_VALIDATION"
    IK_SOLVER = "IK_SOLVER"
    PHYSICAL_EXECUTION = "PHYSICAL_EXECUTION"
    VERIFICATION = "VERIFICATION"
    STATE_OBSERVATION = "STATE_OBSERVATION"


class FallbackStrategy(str, Enum):
    """Explicit strategy selected by the fallback policy."""
    PROPOSAL_ACCEPTED = "PROPOSAL_ACCEPTED"          # Tier 1: Valid AI proposal passed validation
    DETERMINISTIC_RECOVERY = "DETERMINISTIC_RECOVERY"  # Tier 2: Validated deterministic recovery candidate
    SAFE_STOP = "SAFE_STOP"                            # Tier 3: Zero motion / joint hold
    SAFE_ABORT = "SAFE_ABORT"                          # Cognitive abort before physical motion


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
            if any(s in k.lower() for s in ["key", "secret", "token", "password", "auth"]):
                scrubbed[k] = "[REDACTED]"
            else:
                scrubbed[k] = _sanitize_data(v)
        return scrubbed
    elif isinstance(data, (list, tuple)):
        return [_sanitize_data(item) for item in data]
    return data


class ResiliencyRecord(BaseModel):
    """
    Audit record capturing a controlled failure injection scenario,
    its detection stage, fallback selection, validation, and final outcome.
    """
    failure_id: str
    failure_type: FailureType
    source_component: str
    detected_at_stage: DetectionStage
    recovery_strategy: FallbackStrategy
    fallback_action: Optional[str] = None
    validation_result: str  # "accepted", "rejected", "fallback"
    execution_result: Optional[Dict[str, Any]] = None
    verification_result: Optional[bool] = None
    final_status: str  # "COMPLETED", "STOPPED_SAFELY", "FAILED_SAFELY"
    rejection_reason: List[str] = Field(default_factory=list)
    safe_stop: bool = False
    provenance: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp_ns: int = Field(default_factory=time.time_ns)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """Returns clean, secret-free dictionary."""
        return _sanitize_data(self.model_dump())


class V310ResiliencyReport(BaseModel):
    """
    Top-level telemetry and audit report for the V3-10 Failure & Resiliency Suite.
    """
    report_id: str
    timestamp_iso: str
    total_scenarios_evaluated: int
    total_passed: int
    total_failed: int
    unsafe_executions_count: int = 0  # MUST be 0
    ai_to_robot_bypasses_count: int = 0  # MUST be 0
    fallback_hierarchy_description: List[str] = Field(default_factory=list)
    scenarios: List[ResiliencyRecord] = Field(default_factory=list)
    live_validation_summary: Dict[str, Any] = Field(default_factory=dict)
    secret_sanitization_verified: bool = True
    known_limitations: List[str] = Field(default_factory=list)
    resiliency_conclusion: str = ""

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """Returns clean, secret-free dictionary."""
        return _sanitize_data(self.model_dump())
