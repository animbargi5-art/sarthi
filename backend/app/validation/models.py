"""
SĀRTHI Physics Validation Models — Schema & Result Definitions.

Defines structured data structures for the 12 physics validation dimensions:
1. timestep sensitivity
2. joint limits
3. joint velocities
4. end-effector repeatability
5. contact stability
6. placement repeatability
7. collision / clearance
8. gravity / settling
9. IK convergence
10. numerical stability
11. deterministic replay
12. physics regression

Strict Architectural Guarantees:
- Completely separate from decision-making authority.
- Teleportation-free and secret-free.
- Structured, deterministic, and JSON-serializable.
"""

from datetime import datetime, timezone
from enum import Enum
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ValidationDimension(str, Enum):
    """The 12 canonical physics validation dimensions defined in V3 architecture."""
    TIMESTEP_SENSITIVITY = "timestep_sensitivity"
    JOINT_LIMITS = "joint_limits"
    JOINT_VELOCITIES = "joint_velocities"
    END_EFFECTOR_REPEATABILITY = "end_effector_repeatability"
    CONTACT_STABILITY = "contact_stability"
    PLACEMENT_REPEATABILITY = "placement_repeatability"
    COLLISION_CLEARANCE = "collision_clearance"
    GRAVITY_SETTLING = "gravity_settling"
    IK_CONVERGENCE = "ik_convergence"
    NUMERICAL_STABILITY = "numerical_stability"
    DETERMINISTIC_REPLAY = "deterministic_replay"
    PHYSICS_REGRESSION = "physics_regression"


def sanitize_secrets(data: Any) -> Any:
    """Recursively scrubs secrets, bearer tokens, and API keys from arbitrary data."""
    if isinstance(data, dict):
        cleaned = {}
        for k, v in data.items():
            k_lower = str(k).lower()
            if any(term in k_lower for term in ["key", "token", "secret", "bearer", "password", "auth"]):
                cleaned[k] = "[REDACTED]"
            else:
                cleaned[k] = sanitize_secrets(v)
        return cleaned
    elif isinstance(data, list):
        return [sanitize_secrets(item) for item in data]
    elif isinstance(data, str):
        if re.search(r"(Bearer\s+[A-Za-z0-9_\-\.]+)|(sk-[A-Za-z0-9_\-]+)", data, re.IGNORECASE):
            return "[REDACTED_SECRET]"
        return data
    return data


class PhysicsValidationResult(BaseModel):
    """
    Structured result for a single physics validation dimension.
    """
    model_config = ConfigDict(frozen=True)

    validation_id: str = Field(..., description="Unique validation execution ID")
    dimension: str = Field(..., description="Validation dimension name")
    scenario: str = Field(default="tabletop_pick_and_place_mvp", description="Evaluated scenario")
    parameter_configuration: Dict[str, Any] = Field(default_factory=dict, description="Configured parameters")
    run_count: int = Field(default=1, ge=1, description="Number of execution repetitions")
    pass_fail: bool = Field(..., description="True if validation passed all expected bounds")
    measured_values: Dict[str, Any] = Field(default_factory=dict, description="Raw and aggregated measurements")
    expected_bounds: Dict[str, Any] = Field(default_factory=dict, description="Configured tolerance limits")
    failure_reasons: List[str] = Field(default_factory=list, description="Descriptions of any boundary failures")
    reproducibility_information: Dict[str, Any] = Field(default_factory=dict, description="Seed and env info")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Sanitized, secret-free metadata")

    @property
    def passed(self) -> bool:
        return self.pass_fail

    def to_dict(self) -> Dict[str, Any]:
        """Returns secret-free dictionary representation."""
        raw = self.model_dump()
        return sanitize_secrets(raw)


class PhysicsValidationSuiteReport(BaseModel):
    """
    Comprehensive suite report aggregating all 12 physics validation dimensions.
    """
    model_config = ConfigDict(frozen=True)

    suite_id: str = Field(..., description="Suite execution identifier")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    environment: Dict[str, Any] = Field(default_factory=dict)
    robot_model: str = Field(default="Franka Emika Panda (7-DOF)")
    mujoco_version: str = Field(default="unknown")
    scenario_id: str = Field(default="tabletop_pick_and_place_mvp")
    total_validations: int = Field(default=12)
    passed_validations: int = Field(default=0)
    failed_validations: int = Field(default=0)
    all_passed: bool = Field(default=False)
    results: Dict[str, PhysicsValidationResult] = Field(default_factory=dict)
    known_limitations: List[str] = Field(default_factory=list)
    regression_conclusion: str = Field(default="")

    def to_dict(self) -> Dict[str, Any]:
        """Returns clean secret-free dictionary representation."""
        raw = self.model_dump()
        return sanitize_secrets(raw)
