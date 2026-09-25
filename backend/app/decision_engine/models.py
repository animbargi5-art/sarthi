"""
SĀRTHI Decision Engine — Core Data Models and State Representations.
Deterministic, type-safe schemas utilizing Pydantic v2.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class Point3D(BaseModel):
    """3D Cartesian coordinate in world frame (meters)."""
    model_config = ConfigDict(frozen=True)

    x: float = Field(..., description="X coordinate (m)")
    y: float = Field(..., description="Y coordinate (m)")
    z: float = Field(..., description="Z coordinate (m)")

    def distance_to(self, other: Point3D) -> float:
        """Euclidean distance between two 3D points."""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )


class ActionType(str, Enum):
    """Supported candidate action primitives for robotic decision-making."""
    APPROACH = "APPROACH"
    REPOSITION = "REPOSITION"
    GRASP = "GRASP"
    MOVE = "MOVE"
    RELEASE = "RELEASE"
    STOP = "STOP"


class LastActionStatus(str, Enum):
    """Outcome status of the previous action execution."""
    NONE = "NONE"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    INTERRUPTED = "INTERRUPTED"


class ObjectState(str, Enum):
    """Physical state of an object in the world."""
    FREE = "FREE"
    GRASPED = "GRASPED"
    OBSTRUCTED = "OBSTRUCTED"
    PLACED = "PLACED"
    DROPPED = "DROPPED"


class TaskObjective(str, Enum):
    """Primary high-level task goal."""
    PICK_AND_PLACE = "PICK_AND_PLACE"
    CLEAR_OBSTACLE = "CLEAR_OBSTACLE"
    HOLD_POSITION = "HOLD_POSITION"
    INSPECT = "INSPECT"


class RobotState(BaseModel):
    """Current state of the robotic manipulator."""
    model_config = ConfigDict(frozen=True)

    position: Point3D = Field(..., description="Current end-effector position (m)")
    gripper_open: bool = Field(..., description="True if gripper jaws are open")
    holding_object_id: Optional[str] = Field(default=None, description="ID of currently held object, if any")
    payload_mass_kg: float = Field(default=0.0, ge=0.0, description="Current carried payload mass (kg)")
    is_moving: bool = Field(default=False, description="True if joints are currently in motion")
    max_payload_kg: float = Field(default=3.0, gt=0.0, description="Maximum rated payload capacity (kg)")
    max_reach_m: float = Field(default=0.85, gt=0.0, description="Maximum kinematic arm reach from base (m)")


class WorldObject(BaseModel):
    """Physical entity in the workspace."""
    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Unique object identifier")
    name: str = Field(..., description="Human-readable object name")
    position: Point3D = Field(..., description="Object center of mass in world frame (m)")
    bounding_radius_m: float = Field(default=0.05, gt=0.0, description="Collision sphere radius (m)")
    mass_kg: float = Field(default=0.5, gt=0.0, description="Object mass (kg)")
    state: ObjectState = Field(default=ObjectState.FREE, description="Current physical state")
    is_target: bool = Field(default=False, description="True if this object is the manipulation target")
    is_obstacle: bool = Field(default=False, description="True if this object acts as an obstacle")


class TargetZone(BaseModel):
    """Target destination or staging area for task completion."""
    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Target zone identifier")
    position: Point3D = Field(..., description="Target center coordinates (m)")
    tolerance_radius_m: float = Field(default=0.05, gt=0.0, description="Acceptable placement radius (m)")


class EnvironmentState(BaseModel):
    """Global environmental parameters and boundary conditions."""
    model_config = ConfigDict(frozen=True)

    min_x: float = Field(default=-0.8, description="Minimum workspace X boundary (m)")
    max_x: float = Field(default=0.8, description="Maximum workspace X boundary (m)")
    min_y: float = Field(default=-0.8, description="Minimum workspace Y boundary (m)")
    max_y: float = Field(default=0.8, description="Maximum workspace Y boundary (m)")
    min_z: float = Field(default=0.0, description="Minimum workspace Z boundary (m)")
    max_z: float = Field(default=1.2, description="Maximum workspace Z boundary (m)")
    dynamic_obstacles_detected: bool = Field(default=False, description="Flag for moving obstacles")
    slip_risk_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Estimated slip risk (0.0 to 1.0)")
    friction_coefficient: float = Field(default=0.6, gt=0.0, description="Surface friction estimate")


class ActiveConstraint(BaseModel):
    """Operational or safety constraint actively imposed on the system."""
    model_config = ConfigDict(frozen=True)

    constraint_id: str = Field(..., description="Constraint identifier")
    description: str = Field(..., description="Description of the constraint")
    max_force_newtons: Optional[float] = Field(default=None, description="Maximum allowable contact force (N)")
    max_speed_mps: Optional[float] = Field(default=None, description="Maximum allowable velocity (m/s)")
    keep_out_center: Optional[Point3D] = Field(default=None, description="Center of keep-out exclusion zone")
    keep_out_radius: Optional[float] = Field(default=None, description="Radius of keep-out exclusion zone (m)")
    required_clearance_m: Optional[float] = Field(default=0.03, description="Minimum clearance around obstacles (m)")


class LastActionOutcome(BaseModel):
    """Telemetry report describing the outcome of the most recent action."""
    model_config = ConfigDict(frozen=True)

    action_type: Optional[ActionType] = Field(default=None, description="Action type executed")
    status: LastActionStatus = Field(default=LastActionStatus.NONE, description="Execution status")
    error_message: Optional[str] = Field(default=None, description="Diagnostic error or deviation reason")
    contact_force_delta: Optional[float] = Field(default=0.0, description="Observed contact force delta (N)")


class WorldState(BaseModel):
    """
    Complete structured snapshot of the world, robot, task, and constraints.
    Immutable, deterministic representation for decision-making.
    """
    model_config = ConfigDict(frozen=True)

    version: Union[int, str] = Field(default=1, description="Monotonically increasing state version identifier")
    timestamp_ns: int = Field(default=0, description="State capture timestamp in nanoseconds")
    robot: RobotState = Field(..., description="Current robot state")
    objects: List[WorldObject] = Field(default_factory=list, description="All objects detected in workspace")
    target: TargetZone = Field(..., description="Primary target area")
    environment: EnvironmentState = Field(default_factory=EnvironmentState, description="Environment conditions")
    task_objective: TaskObjective = Field(default=TaskObjective.PICK_AND_PLACE, description="Current mission objective")
    active_constraints: List[ActiveConstraint] = Field(default_factory=list, description="Active safety constraints")
    last_action_outcome: LastActionOutcome = Field(default_factory=LastActionOutcome, description="Previous action outcome")


class CandidateAction(BaseModel):
    """
    Structured action proposal submitted to the decision engine.
    Must never control motors directly; defines symbolic and kinematic targets.
    """
    model_config = ConfigDict(frozen=True)

    action_id: str = Field(..., description="Unique action identifier")
    action_type: ActionType = Field(..., description="Action primitive type")
    target_object_id: Optional[str] = Field(default=None, description="Object acted upon, if applicable")
    target_position: Optional[Point3D] = Field(default=None, description="Target spatial coordinates (m)")
    speed_scale: float = Field(default=1.0, ge=0.0, le=1.0, description="Speed scaling factor (0.0 to 1.0)")
    expected_force_n: float = Field(default=5.0, ge=0.0, description="Expected contact/clamping force (N)")
    parameters: Dict[str, Union[str, float, int, bool]] = Field(
        default_factory=dict,
        description="Auxiliary parameters for lower-level controller"
    )


class EvaluationFactor(BaseModel):
    """Numerical factor contribution to a candidate evaluation."""
    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Factor identifier")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized factor score [0.0, 1.0]")
    weight: float = Field(..., ge=0.0, le=1.0, description="Relative weight of this factor")
    description: str = Field(..., description="Objective description of factor scoring")


class CandidateEvaluation(BaseModel):
    """Detailed deterministic assessment of a single CandidateAction."""
    model_config = ConfigDict(frozen=True)

    action: CandidateAction = Field(..., description="The evaluated action")
    is_valid: bool = Field(..., description="True if action satisfies all hard constraints")
    rejection_reasons: List[str] = Field(default_factory=list, description="Reasons for rejection if invalid")
    responsibility_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Safety and stewardship score")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Goal alignment score")
    consequence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Physical stability outcome score")
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Composite weighted ranking score")
    factors: List[EvaluationFactor] = Field(default_factory=list, description="Itemized factor breakdown")


class Decision(BaseModel):
    """
    Structured outcome of the SĀRTHI Decision Engine.
    Fully auditable, deterministic, and free of unstructured thought traces.
    """
    model_config = ConfigDict(frozen=True)

    decision_id: str = Field(..., description="Unique decision UUID or identifier")
    world_state_version: Union[int, str] = Field(..., description="Version of WorldState evaluated")
    selected_action: CandidateAction = Field(..., description="Chosen action to execute")
    candidate_evaluations: List[CandidateEvaluation] = Field(..., description="Evaluations of all evaluated candidates")
    rejection_reasons: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Rejection rationale mapped by action_id"
    )
    decision_factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Key numerical decision factors for the chosen action"
    )
    timestamp_ns: int = Field(default=0, description="Timestamp of decision compilation in nanoseconds")
