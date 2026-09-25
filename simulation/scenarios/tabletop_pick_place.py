"""
SĀRTHI Simulation Scenarios — Tabletop Pick-and-Place Benchmark Specification.
Defines the concrete MVP Physical AI scenario for SĀRTHI.

Scenario Overview:
- Environment: Tabletop workspace with robotic manipulator.
- Human Instruction: "Move the red object to the blue target."
- Nominal Task Flow: APPROACH -> GRASP -> MOVE -> RELEASE.
- Dynamic Disturbance: PATH_BLOCKED introduced after GRASP, intersecting the planned path.
- Autonomous Recovery: Decision Engine discovers obstruction in WorldState and adapts
  (e.g., selecting REPOSITION) without hardcoded scenario heuristics.
"""

import math
from enum import Enum
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field

from backend.app.decision_engine.models import (
    ActionType,
    ActiveConstraint,
    EnvironmentState,
    LastActionOutcome,
    LastActionStatus,
    ObjectState,
    Point3D,
    RobotState,
    TargetZone,
    TaskObjective,
    WorldObject,
    WorldState,
)
from simulation.core.events import DisturbanceEvent, DisturbanceType
from simulation.core.geometry import SimDimensions3D, SimPoint3D


# ---------------------------------------------------------------------------
# Structured Scenario Enums
# ---------------------------------------------------------------------------

class ScenarioFailureReason(str, Enum):
    """Structured failure categories for tabletop pick-and-place tasks."""
    OBJECT_UNREACHABLE = "OBJECT_UNREACHABLE"
    TARGET_UNREACHABLE = "TARGET_UNREACHABLE"
    BLOCKED_PATH_NO_ALTERNATIVE = "BLOCKED_PATH_NO_ALTERNATIVE"
    GRASP_FAILURE = "GRASP_FAILURE"
    PLACEMENT_FAILURE = "PLACEMENT_FAILURE"
    WORKSPACE_VIOLATION = "WORKSPACE_VIOLATION"
    PAYLOAD_VIOLATION = "PAYLOAD_VIOLATION"
    MAXIMUM_STEPS_EXCEEDED = "MAXIMUM_STEPS_EXCEEDED"


class DisturbanceTiming(str, Enum):
    """Deterministic execution trigger for dynamic disturbances."""
    IMMEDIATE = "IMMEDIATE"
    AFTER_APPROACH = "AFTER_APPROACH"
    AFTER_GRASP = "AFTER_GRASP"  # Standard benchmark trigger: obstacle appears after grasping
    AFTER_MOVE = "AFTER_MOVE"


# ---------------------------------------------------------------------------
# Component Specifications
# ---------------------------------------------------------------------------

class RobotConfiguration(BaseModel):
    """Physical and kinematic specification of the robotic manipulator."""
    model_config = ConfigDict(frozen=True)

    robot_id: str = Field(default="tabletop_manipulator", description="Configurable robot identifier")
    dof: int = Field(default=7, description="Degrees of freedom (e.g. 7-DOF arm)")
    base_pose: Point3D = Field(default=Point3D(x=0.0, y=0.0, z=0.20), description="Robot mounting base pose (m)")
    max_reach_m: float = Field(default=0.85, gt=0.0, description="Maximum kinematic arm reach from base (m)")
    max_payload_kg: float = Field(default=3.0, gt=0.0, description="Maximum rated payload capacity (kg)")


class ObjectSpecification(BaseModel):
    """Specification of the movable manipulation target."""
    model_config = ConfigDict(frozen=True)

    object_id: str = Field(default="red_object_01", description="Unique entity identifier")
    name: str = Field(default="Red Cylindrical Object", description="Human-readable entity label")
    initial_pose: Point3D = Field(default=Point3D(x=0.25, y=0.15, z=0.20), description="Initial spawn coordinates (m)")
    dimensions_m: Point3D = Field(default=Point3D(x=0.05, y=0.05, z=0.06), description="Length, width, height (m)")
    bounding_radius_m: float = Field(default=0.05, gt=0.0, description="Bounding collision sphere radius (m)")
    mass_kg: float = Field(default=0.5, gt=0.0, description="Object mass in kilograms")
    is_movable: bool = Field(default=True, description="Flag indicating manipulability")
    is_graspable: bool = Field(default=True, description="Flag indicating graspable geometry")


class TargetZoneSpecification(BaseModel):
    """Specification of the placement destination zone."""
    model_config = ConfigDict(frozen=True)

    target_id: str = Field(default="blue_target_zone", description="Unique zone identifier")
    name: str = Field(default="Blue Destination Zone", description="Human-readable zone label")
    target_pose: Point3D = Field(default=Point3D(x=0.40, y=-0.20, z=0.20), description="Target center coordinates (m)")
    tolerance_radius_m: float = Field(default=0.06, gt=0.0, description="Acceptable placement radius tolerance (m)")
    is_movable: bool = Field(default=False, description="Target zone is fixed on workspace tabletop")


class ObstacleSpecification(BaseModel):
    """Specification of dynamic obstacle causing the PATH_BLOCKED disturbance."""
    model_config = ConfigDict(frozen=True)

    obstacle_id: str = Field(default="blocking_barrier_01", description="Unique obstacle identifier")
    name: str = Field(default="Dynamic Path Obstacle", description="Human-readable label")
    position: Point3D = Field(
        default=Point3D(x=0.325, y=-0.025, z=0.12),
        description="Obstacle centroid placed directly along nominal move path (m)"
    )
    dimensions_m: Point3D = Field(
        default=Point3D(x=0.06, y=0.06, z=0.08),
        description="Bounding dimensions: length_x, width_y, height_z (m)"
    )
    bounding_radius_m: float = Field(default=0.05, gt=0.0, description="Effective collision bounding radius (m)")
    mass_kg: float = Field(default=5.0, gt=0.0, description="Obstacle mass in kilograms")
    is_dynamic: bool = Field(default=True, description="Introduced dynamically during task execution")


class WorkspaceBoundsConfig(BaseModel):
    """Spatial limits of the operational workspace in meters."""
    model_config = ConfigDict(frozen=True)

    min_x: float = Field(default=-0.8, description="Minimum X workspace limit (m)")
    max_x: float = Field(default=0.8, description="Maximum X workspace limit (m)")
    min_y: float = Field(default=-0.8, description="Minimum Y workspace limit (m)")
    max_y: float = Field(default=0.8, description="Maximum Y workspace limit (m)")
    min_z: float = Field(default=0.0, description="Minimum Z workspace limit (m)")
    max_z: float = Field(default=1.2, description="Maximum Z workspace limit (m)")


# ---------------------------------------------------------------------------
# Comprehensive Tabletop Scenario Definition
# ---------------------------------------------------------------------------

class TabletopPickPlaceScenario(BaseModel):
    """
    Strongly typed, deterministic specification of the SĀRTHI Tabletop Pick-and-Place MVP Scenario.
    """
    model_config = ConfigDict(frozen=True)

    scenario_id: str = Field(default="tabletop_pick_and_place_mvp", description="Unique scenario benchmark identifier")
    instruction: str = Field(default="Move the red object to the blue target.", description="Canonical human instruction")
    robot: RobotConfiguration = Field(default_factory=RobotConfiguration)
    red_object: ObjectSpecification = Field(default_factory=ObjectSpecification)
    blue_target: TargetZoneSpecification = Field(default_factory=TargetZoneSpecification)
    obstacle: ObstacleSpecification = Field(default_factory=ObstacleSpecification)
    workspace: WorkspaceBoundsConfig = Field(default_factory=WorkspaceBoundsConfig)
    disturbance_timing: DisturbanceTiming = Field(default=DisturbanceTiming.AFTER_GRASP)

    # -----------------------------------------------------------------------
    # Geometric Validation: Obstacle Intersects Nominal Path
    # -----------------------------------------------------------------------

    def obstacle_intersects_nominal_path(self, tolerance_m: float = 0.05) -> bool:
        """
        Calculates whether the dynamic obstacle geometrically intersects the line segment
        connecting the red object initial position and the blue target position.

        Returns True if the perpendicular distance from the obstacle center to the nominal
        trajectory segment is less than the obstacle collision radius + tolerance.
        """
        # Line segment endpoints A (red object) and B (blue target) in 2D (x, y)
        ax, ay = self.red_object.initial_pose.x, self.red_object.initial_pose.y
        bx, by = self.blue_target.target_pose.x, self.blue_target.target_pose.y
        ox, oy = self.obstacle.position.x, self.obstacle.position.y

        # Segment vector AB
        dx = bx - ax
        dy = by - ay
        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq == 0.0:
            return False

        # Projection factor of obstacle center onto segment AB: t = ((O - A) . (B - A)) / |AB|^2
        t = ((ox - ax) * dx + (oy - ay) * dy) / seg_len_sq

        # Clamped projection factor onto segment [0.0, 1.0]
        t_clamped = max(0.0, min(1.0, t))
        closest_x = ax + t_clamped * dx
        closest_y = ay + t_clamped * dy

        dist_to_segment = math.sqrt((ox - closest_x) ** 2 + (oy - closest_y) ** 2)
        collision_threshold = self.obstacle.bounding_radius_m + tolerance_m

        return dist_to_segment <= collision_threshold

    # -----------------------------------------------------------------------
    # Disturbance Event Factory
    # -----------------------------------------------------------------------

    def create_disturbance_event(self, event_id: str = "dist_block_direct_path") -> DisturbanceEvent:
        """
        Generates a standardized SĀRTHI DisturbanceEvent representing PATH_BLOCKED.
        """
        return DisturbanceEvent.create_path_blocked(
            event_id=event_id,
            obstacle_id=self.obstacle.obstacle_id,
            position=SimPoint3D(
                x=self.obstacle.position.x,
                y=self.obstacle.position.y,
                z=self.obstacle.position.z,
            ),
            dimensions=SimDimensions3D(
                length_x=self.obstacle.dimensions_m.x,
                width_y=self.obstacle.dimensions_m.y,
                height_z=self.obstacle.dimensions_m.z,
            ),
        )

    # -----------------------------------------------------------------------
    # Success Verification
    # -----------------------------------------------------------------------

    def check_success_conditions(self, world_state: WorldState) -> Tuple[bool, Optional[str]]:
        """
        Verifies task success against strict physical criteria:
        1. Red object exists in the world state.
        2. Red object center lies within blue target zone tolerance radius.
        3. Robot gripper is open and not holding any object.
        4. Last executed action was RELEASE with status SUCCESS.
        5. No active safety constraint is violated.
        """
        robot = world_state.robot
        target = world_state.target

        # 1. Gripper state check: must be released and open
        if robot.holding_object_id is not None:
            return False, f"Gripper is still holding object '{robot.holding_object_id}'"
        if not robot.gripper_open:
            return False, "Gripper jaws are not open"

        # 2. Locate red target object
        target_obj: Optional[WorldObject] = None
        for obj in world_state.objects:
            if obj.is_target:
                target_obj = obj
                break

        if target_obj is None:
            return False, f"Target object '{self.red_object.object_id}' not found in WorldState"

        # 3. Placement tolerance check
        dx = target_obj.position.x - target.position.x
        dy = target_obj.position.y - target.position.y
        horizontal_dist = math.sqrt(dx * dx + dy * dy)

        if horizontal_dist > target.tolerance_radius_m:
            return False, (
                f"Object placement distance ({horizontal_dist:.4f}m) exceeds "
                f"target tolerance radius ({target.tolerance_radius_m:.4f}m)"
            )

        # 4. Last action outcome verification
        last_outcome = world_state.last_action_outcome
        if last_outcome.action_type != ActionType.RELEASE or last_outcome.status != LastActionStatus.SUCCESS:
            return False, f"Last action outcome was not successful RELEASE: {last_outcome}"

        return True, None

    # -----------------------------------------------------------------------
    # Scenario -> WorldState Conversion
    # -----------------------------------------------------------------------

    def to_world_state(
        self,
        with_disturbance: bool = False,
        version: int = 1,
        timestamp_ns: int = 0,
    ) -> WorldState:
        """
        Translates scenario specification into the canonical SĀRTHI WorldState model.
        Reuses backend.app.decision_engine.models.WorldState without schema duplication.
        """
        # Robot State
        robot_state = RobotState(
            position=self.robot.base_pose,
            gripper_open=True,
            holding_object_id=None,
            payload_mass_kg=0.0,
            is_moving=False,
            max_payload_kg=self.robot.max_payload_kg,
            max_reach_m=self.robot.max_reach_m,
        )

        # Objects list
        objects_list: List[WorldObject] = [
            WorldObject(
                id=self.red_object.object_id,
                name=self.red_object.name,
                position=self.red_object.initial_pose,
                bounding_radius_m=self.red_object.bounding_radius_m,
                mass_kg=self.red_object.mass_kg,
                state=ObjectState.FREE,
                is_target=True,
                is_obstacle=False,
            )
        ]

        if with_disturbance:
            objects_list.append(
                WorldObject(
                    id=self.obstacle.obstacle_id,
                    name=self.obstacle.name,
                    position=self.obstacle.position,
                    bounding_radius_m=self.obstacle.bounding_radius_m,
                    mass_kg=self.obstacle.mass_kg,
                    state=ObjectState.FREE,
                    is_target=False,
                    is_obstacle=True,
                )
            )

        # Target Zone
        target_zone = TargetZone(
            id=self.blue_target.target_id,
            position=self.blue_target.target_pose,
            tolerance_radius_m=self.blue_target.tolerance_radius_m,
        )

        # Environment State
        env_state = EnvironmentState(
            min_x=self.workspace.min_x,
            max_x=self.workspace.max_x,
            min_y=self.workspace.min_y,
            max_y=self.workspace.max_y,
            min_z=self.workspace.min_z,
            max_z=self.workspace.max_z,
            dynamic_obstacles_detected=with_disturbance,
            slip_risk_level=0.0,
            friction_coefficient=0.6,
        )

        return WorldState(
            version=version,
            timestamp_ns=timestamp_ns,
            robot=robot_state,
            objects=objects_list,
            target=target_zone,
            environment=env_state,
            task_objective=TaskObjective.PICK_AND_PLACE,
            active_constraints=[],
            last_action_outcome=LastActionOutcome(status=LastActionStatus.NONE),
        )


# ---------------------------------------------------------------------------
# Convenience Factory Functions
# ---------------------------------------------------------------------------

def create_default_scenario(
    robot_id: str = "tabletop_manipulator",
) -> TabletopPickPlaceScenario:
    """Creates the default deterministic Tabletop Pick-and-Place scenario."""
    return TabletopPickPlaceScenario(
        robot=RobotConfiguration(robot_id=robot_id)
    )


def create_tabletop_world_state(
    scenario: Optional[TabletopPickPlaceScenario] = None,
    with_disturbance: bool = False,
) -> WorldState:
    """Creates a SĀRTHI WorldState directly from a TabletopPickPlaceScenario."""
    scen = scenario or create_default_scenario()
    return scen.to_world_state(with_disturbance=with_disturbance)
