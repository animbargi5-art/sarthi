"""
SĀRTHI Simulation Core — Deterministic World State and Action Dispatcher.
Maintains physical state of robot, objects, obstacles, and environmental conditions.
Executes discrete kinematic actions and dynamic disturbance injections.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from simulation.core.events import ActionExecutionResult, DisturbanceEvent, DisturbanceType
from simulation.core.geometry import (
    SimDimensions3D,
    SimPoint3D,
    WorkspaceBounds,
    distance_point_to_line_segment,
)
from simulation.core.objects import SimulatedObject, SimulatedObstacle, SimulatedTargetZone
from simulation.core.robot import SimulatedRobot

# Integration with backend decision engine models
from backend.app.decision_engine.models import (
    ActionType,
    CandidateAction,
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


class SimulationWorld:
    """
    Deterministic local simulation world.
    Maintains exact spatial, kinematic, and topological state for robotics testing.
    Incrementing world_state_version on every physical state change guarantees auditability.
    """

    def __init__(
        self,
        robot: Optional[SimulatedRobot] = None,
        workspace_bounds: Optional[WorkspaceBounds] = None,
    ):
        self.workspace_bounds = workspace_bounds or WorkspaceBounds()
        self.robot = robot or SimulatedRobot(workspace_limits=self.workspace_bounds)
        self.objects: Dict[str, SimulatedObject] = {}
        self.obstacles: Dict[str, SimulatedObstacle] = {}
        self.target_zones: Dict[str, SimulatedTargetZone] = {}
        self.simulation_time: float = 0.0
        self.world_state_version: int = 1
        self.active_disturbances: List[DisturbanceEvent] = []
        self.execution_history: List[ActionExecutionResult] = []
        self.last_action_outcome: LastActionOutcome = LastActionOutcome(status=LastActionStatus.NONE)

    # --- Setup & Entity Management ---

    def add_object(self, obj: SimulatedObject) -> None:
        """Add an object to the simulation."""
        self.objects[obj.id] = obj
        self.world_state_version += 1

    def add_obstacle(self, obstacle: SimulatedObstacle) -> None:
        """Add a static or dynamic obstacle to the simulation."""
        self.obstacles[obstacle.id] = obstacle
        self.world_state_version += 1

    def add_target_zone(self, zone: SimulatedTargetZone) -> None:
        """Add a destination target zone."""
        self.target_zones[zone.id] = zone
        self.world_state_version += 1

    def step(self, dt: float = 0.01) -> None:
        """Advance simulation clock."""
        self.simulation_time = round(self.simulation_time + dt, 4)

    # --- Disturbance Injection ---

    def inject_disturbance(self, disturbance: DisturbanceEvent) -> bool:
        """
        Inject an environmental disturbance into the active simulation.
        The first supported primitive is PATH_BLOCKED, which adds/activates an obstacle.
        """
        if disturbance.disturbance_type == DisturbanceType.PATH_BLOCKED:
            obs_id = disturbance.parameters.get("obstacle_id", f"obs_disturb_{len(self.obstacles) + 1}")
            pos_dict = disturbance.parameters.get("position", {"x": 0.25, "y": 0.0, "z": 0.2})
            dim_dict = disturbance.parameters.get("dimensions", {"length_x": 0.08, "width_y": 0.08, "height_z": 0.20})

            pos = SimPoint3D(x=pos_dict["x"], y=pos_dict["y"], z=pos_dict["z"])
            dims = SimDimensions3D(
                length_x=dim_dict.get("length_x", 0.08),
                width_y=dim_dict.get("width_y", 0.08),
                height_z=dim_dict.get("height_z", 0.20),
            )

            obstacle = SimulatedObstacle(
                id=obs_id,
                name=f"Dynamic Obstacle ({obs_id})",
                position=pos,
                dimensions=dims,
                is_active=True,
            )
            self.obstacles[obs_id] = obstacle
            self.active_disturbances.append(disturbance)
            self.world_state_version += 1
            return True

        return False

    # --- Collision Detection Helper ---

    def _check_trajectory_collision(self, start: SimPoint3D, end: SimPoint3D, clearance: float = 0.04) -> Optional[str]:
        """
        Check if linear trajectory segment between start and end intersects any active obstacle.
        Returns obstacle_id if blocked, or None if clear.
        """
        for obs in self.obstacles.values():
            if not obs.is_active:
                continue
            dist = distance_point_to_line_segment(obs.position, start, end)
            safe_distance = obs.bounding_radius + clearance
            if dist < safe_distance:
                return obs.id
        return None

    # --- Action Execution ---

    def execute_action(self, action: Union[CandidateAction, Dict[str, Any]]) -> ActionExecutionResult:
        """
        Deterministically executes a CandidateAction in the simulation world.
        Updates world state and increments world_state_version on success.
        Preserves state and returns structured failure reasons on error.
        """
        # Normalize input to CandidateAction if dict passed
        if isinstance(action, dict):
            act_type = ActionType(action["action_type"])
            act_id = action.get("action_id", "act_001")
            tgt_pos_dict = action.get("target_position")
            tgt_pos = SimPoint3D(**tgt_pos_dict) if tgt_pos_dict else None
            tgt_obj_id = action.get("target_object_id")
        else:
            act_type = action.action_type
            act_id = action.action_id
            tgt_pos = SimPoint3D(x=action.target_position.x, y=action.target_position.y, z=action.target_position.z) if action.target_position else None
            tgt_obj_id = action.target_object_id

        prev_version = self.world_state_version
        self.step(0.05)  # advance simulation clock by 50ms per action step

        # Dispatch action primitive
        if act_type == ActionType.STOP:
            return self._handle_stop(act_id, prev_version)
        elif act_type == ActionType.APPROACH:
            return self._handle_approach(act_id, tgt_pos, tgt_obj_id, prev_version)
        elif act_type == ActionType.REPOSITION:
            return self._handle_reposition(act_id, tgt_pos, prev_version)
        elif act_type == ActionType.GRASP:
            return self._handle_grasp(act_id, tgt_obj_id, prev_version)
        elif act_type == ActionType.MOVE:
            return self._handle_move(act_id, tgt_pos, prev_version)
        elif act_type == ActionType.RELEASE:
            return self._handle_release(act_id, prev_version)
        else:
            return self._fail(act_id, str(act_type), f"Unsupported action type: {act_type}", prev_version)

    # --- Action Handlers ---

    def _handle_stop(self, action_id: str, prev_version: int) -> ActionExecutionResult:
        """Halts the robot. Always succeeds and updates state version."""
        self.world_state_version += 1
        self.last_action_outcome = LastActionOutcome(
            action_type=ActionType.STOP,
            status=LastActionStatus.SUCCESS,
            error_message=None,
        )
        result = ActionExecutionResult(
            success=True,
            action_type=ActionType.STOP.value,
            action_id=action_id,
            simulation_time=self.simulation_time,
            previous_world_state_version=prev_version,
            new_world_state_version=self.world_state_version,
            failure_reason=None,
            details={"status": "halted_at_current_pose"},
        )
        self.execution_history.append(result)
        return result

    def _handle_approach(
        self, action_id: str, target_pos: Optional[SimPoint3D], target_obj_id: Optional[str], prev_version: int
    ) -> ActionExecutionResult:
        """Moves end-effector toward target position or standoff position of an object."""
        if target_pos is None and target_obj_id is not None:
            obj = self.objects.get(target_obj_id)
            if obj:
                # Approach directly to object position
                target_pos = obj.position

        if target_pos is None:
            return self._fail(action_id, ActionType.APPROACH.value, "Missing target position for APPROACH", prev_version)

        # Bounds and Reach checks
        if not self.robot.is_in_workspace(target_pos):
            return self._fail(action_id, ActionType.APPROACH.value, "Target position outside workspace bounds", prev_version)
        if not self.robot.can_reach(target_pos):
            return self._fail(action_id, ActionType.APPROACH.value, "Target position exceeds robot max reach", prev_version)

        # Obstacle collision check
        blocked_obs = self._check_trajectory_collision(self.robot.position, target_pos)
        if blocked_obs:
            return self._fail(
                action_id,
                ActionType.APPROACH.value,
                f"BlockedPath: Obstacle '{blocked_obs}' intersects approach trajectory",
                prev_version,
            )

        # Execute movement
        self.robot.move_to(target_pos)
        self.world_state_version += 1
        self.last_action_outcome = LastActionOutcome(
            action_type=ActionType.APPROACH,
            status=LastActionStatus.SUCCESS,
        )

        result = ActionExecutionResult(
            success=True,
            action_type=ActionType.APPROACH.value,
            action_id=action_id,
            simulation_time=self.simulation_time,
            previous_world_state_version=prev_version,
            new_world_state_version=self.world_state_version,
            failure_reason=None,
            details={"new_position": {"x": target_pos.x, "y": target_pos.y, "z": target_pos.z}},
        )
        self.execution_history.append(result)
        return result

    def _handle_reposition(
        self, action_id: str, target_pos: Optional[SimPoint3D], prev_version: int
    ) -> ActionExecutionResult:
        """Repositions end-effector to a waypoint. If holding an object, carries it along."""
        if target_pos is None:
            return self._fail(action_id, ActionType.REPOSITION.value, "Missing target position for REPOSITION", prev_version)

        if not self.robot.is_in_workspace(target_pos):
            return self._fail(action_id, ActionType.REPOSITION.value, "Target position outside workspace bounds", prev_version)
        if not self.robot.can_reach(target_pos):
            return self._fail(action_id, ActionType.REPOSITION.value, "Target position exceeds robot max reach", prev_version)

        blocked_obs = self._check_trajectory_collision(self.robot.position, target_pos)
        if blocked_obs:
            return self._fail(
                action_id,
                ActionType.REPOSITION.value,
                f"BlockedPath: Obstacle '{blocked_obs}' intersects reposition trajectory",
                prev_version,
            )

        self.robot.move_to(target_pos)
        # If carrying an object, update object position
        if self.robot.carrying_object_id:
            held_obj = self.objects.get(self.robot.carrying_object_id)
            if held_obj:
                held_obj.move_to(target_pos)

        self.world_state_version += 1
        self.last_action_outcome = LastActionOutcome(
            action_type=ActionType.REPOSITION,
            status=LastActionStatus.SUCCESS,
        )

        result = ActionExecutionResult(
            success=True,
            action_type=ActionType.REPOSITION.value,
            action_id=action_id,
            simulation_time=self.simulation_time,
            previous_world_state_version=prev_version,
            new_world_state_version=self.world_state_version,
            failure_reason=None,
            details={"new_position": {"x": target_pos.x, "y": target_pos.y, "z": target_pos.z}},
        )
        self.execution_history.append(result)
        return result

    def _handle_grasp(
        self, action_id: str, target_obj_id: Optional[str], prev_version: int
    ) -> ActionExecutionResult:
        """Grasps target object if within reach, gripper open, and payload capacity satisfied."""
        if self.robot.carrying_object_id is not None:
            return self._fail(
                action_id,
                ActionType.GRASP.value,
                f"Gripper already carrying object '{self.robot.carrying_object_id}'",
                prev_version,
            )

        if not target_obj_id or target_obj_id not in self.objects:
            # Fallback: check if an object is within grasp proximity
            closest_obj = None
            min_d = 999.0
            for obj in self.objects.values():
                d = self.robot.position.distance_to(obj.position)
                if d < min_d:
                    min_d = d
                    closest_obj = obj
            if closest_obj and min_d <= 0.12:
                target_obj = closest_obj
            else:
                return self._fail(action_id, ActionType.GRASP.value, f"Target object '{target_obj_id}' not found", prev_version)
        else:
            target_obj = self.objects[target_obj_id]

        if not target_obj.is_graspable:
            return self._fail(action_id, ActionType.GRASP.value, f"Object '{target_obj.id}' is not graspable", prev_version)

        dist = self.robot.position.distance_to(target_obj.position)
        if dist > 0.12:
            return self._fail(
                action_id,
                ActionType.GRASP.value,
                f"ProximityViolation: Distance {dist:.3f}m exceeds grasp threshold (0.12m)",
                prev_version,
            )

        if target_obj.mass > self.robot.max_payload_kg:
            return self._fail(
                action_id,
                ActionType.GRASP.value,
                f"PayloadViolation: Object mass {target_obj.mass:.2f}kg exceeds robot limit {self.robot.max_payload_kg:.2f}kg",
                prev_version,
            )

        # Successful grasp
        self.robot.attach_object(target_obj.id)
        target_obj.current_holder = self.robot.robot_id
        self.world_state_version += 1
        self.last_action_outcome = LastActionOutcome(
            action_type=ActionType.GRASP,
            status=LastActionStatus.SUCCESS,
        )

        result = ActionExecutionResult(
            success=True,
            action_type=ActionType.GRASP.value,
            action_id=action_id,
            simulation_time=self.simulation_time,
            previous_world_state_version=prev_version,
            new_world_state_version=self.world_state_version,
            failure_reason=None,
            details={"grasped_object_id": target_obj.id, "mass_kg": target_obj.mass},
        )
        self.execution_history.append(result)
        return result

    def _handle_move(
        self, action_id: str, target_pos: Optional[SimPoint3D], prev_version: int
    ) -> ActionExecutionResult:
        """Moves robot and carried payload to target destination."""
        if target_pos is None:
            return self._fail(action_id, ActionType.MOVE.value, "Missing target position for MOVE", prev_version)

        if not self.robot.is_in_workspace(target_pos):
            return self._fail(action_id, ActionType.MOVE.value, "Target position outside workspace bounds", prev_version)
        if not self.robot.can_reach(target_pos):
            return self._fail(action_id, ActionType.MOVE.value, "Target position exceeds robot max reach", prev_version)

        blocked_obs = self._check_trajectory_collision(self.robot.position, target_pos)
        if blocked_obs:
            return self._fail(
                action_id,
                ActionType.MOVE.value,
                f"BlockedPath: Obstacle '{blocked_obs}' intersects move trajectory",
                prev_version,
            )

        self.robot.move_to(target_pos)
        if self.robot.carrying_object_id:
            held_obj = self.objects.get(self.robot.carrying_object_id)
            if held_obj:
                held_obj.move_to(target_pos)

        self.world_state_version += 1
        self.last_action_outcome = LastActionOutcome(
            action_type=ActionType.MOVE,
            status=LastActionStatus.SUCCESS,
        )

        result = ActionExecutionResult(
            success=True,
            action_type=ActionType.MOVE.value,
            action_id=action_id,
            simulation_time=self.simulation_time,
            previous_world_state_version=prev_version,
            new_world_state_version=self.world_state_version,
            failure_reason=None,
            details={
                "new_position": {"x": target_pos.x, "y": target_pos.y, "z": target_pos.z},
                "carrying_object_id": self.robot.carrying_object_id,
            },
        )
        self.execution_history.append(result)
        return result

    def _handle_release(self, action_id: str, prev_version: int) -> ActionExecutionResult:
        """Releases the currently held object."""
        if self.robot.carrying_object_id is None:
            return self._fail(action_id, ActionType.RELEASE.value, "Gripper is not carrying any object to release", prev_version)

        released_id = self.robot.carrying_object_id
        held_obj = self.objects.get(released_id)
        if held_obj:
            held_obj.current_holder = None
            # Place object at table surface or current robot xy position
            held_obj.move_to(SimPoint3D(x=self.robot.position.x, y=self.robot.position.y, z=self.robot.position.z))

        self.robot.detach_object()
        self.world_state_version += 1
        self.last_action_outcome = LastActionOutcome(
            action_type=ActionType.RELEASE,
            status=LastActionStatus.SUCCESS,
        )

        result = ActionExecutionResult(
            success=True,
            action_type=ActionType.RELEASE.value,
            action_id=action_id,
            simulation_time=self.simulation_time,
            previous_world_state_version=prev_version,
            new_world_state_version=self.world_state_version,
            failure_reason=None,
            details={"released_object_id": released_id},
        )
        self.execution_history.append(result)
        return result

    def _fail(self, action_id: str, action_type: str, reason: str, prev_version: int) -> ActionExecutionResult:
        """Handles action failure without incrementing physical world state version."""
        self.last_action_outcome = LastActionOutcome(
            action_type=ActionType(action_type) if action_type in ActionType.__members__ else None,
            status=LastActionStatus.FAILURE,
            error_message=reason,
        )
        result = ActionExecutionResult(
            success=False,
            action_type=action_type,
            action_id=action_id,
            simulation_time=self.simulation_time,
            previous_world_state_version=prev_version,
            new_world_state_version=prev_version,  # state remains unchanged on failure!
            failure_reason=reason,
        )
        self.execution_history.append(result)
        return result

    # --- Bridge to Decision Engine Models ---

    def to_decision_world_state(self, objective: TaskObjective = TaskObjective.PICK_AND_PLACE) -> WorldState:
        """
        Exports current simulation snapshot as a SĀRTHI Decision Engine WorldState.
        Ensures 100% interoperability between local simulation and the decision engine.
        """
        # 1. Robot
        de_robot = RobotState(
            position=Point3D(x=self.robot.position.x, y=self.robot.position.y, z=self.robot.position.z),
            gripper_open=self.robot.gripper_open,
            holding_object_id=self.robot.carrying_object_id,
            payload_mass_kg=(
                self.objects[self.robot.carrying_object_id].mass if self.robot.carrying_object_id and self.robot.carrying_object_id in self.objects else 0.0
            ),
            is_moving=False,
            max_payload_kg=self.robot.max_payload_kg,
            max_reach_m=self.robot.max_reach,
        )

        # 2. Objects (including active obstacles as obstacle entities)
        de_objects: List[WorldObject] = []
        for obj in self.objects.values():
            st = ObjectState.GRASPED if obj.current_holder == self.robot.robot_id else ObjectState.FREE
            de_objects.append(WorldObject(
                id=obj.id,
                name=obj.name,
                position=Point3D(x=obj.position.x, y=obj.position.y, z=obj.position.z),
                bounding_radius_m=obj.bounding_radius,
                mass_kg=obj.mass,
                state=st,
                is_target=obj.is_target,
                is_obstacle=False,
            ))

        for obs in self.obstacles.values():
            if obs.is_active:
                de_objects.append(WorldObject(
                    id=obs.id,
                    name=obs.name,
                    position=Point3D(x=obs.position.x, y=obs.position.y, z=obs.position.z),
                    bounding_radius_m=obs.bounding_radius,
                    mass_kg=5.0,
                    state=ObjectState.FREE,
                    is_target=False,
                    is_obstacle=True,
                ))

        # 3. Target Zone (Use first target zone or default)
        if self.target_zones:
            first_tz = next(iter(self.target_zones.values()))
            de_target = TargetZone(
                id=first_tz.id,
                position=Point3D(x=first_tz.position.x, y=first_tz.position.y, z=first_tz.position.z),
                tolerance_radius_m=first_tz.tolerance_radius,
            )
        else:
            de_target = TargetZone(
                id="default_zone",
                position=Point3D(x=0.4, y=-0.2, z=0.1),
                tolerance_radius_m=0.05,
            )

        # 4. Environment
        de_env = EnvironmentState(
            min_x=self.workspace_bounds.min_x,
            max_x=self.workspace_bounds.max_x,
            min_y=self.workspace_bounds.min_y,
            max_y=self.workspace_bounds.max_y,
            min_z=self.workspace_bounds.min_z,
            max_z=self.workspace_bounds.max_z,
            dynamic_obstacles_detected=len(self.active_disturbances) > 0,
            slip_risk_level=0.1,
            friction_coefficient=0.6,
        )

        return WorldState(
            version=self.world_state_version,
            timestamp_ns=int(self.simulation_time * 1e9),
            robot=de_robot,
            objects=de_objects,
            target=de_target,
            environment=de_env,
            task_objective=objective,
            active_constraints=[],
            last_action_outcome=self.last_action_outcome,
        )
