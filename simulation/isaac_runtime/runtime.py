"""
SĀRTHI Isaac Sim Runtime — Simulation Lifecycle & Controller Interface.
Controls Isaac Sim execution lifecycle, step loops, disturbance injection,
and physical action dispatch received from IsaacSimAdapter.

Strict Architectural Boundaries:
- Receives actions from SĀRTHI IsaacSimAdapter; NEVER decides actions autonomously.
- Does NOT implement a second Decision Engine.
- Reuses canonical backend.app.decision_engine.models.WorldState.
"""

import math
from typing import Any, Dict, List, Optional, Union
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
from simulation.adapters.isaac_sim import (
    IsaacSimAction,
    IsaacSimDisturbance,
    is_isaac_sim_available,
)
from simulation.core.events import (
    ActionExecutionResult,
    DisturbanceEvent,
    DisturbanceType,
)
from simulation.isaac_runtime.articulation import SarthiArticulationController
from simulation.isaac_runtime.disturbances import PathBlockedDisturbance
from simulation.isaac_runtime.robot_scene import RobotSceneConfig
from simulation.isaac_runtime.scene_builder import SarthiSceneBuilder
from simulation.scenarios.tabletop_pick_place import (
    TabletopPickPlaceScenario,
    create_default_scenario,
)


class IsaacSimRuntimeError(RuntimeError):
    """
    Raised when Isaac Sim operations are attempted outside an active Isaac Sim environment.
    """
    DEFAULT_MESSAGE = (
        "Isaac Sim runtime is required. Run this script using the Isaac Sim Python environment."
    )

    def __init__(self, message: Optional[str] = None):
        super().__init__(message or self.DEFAULT_MESSAGE)


class SarthiIsaacRuntime:
    """
    Manages the lifecycle of an NVIDIA Isaac Sim 6.x execution session.

    Execution Flow:
    1. initialize()       -> Start SimulationApp (headless or GUI)
    2. load_scenario()    -> Assemble USD scene using SarthiSceneBuilder
    3. step()             -> Advance physics and rendering
    4. execute_action()   -> Execute action received from IsaacSimAdapter
    5. inject_disturbance()-> Spawn dynamic obstacle in USD stage
    6. shutdown()         -> Clean up stage and terminate SimulationApp
    """

    def __init__(
        self,
        scenario: Optional[TabletopPickPlaceScenario] = None,
        robot_config: Optional[RobotSceneConfig] = None,
        physics_steps_per_action: int = 60,
    ):
        self.scenario = scenario or create_default_scenario()
        self.robot_config = robot_config or RobotSceneConfig(
            robot_id=self.scenario.robot.robot_id
        )
        self.builder = SarthiSceneBuilder(
            scenario=self.scenario,
            robot_config=self.robot_config,
        )
        self.disturbance = PathBlockedDisturbance(scenario=self.scenario)

        self.articulation_controller = SarthiArticulationController(
            robot_id=self.scenario.robot.robot_id,
            usd_prim_path=self.robot_config.usd_prim_path,
        )

        self.physics_steps_per_action = physics_steps_per_action
        self._simulation_app: Optional[Any] = None
        self._world: Optional[Any] = None
        self._is_initialized: bool = False
        self._world_state_version: int = 1
        self._simulation_time: float = 0.0
        self._last_action_outcome: LastActionOutcome = LastActionOutcome(
            status=LastActionStatus.NONE
        )
        self._released_object_positions: Dict[str, Point3D] = {}
        self._last_verification_failure: Optional[str] = None

    @property
    def is_initialized(self) -> bool:
        """True if the Isaac Sim simulation application is active."""
        return self._is_initialized

    @property
    def last_verification_failure(self) -> Optional[str]:
        """Human-readable failure reason from the most recent action verification."""
        return self._last_verification_failure

    def initialize(
        self,
        headless: bool = False,
        stage_path: Optional[str] = None,
    ) -> None:
        """
        Starts the Isaac Sim application.

        Raises:
            IsaacSimRuntimeError: If executed outside an Isaac Sim runtime environment.
        """
        if not is_isaac_sim_available():
            raise IsaacSimRuntimeError()

        from isaacsim import SimulationApp

        self._simulation_app = SimulationApp({"headless": headless})
        self._is_initialized = True

    def load_scenario(self, scenario: Optional[TabletopPickPlaceScenario] = None) -> Dict[str, Any]:
        """
        Builds and populates the tabletop scenario inside the active stage.
        """
        if not self._is_initialized or not is_isaac_sim_available():
            raise IsaacSimRuntimeError()

        self._released_object_positions.clear()

        if scenario:
            self.scenario = scenario
            self.builder = SarthiSceneBuilder(
                scenario=self.scenario,
                robot_config=self.robot_config,
            )
            self.disturbance = PathBlockedDisturbance(scenario=self.scenario)

        prims = self.builder.build()
        self._world = self.builder._world
        self._world.reset()

        # Bind articulation controller to instantiated robot prim
        if "robot" in prims and prims["robot"] is not None:
            self.articulation_controller.bind_robot(prims["robot"])

        return prims

    def step(self, render: bool = True) -> None:
        """Advances physics and optional rendering by one simulation tick."""
        if not self._is_initialized or not is_isaac_sim_available():
            raise IsaacSimRuntimeError()

        if self._world:
            self._world.step(render=render)
            self._simulation_time += 1.0 / 60.0

    def _get_object_position(self, object_id: str) -> Point3D:
        """
        Returns current Cartesian coordinates of the specified object.
        Queries live USD prim if available, otherwise falls back to last physically
        released coordinates, or scenario initial pose.
        """
        if self._world is not None and is_isaac_sim_available():
            try:
                from omni.isaac.core.prims import XFormPrim
                prim_path = f"/World/Objects/{object_id}"
                prim = XFormPrim(prim_path=prim_path)
                if prim.is_valid():
                    pos, _ = prim.get_world_pose()
                    return Point3D(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            except Exception:
                pass

        if object_id in self._released_object_positions:
            return self._released_object_positions[object_id]

        if object_id == self.scenario.red_object.object_id:
            return self.scenario.red_object.initial_pose

        return Point3D(x=0.0, y=0.0, z=0.0)

    def inject_disturbance(
        self,
        disturbance: Optional[Union[DisturbanceEvent, IsaacSimDisturbance, Dict[str, Any]]] = None,
    ) -> bool:
        """
        Spawns the dynamic PATH_BLOCKED obstacle inside the USD stage.
        Accepts both DisturbanceEvent and IsaacSimDisturbance, normalizing internally.
        """
        if not self._is_initialized or not is_isaac_sim_available():
            raise IsaacSimRuntimeError()

        if disturbance is not None:
            if isinstance(disturbance, DisturbanceEvent):
                from simulation.adapters.isaac_sim import IsaacSimAdapter
                norm_dist = IsaacSimAdapter.translate_disturbance(disturbance)
            elif isinstance(disturbance, IsaacSimDisturbance):
                norm_dist = disturbance
            elif isinstance(disturbance, dict):
                if "usd_prim_path" in disturbance:
                    norm_dist = IsaacSimDisturbance.model_validate(disturbance)
                else:
                    dist_ev = DisturbanceEvent.model_validate(disturbance)
                    from simulation.adapters.isaac_sim import IsaacSimAdapter
                    norm_dist = IsaacSimAdapter.translate_disturbance(dist_ev)
            else:
                raise TypeError(f"Unsupported disturbance type: {type(disturbance).__name__}")

            if hasattr(norm_dist, "position"):
                pos = norm_dist.position
                self.disturbance.position = Point3D(x=pos["x"], y=pos["y"], z=pos["z"])

        self.disturbance.spawn(self._world)
        self._world_state_version += 1
        return True

    def read_live_world_state(self) -> WorldState:
        """
        Reads live telemetry from the articulation controller and scene prims,
        and constructs the canonical SĀRTHI WorldState without schema duplication.
        Exposes:
        - robot joint positions and velocities
        - end-effector pose
        - gripper state (open/closed/holding)
        - movable object pose (follows end-effector when held; persists after release)
        - target pose
        - dynamic obstacle state
        """
        if not self._is_initialized or not is_isaac_sim_available():
            raise IsaacSimRuntimeError()

        if self.articulation_controller.is_bound:
            ctrl = self.articulation_controller
            joint_positions = ctrl.get_joint_positions()
            joint_velocities = ctrl.get_joint_velocities()
            ee_pos = ctrl.get_end_effector_position()
            is_closed = ctrl.is_gripper_closed
            is_holding = ctrl.is_holding_object
            holding_id = ctrl.holding_object_id

            # Determine object position (linked to end-effector if held)
            if is_holding:
                obj_pos = ee_pos
                obj_state = ObjectState.GRASPED
            else:
                obj_pos = self._get_object_position(self.scenario.red_object.object_id)
                target_pos = self.scenario.blue_target.target_pose
                dist_to_target = math.sqrt(
                    (obj_pos.x - target_pos.x) ** 2
                    + (obj_pos.y - target_pos.y) ** 2
                    + (obj_pos.z - target_pos.z) ** 2
                )
                if dist_to_target <= self.scenario.blue_target.tolerance_radius_m:
                    obj_state = ObjectState.PLACED
                else:
                    obj_state = ObjectState.FREE

            robot_state = RobotState(
                position=ee_pos,
                gripper_open=not is_closed,
                holding_object_id=holding_id,
                payload_mass_kg=self.scenario.red_object.mass_kg if is_holding else 0.0,
                is_moving=not ctrl.is_stopped,
                max_payload_kg=self.scenario.robot.max_payload_kg,
                max_reach_m=self.scenario.robot.max_reach_m,
            )

            objects_list = [
                WorldObject(
                    id=self.scenario.red_object.object_id,
                    name=self.scenario.red_object.name,
                    position=obj_pos,
                    bounding_radius_m=self.scenario.red_object.bounding_radius_m,
                    mass_kg=self.scenario.red_object.mass_kg,
                    state=obj_state,
                    is_target=True,
                    is_obstacle=False,
                )
            ]

            if self.disturbance.is_active:
                objects_list.append(
                    WorldObject(
                        id=self.scenario.obstacle.obstacle_id,
                        name=self.scenario.obstacle.name,
                        position=self.scenario.obstacle.position,
                        bounding_radius_m=self.scenario.obstacle.bounding_radius_m,
                        mass_kg=self.scenario.obstacle.mass_kg,
                        state=ObjectState.FREE,
                        is_target=False,
                        is_obstacle=True,
                    )
                )

            target_zone = TargetZone(
                id=self.scenario.blue_target.target_id,
                position=self.scenario.blue_target.target_pose,
                tolerance_radius_m=self.scenario.blue_target.tolerance_radius_m,
            )

            env_state = EnvironmentState(
                min_x=self.scenario.workspace.min_x,
                max_x=self.scenario.workspace.max_x,
                min_y=self.scenario.workspace.min_y,
                max_y=self.scenario.workspace.max_y,
                min_z=self.scenario.workspace.min_z,
                max_z=self.scenario.workspace.max_z,
                dynamic_obstacles_detected=self.disturbance.is_active,
                slip_risk_level=0.0,
                friction_coefficient=0.6,
            )

            return WorldState(
                version=self._world_state_version,
                timestamp_ns=int(self._simulation_time * 1e9),
                robot=robot_state,
                objects=objects_list,
                target=target_zone,
                environment=env_state,
                task_objective=TaskObjective.PICK_AND_PLACE,
                active_constraints=[],
                last_action_outcome=self._last_action_outcome,
            )

        # Fallback to scenario world state if articulation is not yet bound
        ws = self.scenario.to_world_state(
            with_disturbance=self.disturbance.is_active,
            version=self._world_state_version,
            timestamp_ns=int(self._simulation_time * 1e9),
        )
        if self.scenario.red_object.object_id in self._released_object_positions:
            rel_pos = self._released_object_positions[self.scenario.red_object.object_id]
            for obj in ws.objects:
                if obj.id == self.scenario.red_object.object_id:
                    obj.position = rel_pos
                    target_pos = self.scenario.blue_target.target_pose
                    dist_to_target = math.sqrt(
                        (rel_pos.x - target_pos.x) ** 2
                        + (rel_pos.y - target_pos.y) ** 2
                        + (rel_pos.z - target_pos.z) ** 2
                    )
                    if dist_to_target <= self.scenario.blue_target.tolerance_radius_m:
                        obj.state = ObjectState.PLACED
        return ws

    def get_world_state(self) -> WorldState:
        """
        Queries simulator stage and compiles the canonical SĀRTHI WorldState.
        Reuses backend.app.decision_engine.models.WorldState directly.
        """
        return self.read_live_world_state()

    def get_telemetry(self) -> Dict[str, Any]:
        """
        Exposes full runtime telemetry including joint positions/velocities,
        end-effector Cartesian pose, gripper state, object poses, and disturbance status.
        """
        art_state = self.articulation_controller.get_articulation_state()
        return {
            "robot_id": self.scenario.robot.robot_id,
            "joint_positions": art_state["joint_positions"],
            "joint_velocities": art_state["joint_velocities"],
            "end_effector_position": art_state["end_effector_position"],
            "gripper_state": art_state["gripper_state"],
            "is_gripper_closed": art_state["is_gripper_closed"],
            "is_holding_object": art_state["is_holding_object"],
            "holding_object_id": art_state["holding_object_id"],
            "movable_object_pose": (
                art_state["end_effector_position"]
                if art_state["is_holding_object"]
                else self._get_object_position(self.scenario.red_object.object_id)
            ),
            "target_pose": self.scenario.blue_target.target_pose,
            "dynamic_obstacles_detected": self.disturbance.is_active,
            "is_stopped": art_state["is_stopped"],
            "world_state_version": self._world_state_version,
            "simulation_time": self._simulation_time,
        }

    def execute_action(
        self,
        action: Union[CandidateAction, IsaacSimAction, Dict[str, Any]],
        physics_steps: Optional[int] = None,
    ) -> ActionExecutionResult:
        """
        Dispatches action primitive to Isaac Sim articulation controller.
        Action must be determined externally (by Decision Engine through adapter).
        """
        if not self._is_initialized or not is_isaac_sim_available():
            raise IsaacSimRuntimeError()

        prev_version = self._world_state_version
        previously_held_id = self.articulation_controller.holding_object_id

        # Dispatch action through articulation controller
        result = self.articulation_controller.dispatch_action(
            action=action,
            sim_time=self._simulation_time,
            world_version=prev_version,
        )

        # Advance simulation physics to allow articulation convergence
        if result.success and self._world is not None:
            ticks = physics_steps if physics_steps is not None else self.physics_steps_per_action
            if ticks > 0:
                for _ in range(ticks):
                    self.step(render=False)
            result.simulation_time = self._simulation_time

        act_type_enum = None
        if hasattr(action, "action_type"):
            act_type_enum = (
                action.action_type
                if isinstance(action.action_type, ActionType)
                else ActionType(str(action.action_type))
            )
        elif isinstance(action, dict) and "action_type" in action:
            act_type_enum = ActionType(str(action["action_type"]))

        if result.success:
            self._world_state_version += 1
            result.new_world_state_version = self._world_state_version

            # When releasing, persist the released payload coordinates
            if act_type_enum == ActionType.RELEASE:
                ee_pos = self.articulation_controller.get_end_effector_position()
                released_id = previously_held_id
                if not released_id:
                    if hasattr(action, "target_object_id") and action.target_object_id:
                        released_id = str(action.target_object_id)
                    elif isinstance(action, dict) and action.get("target_object_id"):
                        released_id = str(action["target_object_id"])
                    else:
                        released_id = self.scenario.red_object.object_id
                self._released_object_positions[released_id] = ee_pos
        else:
            # Preserve world state version intact on failure
            result.new_world_state_version = self._world_state_version

        status_enum = LastActionStatus.SUCCESS if result.success else LastActionStatus.FAILURE
        self._last_action_outcome = LastActionOutcome(
            action_type=act_type_enum,
            status=status_enum,
            error_message=result.failure_reason,
        )

        return result

    def verify_action_result(
        self,
        action: Any,
        expected_outcome: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Verifies state in Isaac Sim matches physical expectation.
        Performs explicit validation for:
        - APPROACH: end-effector reached target within tolerance
        - REPOSITION: end-effector reached safe waypoint within tolerance
        - MOVE: end-effector/payload reached destination; blocked motion rejected
        - GRASP: gripper closed, object within grasp radius, payload attached
        - RELEASE: gripper open, payload detached, object at expected released location
        - STOP: robot stopped safely
        """
        if not self._is_initialized or not is_isaac_sim_available():
            raise IsaacSimRuntimeError()

        outcome = expected_outcome or {}
        ctrl = self.articulation_controller

        # If last action execution failed, verification fails
        if self._last_action_outcome.status == LastActionStatus.FAILURE:
            self._last_verification_failure = (
                f"Action execution failed: {self._last_action_outcome.error_message}"
            )
            return False

        # Extract ActionType
        act_type: Optional[ActionType] = None
        if hasattr(action, "action_type"):
            raw = action.action_type
            act_type = raw if isinstance(raw, ActionType) else ActionType(str(raw))
        elif isinstance(action, dict) and "action_type" in action:
            act_type = ActionType(str(action["action_type"]))

        if act_type is None:
            self._last_verification_failure = "Unable to determine action_type for verification."
            return False

        tolerance = float(outcome.get("tolerance", 0.05))

        # Helper to extract target position
        target_pos = outcome.get("target_position")
        if target_pos is None:
            if hasattr(action, "target_position") and action.target_position:
                target_pos = action.target_position
            elif isinstance(action, dict) and "target_position" in action:
                target_pos = action.get("target_position")

        if act_type == ActionType.APPROACH:
            if target_pos is not None:
                if not ctrl.is_at_target(target_pos, tolerance_m=tolerance):
                    current_ee = ctrl.get_end_effector_position()
                    self._last_verification_failure = (
                        f"APPROACH verification failed: End-effector at ({current_ee.x:.3f}, {current_ee.y:.3f}, {current_ee.z:.3f}) "
                        f"not within {tolerance:.3f}m of target."
                    )
                    return False
            self._last_verification_failure = None
            return True

        elif act_type == ActionType.REPOSITION:
            if target_pos is not None:
                if not ctrl.is_at_target(target_pos, tolerance_m=tolerance):
                    current_ee = ctrl.get_end_effector_position()
                    self._last_verification_failure = (
                        f"REPOSITION verification failed: End-effector at ({current_ee.x:.3f}, {current_ee.y:.3f}, {current_ee.z:.3f}) "
                        f"did not reach waypoint within {tolerance:.3f}m."
                    )
                    return False
            self._last_verification_failure = None
            return True

        elif act_type == ActionType.MOVE:
            # 1. Blocked/invalid movement check: dynamic obstacle in collision path
            if self.disturbance.is_active:
                ee_pos = ctrl.get_end_effector_position()
                obs_pos = self.disturbance.position
                dist_to_obs = math.sqrt(
                    (ee_pos.x - obs_pos.x) ** 2
                    + (ee_pos.y - obs_pos.y) ** 2
                    + (ee_pos.z - obs_pos.z) ** 2
                )
                safe_margin = 0.10
                if dist_to_obs < safe_margin:
                    self._last_verification_failure = (
                        f"MOVE verification failed: Path blocked by obstacle at ({obs_pos.x:.2f}, {obs_pos.y:.2f}, {obs_pos.z:.2f}) "
                        f"(distance {dist_to_obs:.3f}m < safe margin {safe_margin:.3f}m)."
                    )
                    return False

            # 2. Check destination reach
            if target_pos is not None:
                if not ctrl.is_at_target(target_pos, tolerance_m=tolerance):
                    current_ee = ctrl.get_end_effector_position()
                    self._last_verification_failure = (
                        f"MOVE verification failed: End-effector at ({current_ee.x:.3f}, {current_ee.y:.3f}, {current_ee.z:.3f}) "
                        f"not within {tolerance:.3f}m of target destination."
                    )
                    return False
            self._last_verification_failure = None
            return True

        elif act_type == ActionType.GRASP:
            # Gripper must be closed
            if not ctrl.is_gripper_closed:
                self._last_verification_failure = "GRASP verification failed: Gripper is not closed."
                return False

            # Payload must be attached
            if not ctrl.is_holding_object:
                self._last_verification_failure = "GRASP verification failed: No object is attached."
                return False

            # If expected carrying object is specified, verify match
            expected_payload = outcome.get("carrying_object_id")
            if expected_payload and ctrl.holding_object_id != str(expected_payload):
                self._last_verification_failure = (
                    f"GRASP verification failed: Attached object '{ctrl.holding_object_id}' != expected '{expected_payload}'."
                )
                return False

            # Object must be within configured grasp radius
            grasp_radius = float(outcome.get("grasp_radius", 0.12))
            if target_pos is not None:
                if not ctrl.is_at_target(target_pos, tolerance_m=grasp_radius):
                    self._last_verification_failure = (
                        f"GRASP verification failed: Target position is outside grasp radius {grasp_radius:.3f}m."
                    )
                    return False

            self._last_verification_failure = None
            return True

        elif act_type == ActionType.RELEASE:
            # Gripper must be open
            if ctrl.is_gripper_closed:
                self._last_verification_failure = "RELEASE verification failed: Gripper is still closed."
                return False

            # Payload must no longer be attached
            if ctrl.is_holding_object or ctrl.holding_object_id is not None:
                self._last_verification_failure = "RELEASE verification failed: Payload is still attached."
                return False

            # Object remains at expected released location
            expected_rel_pos = outcome.get("target_position") or target_pos
            if expected_rel_pos is not None:
                target_obj_id = outcome.get("target_object_id")
                if not target_obj_id:
                    if hasattr(action, "target_object_id") and action.target_object_id:
                        target_obj_id = action.target_object_id
                    elif isinstance(action, dict) and "target_object_id" in action:
                        target_obj_id = action.get("target_object_id")
                    else:
                        target_obj_id = self.scenario.red_object.object_id

                obj_actual_pos = self._get_object_position(str(target_obj_id))
                if isinstance(expected_rel_pos, dict):
                    rx = float(expected_rel_pos.get("x", 0.0))
                    ry = float(expected_rel_pos.get("y", 0.0))
                    rz = float(expected_rel_pos.get("z", 0.0))
                else:
                    rx, ry, rz = expected_rel_pos.x, expected_rel_pos.y, expected_rel_pos.z

                dist_rel = math.sqrt(
                    (obj_actual_pos.x - rx) ** 2
                    + (obj_actual_pos.y - ry) ** 2
                    + (obj_actual_pos.z - rz) ** 2
                )
                if dist_rel > tolerance:
                    self._last_verification_failure = (
                        f"RELEASE verification failed: Object at ({obj_actual_pos.x:.3f}, {obj_actual_pos.y:.3f}, {obj_actual_pos.z:.3f}) "
                        f"is {dist_rel:.3f}m from expected release position ({rx:.3f}, {ry:.3f}, {rz:.3f}) > {tolerance:.3f}m."
                    )
                    return False

            self._last_verification_failure = None
            return True

        elif act_type == ActionType.STOP:
            if not ctrl.is_stopped:
                self._last_verification_failure = "STOP verification failed: Robot is not in stopped state."
                return False
            self._last_verification_failure = None
            return True

        self._last_verification_failure = f"Unknown action type: {act_type}"
        return False

    def shutdown(self) -> None:
        """Terminates simulation application cleanly."""
        if self._simulation_app:
            self._simulation_app.close()
            self._simulation_app = None
        self._is_initialized = False
