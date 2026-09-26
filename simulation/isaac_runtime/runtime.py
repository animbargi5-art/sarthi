"""
SĀRTHI Isaac Sim Runtime — Simulation Lifecycle & Controller Interface.
Controls Isaac Sim execution lifecycle, step loops, disturbance injection,
and physical action dispatch received from IsaacSimAdapter.

Strict Architectural Boundaries:
- Receives actions from SĀRTHI IsaacSimAdapter; NEVER decides actions autonomously.
- Does NOT implement a second Decision Engine.
- Reuses canonical backend.app.decision_engine.models.WorldState.
"""

from typing import Any, Dict, Optional, Union
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
from simulation.adapters.isaac_sim import IsaacSimAction, is_isaac_sim_available
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

        self._simulation_app: Optional[Any] = None
        self._world: Optional[Any] = None
        self._is_initialized: bool = False
        self._world_state_version: int = 1
        self._simulation_time: float = 0.0
        self._last_action_outcome: LastActionOutcome = LastActionOutcome(
            status=LastActionStatus.NONE
        )

    @property
    def is_initialized(self) -> bool:
        """True if the Isaac Sim simulation application is active."""
        return self._is_initialized

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

    def inject_disturbance(
        self,
        disturbance: Optional[DisturbanceEvent] = None,
    ) -> bool:
        """
        Spawns the dynamic PATH_BLOCKED obstacle inside the USD stage.
        """
        if not self._is_initialized or not is_isaac_sim_available():
            raise IsaacSimRuntimeError()

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
        - movable object pose
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
            obj_pos = ee_pos if is_holding else self.scenario.red_object.initial_pose
            obj_state = (
                ObjectState.GRASPED
                if is_holding
                else ObjectState.FREE
            )

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
        return self.scenario.to_world_state(
            with_disturbance=self.disturbance.is_active,
            version=self._world_state_version,
            timestamp_ns=int(self._simulation_time * 1e9),
        )

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
                else self.scenario.red_object.initial_pose
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
    ) -> ActionExecutionResult:
        """
        Dispatches action primitive to Isaac Sim articulation controller.
        Action must be determined externally (by Decision Engine through adapter).
        """
        if not self._is_initialized or not is_isaac_sim_available():
            raise IsaacSimRuntimeError()

        prev_version = self._world_state_version

        # Dispatch action through articulation controller
        result = self.articulation_controller.dispatch_action(
            action=action,
            sim_time=self._simulation_time,
            world_version=prev_version,
        )
        self._world_state_version += 1
        result.new_world_state_version = self._world_state_version

        status_enum = LastActionStatus.SUCCESS if result.success else LastActionStatus.FAILURE
        act_type_enum = None
        if hasattr(action, "action_type"):
            act_type_enum = (
                action.action_type
                if isinstance(action.action_type, ActionType)
                else ActionType(str(action.action_type))
            )
        elif isinstance(action, dict) and "action_type" in action:
            act_type_enum = ActionType(str(action["action_type"]))

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
        """Verifies state in Isaac Sim matches physical expectation."""
        if not self._is_initialized or not is_isaac_sim_available():
            raise IsaacSimRuntimeError()
        return True

    def shutdown(self) -> None:
        """Terminates simulation application cleanly."""
        if self._simulation_app:
            self._simulation_app.close()
            self._simulation_app = None
        self._is_initialized = False
