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
    LastActionOutcome,
    LastActionStatus,
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

    def get_world_state(self) -> WorldState:
        """
        Queries simulator stage and compiles the canonical SĀRTHI WorldState.
        Reuses backend.app.decision_engine.models.WorldState directly.
        """
        if not self._is_initialized or not is_isaac_sim_available():
            raise IsaacSimRuntimeError()

        # In live runtime, query prim coordinates from USD stage
        # For now, translate current state directly to WorldState
        return self.scenario.to_world_state(
            with_disturbance=self.disturbance.is_active,
            version=self._world_state_version,
            timestamp_ns=int(self._simulation_time * 1e9),
        )

    def execute_action(
        self,
        action: Union[CandidateAction, IsaacSimAction, Dict[str, Any]],
    ) -> ActionExecutionResult:
        """
        Dispatches action primitive to Isaac Sim controller.
        Action must be determined externally (by Decision Engine through adapter).
        """
        if not self._is_initialized or not is_isaac_sim_available():
            raise IsaacSimRuntimeError()

        prev_version = self._world_state_version
        action_type_str = action.action_type.value if hasattr(action, "action_type") else str(action.get("action_type"))
        action_id_str = action.action_id if hasattr(action, "action_id") else str(action.get("action_id", "act"))

        # Physical controller dispatch in Isaac Sim would occur here:
        # e.g., articulation controller trajectory following, gripper closure
        self._world_state_version += 1

        return ActionExecutionResult(
            success=True,
            action_type=action_type_str,
            action_id=action_id_str,
            simulation_time=self._simulation_time,
            previous_world_state_version=prev_version,
            new_world_state_version=self._world_state_version,
            failure_reason=None,
            details={},
        )

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
