"""
SĀRTHI Simulation Adapters — NVIDIA Isaac Sim Adapter Boundary.
Provides the integration boundary between SĀRTHI and NVIDIA Isaac Sim 6.x.

Strict Architectural Guarantees:
- Never imports Isaac Sim packages at module top-level.
- Module remains importable in lightweight / CPU / CI environments without Isaac Sim.
- Reuses canonical SĀRTHI WorldState and ActionType definitions; never duplicates them.
- Strictly decoupled from Decision Engine: adapter only executes and translates,
  never selects actions or bypasses constraints.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

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
from simulation.adapters.base import SimulationAdapter
from simulation.core.events import (
    ActionExecutionResult,
    DisturbanceEvent,
    DisturbanceType,
)


# ---------------------------------------------------------------------------
# Exceptions & Availability Detection
# ---------------------------------------------------------------------------

class IsaacSimUnavailableError(RuntimeError):
    """
    Raised when Isaac Sim packages or live runtime environment are not available.
    Ensures clear, actionable error messaging without failing at module import time.
    """
    DEFAULT_MESSAGE = (
        "Isaac Sim is not installed in this environment. "
        "Use the Isaac Sim runtime environment to execute IsaacSimAdapter."
    )

    def __init__(self, message: Optional[str] = None):
        super().__init__(message or self.DEFAULT_MESSAGE)


def is_isaac_sim_available() -> bool:
    """
    Checks if NVIDIA Isaac Sim packages are importable in the active environment.
    Evaluated lazily; does not raise exceptions.
    """
    try:
        import isaacsim  # noqa: F401
        return True
    except ImportError:
        try:
            import omni.isaac.core  # noqa: F401
            return True
        except ImportError:
            return False


def require_isaac_sim() -> None:
    """
    Asserts Isaac Sim availability before dispatching calls to the simulator runtime.
    Raises IsaacSimUnavailableError if unavailable.
    """
    if not is_isaac_sim_available():
        raise IsaacSimUnavailableError()


# ---------------------------------------------------------------------------
# Isaac Sim Representation Models
# ---------------------------------------------------------------------------

class IsaacSimAction(BaseModel):
    """
    Structured action command mapped for NVIDIA Isaac Sim controllers.
    Carries low-level controller primitives derived from canonical SĀRTHI CandidateAction.
    """
    action_type: ActionType = Field(..., description="Canonical SĀRTHI action primitive")
    action_id: str = Field(..., description="Unique action identifier")
    primitive_name: str = Field(..., description="Isaac Sim controller primitive (e.g. cartesian_approach)")
    target_position: Optional[Dict[str, float]] = Field(
        default=None,
        description="Target Cartesian coordinates in USD stage world frame (m)"
    )
    gripper_target: Optional[str] = Field(
        default=None,
        description="Gripper jaw command: OPEN, CLOSE, or HOLD"
    )
    speed_scale: float = Field(default=1.0, ge=0.0, le=1.0, description="Controller speed scale [0.0, 1.0]")
    expected_force_n: float = Field(default=5.0, ge=0.0, description="Contact/grasp force limit (N)")
    target_object_id: Optional[str] = Field(default=None, description="Manipulated object entity ID")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Auxiliary controller parameters")


class IsaacSimDisturbance(BaseModel):
    """
    Structured representation of an environmental disturbance in the USD stage.
    Maps SĀRTHI conceptual disturbances (e.g. PATH_BLOCKED) to USD rigid bodies / colliders.
    """
    event_id: str = Field(..., description="Unique disturbance event identifier")
    disturbance_type: str = Field(..., description="Disturbance primitive type")
    usd_prim_path: str = Field(..., description="USD stage prim path for obstacle entity")
    usd_prim_type: str = Field(default="Cube", description="Geometric archetype (Cube, Cylinder, Mesh)")
    position: Dict[str, float] = Field(..., description="Spatial position in USD stage frame (m)")
    dimensions: Dict[str, float] = Field(..., description="Bounding dimensions in meters")
    is_active: bool = Field(default=True, description="Active collision state in physics scene")
    physics_enabled: bool = Field(default=True, description="Rigid body dynamics enabled")


# ---------------------------------------------------------------------------
# Isaac Sim Adapter Implementation
# ---------------------------------------------------------------------------

class IsaacSimAdapter(SimulationAdapter):
    """
    Concrete adapter bridging SĀRTHI with NVIDIA Isaac Sim 6.x.
    Fulfills the SimulationAdapter abstract contract.

    Responsibilities:
    - Simulator interaction and lifecycle management.
    - Reading raw simulator state and translating to canonical SĀRTHI WorldState.
    - Mapping SĀRTHI CandidateAction into Isaac Sim controller actions.
    - Executing actions via Isaac Sim articulation/Cartesian controllers.
    - Injecting environmental disturbances into the USD stage.

    Non-Responsibilities:
    - Never evaluates constraints or decides physical actions (Decision Engine responsibility).
    - Never hardcodes recovery trajectories (e.g., REPOSITION).
    """

    # Deterministic mapping table: SĀRTHI ActionType -> Isaac Sim primitive name & gripper command
    ACTION_PRIMITIVE_MAP = {
        ActionType.APPROACH: ("cartesian_approach", "OPEN"),
        ActionType.REPOSITION: ("cartesian_reposition", "HOLD"),
        ActionType.GRASP: ("gripper_close", "CLOSE"),
        ActionType.MOVE: ("cartesian_move", "HOLD"),
        ActionType.RELEASE: ("gripper_open", "OPEN"),
        ActionType.STOP: ("emergency_hold", "HOLD"),
    }

    def __init__(
        self,
        sim_backend: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the Isaac Sim adapter boundary.

        Args:
            sim_backend: Optional live or mocked simulation backend interface.
                         If None, live Isaac Sim runtime is required on method invocation.
            config: Optional configuration dictionary (USD stage path, physics rate, etc.).
        """
        self._backend = sim_backend
        self._config = config or {}
        self._world_state_version: int = 1
        self._simulation_time: float = 0.0
        self._last_action_outcome: LastActionOutcome = LastActionOutcome(status=LastActionStatus.NONE)
        self._execution_history: List[ActionExecutionResult] = []
        self._active_obstacles: Dict[str, Dict[str, Any]] = {}

    @property
    def is_connected(self) -> bool:
        """True if a simulation backend is connected or Isaac Sim is available."""
        if self._backend is not None:
            return True
        return is_isaac_sim_available()

    # -----------------------------------------------------------------------
    # Action Translation (SĀRTHI -> Isaac Sim)
    # -----------------------------------------------------------------------

    @classmethod
    def translate_action(
        cls,
        action: Union[CandidateAction, Dict[str, Any]],
    ) -> IsaacSimAction:
        """
        Deterministically translates a SĀRTHI CandidateAction into an Isaac Sim action representation.
        The Decision Engine retains complete authority over action selection; this method
        serves strictly as a command formatting bridge.
        """
        if isinstance(action, dict):
            # Parse dict into CandidateAction for schema safety
            action_type_val = action.get("action_type")
            if isinstance(action_type_val, str):
                action_type = ActionType(action_type_val)
            else:
                action_type = action_type_val

            action_id = str(action.get("action_id", "act_unknown"))
            target_obj = action.get("target_object_id")
            pos_dict = action.get("target_position")
            target_pos = None
            if isinstance(pos_dict, dict):
                target_pos = {"x": float(pos_dict["x"]), "y": float(pos_dict["y"]), "z": float(pos_dict["z"])}
            elif isinstance(pos_dict, Point3D):
                target_pos = {"x": pos_dict.x, "y": pos_dict.y, "z": pos_dict.z}

            speed = float(action.get("speed_scale", 1.0))
            force = float(action.get("expected_force_n", 5.0))
            params = dict(action.get("parameters", {}))
        elif isinstance(action, CandidateAction):
            action_type = action.action_type
            action_id = action.action_id
            target_obj = action.target_object_id
            target_pos = None
            if action.target_position:
                target_pos = {
                    "x": action.target_position.x,
                    "y": action.target_position.y,
                    "z": action.target_position.z,
                }
            speed = action.speed_scale
            force = action.expected_force_n
            params = dict(action.parameters)
        else:
            raise TypeError(f"Expected CandidateAction or dict, got {type(action).__name__}")

        if action_type not in cls.ACTION_PRIMITIVE_MAP:
            raise ValueError(f"Unsupported action primitive for Isaac Sim translation: {action_type}")

        prim_name, gripper_cmd = cls.ACTION_PRIMITIVE_MAP[action_type]

        return IsaacSimAction(
            action_type=action_type,
            action_id=action_id,
            primitive_name=prim_name,
            target_position=target_pos,
            gripper_target=gripper_cmd,
            speed_scale=speed,
            expected_force_n=force,
            target_object_id=target_obj,
            parameters=params,
        )

    # -----------------------------------------------------------------------
    # Disturbance Translation (SĀRTHI -> Isaac Sim USD Stage)
    # -----------------------------------------------------------------------

    @classmethod
    def translate_disturbance(cls, disturbance: DisturbanceEvent) -> IsaacSimDisturbance:
        """
        Translates a conceptual SĀRTHI DisturbanceEvent into a USD obstacle specification.
        Currently supports PATH_BLOCKED by provisioning a dynamic collision obstacle in the stage.
        """
        if disturbance.disturbance_type != DisturbanceType.PATH_BLOCKED:
            raise ValueError(f"Unsupported disturbance type: {disturbance.disturbance_type}")

        obs_id = disturbance.parameters.get("obstacle_id", f"obs_{disturbance.event_id}")
        pos_raw = disturbance.parameters.get("position", {"x": 0.25, "y": 0.0, "z": 0.20})
        dim_raw = disturbance.parameters.get("dimensions", {"length_x": 0.08, "width_y": 0.08, "height_z": 0.20})

        pos = {
            "x": float(pos_raw["x"] if isinstance(pos_raw, dict) else pos_raw.x),
            "y": float(pos_raw["y"] if isinstance(pos_raw, dict) else pos_raw.y),
            "z": float(pos_raw["z"] if isinstance(pos_raw, dict) else pos_raw.z),
        }
        dims = {
            "length_x": float(dim_raw["length_x"] if isinstance(dim_raw, dict) else dim_raw.length_x),
            "width_y": float(dim_raw["width_y"] if isinstance(dim_raw, dict) else dim_raw.width_y),
            "height_z": float(dim_raw["height_z"] if isinstance(dim_raw, dict) else dim_raw.height_z),
        }

        usd_path = f"/World/Obstacles/{obs_id}"

        return IsaacSimDisturbance(
            event_id=disturbance.event_id,
            disturbance_type=disturbance.disturbance_type.value,
            usd_prim_path=usd_path,
            usd_prim_type="Cube",
            position=pos,
            dimensions=dims,
            is_active=True,
            physics_enabled=True,
        )

    # -----------------------------------------------------------------------
    # World State Translation (Isaac Sim Raw State -> SĀRTHI WorldState)
    # -----------------------------------------------------------------------

    @classmethod
    def translate_world_state(cls, raw_state: Union[WorldState, Dict[str, Any]]) -> WorldState:
        """
        Translates raw Isaac Sim telemetry into the canonical, immutable SĀRTHI WorldState.
        Never duplicates the WorldState model schema.
        """
        if isinstance(raw_state, WorldState):
            return raw_state

        if not isinstance(raw_state, dict):
            raise TypeError(f"Expected dict or WorldState, got {type(raw_state).__name__}")

        # 1. Translate Robot State
        raw_robot = raw_state.get("robot", {})
        pos_r = raw_robot.get("position", {"x": 0.0, "y": 0.0, "z": 0.20})
        robot_state = RobotState(
            position=Point3D(x=pos_r["x"], y=pos_r["y"], z=pos_r["z"]),
            gripper_open=bool(raw_robot.get("gripper_open", True)),
            holding_object_id=raw_robot.get("holding_object_id"),
            payload_mass_kg=float(raw_robot.get("payload_mass_kg", 0.0)),
            is_moving=bool(raw_robot.get("is_moving", False)),
            max_payload_kg=float(raw_robot.get("max_payload_kg", 3.0)),
            max_reach_m=float(raw_robot.get("max_reach_m", 0.85)),
        )

        # 2. Translate Objects & Obstacles
        objects_list: List[WorldObject] = []
        for obj_data in raw_state.get("objects", []):
            pos_o = obj_data["position"]
            raw_state_str = obj_data.get("state", "FREE")
            state_enum = ObjectState(raw_state_str) if isinstance(raw_state_str, str) else raw_state_str

            objects_list.append(WorldObject(
                id=str(obj_data["id"]),
                name=str(obj_data.get("name", obj_data["id"])),
                position=Point3D(x=pos_o["x"], y=pos_o["y"], z=pos_o["z"]),
                bounding_radius_m=float(obj_data.get("bounding_radius_m", 0.05)),
                mass_kg=float(obj_data.get("mass_kg", 0.5)),
                state=state_enum,
                is_target=bool(obj_data.get("is_target", False)),
                is_obstacle=bool(obj_data.get("is_obstacle", False)),
            ))

        # 3. Translate Target Zone
        raw_target = raw_state.get("target", {})
        pos_t = raw_target.get("position", {"x": 0.40, "y": -0.20, "z": 0.20})
        target_zone = TargetZone(
            id=str(raw_target.get("id", "blue_target_zone")),
            position=Point3D(x=pos_t["x"], y=pos_t["y"], z=pos_t["z"]),
            tolerance_radius_m=float(raw_target.get("tolerance_radius_m", 0.06)),
        )

        # 4. Translate Environment
        raw_env = raw_state.get("environment", {})
        env_state = EnvironmentState(
            min_x=float(raw_env.get("min_x", -0.8)),
            max_x=float(raw_env.get("max_x", 0.8)),
            min_y=float(raw_env.get("min_y", -0.8)),
            max_y=float(raw_env.get("max_y", 0.8)),
            min_z=float(raw_env.get("min_z", 0.0)),
            max_z=float(raw_env.get("max_z", 1.2)),
            dynamic_obstacles_detected=bool(raw_env.get("dynamic_obstacles_detected", False)),
            slip_risk_level=float(raw_env.get("slip_risk_level", 0.0)),
            friction_coefficient=float(raw_env.get("friction_coefficient", 0.6)),
        )

        # 5. Translate Last Action Outcome
        raw_outcome = raw_state.get("last_action_outcome", {})
        act_type_raw = raw_outcome.get("action_type")
        act_type_enum = ActionType(act_type_raw) if act_type_raw else None
        status_raw = raw_outcome.get("status", "NONE")
        status_enum = LastActionStatus(status_raw) if isinstance(status_raw, str) else status_raw

        outcome = LastActionOutcome(
            action_type=act_type_enum,
            status=status_enum,
            error_message=raw_outcome.get("error_message"),
            contact_force_delta=float(raw_outcome.get("contact_force_delta", 0.0)),
        )

        # 6. Task Objective
        obj_raw = raw_state.get("task_objective", TaskObjective.PICK_AND_PLACE)
        obj_enum = TaskObjective(obj_raw) if isinstance(obj_raw, str) else obj_raw

        return WorldState(
            version=raw_state.get("version", 1),
            timestamp_ns=int(raw_state.get("timestamp_ns", 0)),
            robot=robot_state,
            objects=objects_list,
            target=target_zone,
            environment=env_state,
            task_objective=obj_enum,
            active_constraints=[],
            last_action_outcome=outcome,
        )

    # -----------------------------------------------------------------------
    # SimulationAdapter Protocol Implementation
    # -----------------------------------------------------------------------

    def get_world_state(self) -> WorldState:
        """
        Queries simulator and returns current physical state as structured SĀRTHI WorldState.
        Requires active backend or live Isaac Sim runtime.
        """
        if self._backend is None:
            require_isaac_sim()
            # If Isaac Sim were available, runtime code would query stage prims here
            raise NotImplementedError("Isaac Sim live stage query is pending GPU environment.")

        raw_state = self._backend.get_world_state()
        return self.translate_world_state(raw_state)

    def execute_action(
        self,
        action: Union[CandidateAction, Dict[str, Any]],
    ) -> ActionExecutionResult:
        """
        Translates candidate action and dispatches it to Isaac Sim controllers.
        Requires active backend or live Isaac Sim runtime.
        """
        if self._backend is None:
            require_isaac_sim()
            raise NotImplementedError("Isaac Sim controller execution is pending GPU environment.")

        # Translate SĀRTHI candidate action to Isaac Sim representation
        isaac_action = self.translate_action(action)
        result = self._backend.execute_action(isaac_action)
        if isinstance(result, ActionExecutionResult):
            self._execution_history.append(result)
            return result
        elif isinstance(result, dict):
            parsed_result = ActionExecutionResult.model_validate(result)
            self._execution_history.append(parsed_result)
            return parsed_result
        else:
            raise TypeError(f"Backend returned unexpected result type: {type(result).__name__}")

    def inject_disturbance(self, disturbance: DisturbanceEvent) -> bool:
        """
        Injects environmental disturbance (e.g. PATH_BLOCKED obstacle) into USD stage.
        Requires active backend or live Isaac Sim runtime.
        """
        if self._backend is None:
            require_isaac_sim()
            raise NotImplementedError("Isaac Sim disturbance injection is pending GPU environment.")

        isaac_dist = self.translate_disturbance(disturbance)
        return bool(self._backend.inject_disturbance(isaac_dist))

    def verify_action_result(
        self,
        action: Union[CandidateAction, Dict[str, Any]],
        expected_outcome: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Verifies that the executed action achieved expected physical state in the simulator.
        Requires active backend or live Isaac Sim runtime.
        """
        if self._backend is None:
            require_isaac_sim()
            raise NotImplementedError("Isaac Sim verification is pending GPU environment.")

        return bool(self._backend.verify_action_result(action, expected_outcome))
