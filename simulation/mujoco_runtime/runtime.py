"""
SĀRTHI MuJoCo Runtime — Physical Simulation Adapter.

Connects the SĀRTHI TaskRunner and Decision Engine to the physical MuJoCo simulation.
Fulfills the abstract SimulationAdapter contract.

Strict Architectural Guarantees:
- Contains NO cognitive or replanning logic; SĀRTHI Decision Engine is the sole physical-action authority.
- The Panda moves exclusively through its actual joints and actuators.
- Never teleports robot, end-effector, or objects.
- Bounded physics stepping with deterministic timeouts.
- Uses physical telemetry and contact information for verification.
"""

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from backend.app.decision_engine.models import (
    ActionType,
    CandidateAction,
    LastActionOutcome,
    LastActionStatus,
    Point3D,
    WorldState,
)
from simulation.adapters.base import SimulationAdapter
from simulation.core.events import ActionExecutionResult, DisturbanceEvent, DisturbanceType
from simulation.mujoco_runtime import require_mujoco
from simulation.mujoco_runtime.articulation import (
    MuJoCoIKConfig,
    SarthiMuJoCoArticulation,
)
from simulation.mujoco_runtime.scene_builder import SarthiMuJoCoSceneBuilder
from simulation.mujoco_runtime.state_reader import SarthiMuJoCoStateReader
from simulation.mujoco_runtime.verifier import SarthiMuJoCoVerifier
from simulation.scenarios.tabletop_pick_place import (
    TabletopPickPlaceScenario,
    create_default_scenario,
)


class SarthiMuJoCoRuntime(SimulationAdapter):
    """
    Concrete physical simulation adapter bridging the SĀRTHI architecture
    with the MuJoCo physics engine.
    """

    # Inactive staged coordinates for obstacle when disturbance is not active
    INACTIVE_OBSTACLE_POS = [0.0, 0.0, -10.0]

    # Canonical active obstacle blocking coordinates (matches TabletopPickPlaceScenario)
    ACTIVE_OBSTACLE_POS = [0.325, -0.025, 0.12]

    def __init__(
        self,
        scene_builder: Optional[SarthiMuJoCoSceneBuilder] = None,
        scenario: Optional[TabletopPickPlaceScenario] = None,
        with_disturbance: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ):
        require_mujoco()
        import mujoco

        self._scenario = scenario or create_default_scenario()
        self._scene_builder = scene_builder or SarthiMuJoCoSceneBuilder(self._scenario)
        self._config = config or {}

        # Build MuJoCo model and data
        self._model, self._data = self._scene_builder.build()

        # Step configuration
        self._timestep: float = float(self._model.opt.timestep)
        self._steps_per_action: int = int(self._config.get("steps_per_action", 350))
        self._maximum_execution_ticks: int = int(self._config.get("max_execution_ticks", 500))
        self._convergence_tolerance: float = float(self._config.get("convergence_tolerance", 0.03))
        self._action_timeout_s: float = float(self._config.get("action_timeout_s", 2.0))

        # Reset Panda to home keyframe while preserving initial object position
        self._reset_to_home_keyframe()

        # Obstacle activation state
        self._obstacle_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "obstacle")
        self._is_disturbance_active: bool = with_disturbance
        if not with_disturbance and self._obstacle_id >= 0:
            self._model.body_pos[self._obstacle_id] = self.INACTIVE_OBSTACLE_POS
            mujoco.mj_forward(self._model, self._data)

        # Initialize articulation controller, state reader, and verifier
        self._articulation = SarthiMuJoCoArticulation(self._model, self._data)
        self._state_reader = SarthiMuJoCoStateReader(self._model, self._data, scenario=self._scenario)
        self._verifier = SarthiMuJoCoVerifier(self._state_reader)

        # Simulation lifecycle tracking
        self._world_state_version: int = 1
        self._last_action_outcome: Optional[LastActionOutcome] = None
        self._execution_history: List[ActionExecutionResult] = []

    def _reset_to_home_keyframe(self) -> None:
        """Resets the robot to the canonical 'home' keyframe while keeping object poses."""
        import mujoco

        key_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            # Preserve red object initial qpos from scene builder
            obj_qpos = self._data.qpos[9:16].copy()
            mujoco.mj_resetDataKeyframe(self._model, self._data, key_id)
            self._data.qpos[9:16] = obj_qpos
            mujoco.mj_forward(self._model, self._data)

    # -----------------------------------------------------------------------
    # Subsystem Properties
    # -----------------------------------------------------------------------

    @property
    def model(self) -> Any:
        return self._model

    @property
    def data(self) -> Any:
        return self._data

    @property
    def articulation(self) -> SarthiMuJoCoArticulation:
        return self._articulation

    @property
    def state_reader(self) -> SarthiMuJoCoStateReader:
        return self._state_reader

    @property
    def verifier(self) -> SarthiMuJoCoVerifier:
        return self._verifier

    @property
    def simulation_time(self) -> float:
        return float(self._data.time)

    @property
    def world_state_version(self) -> int:
        return self._world_state_version

    @property
    def is_disturbance_active(self) -> bool:
        return self._is_disturbance_active

    @property
    def execution_history(self) -> List[ActionExecutionResult]:
        return list(self._execution_history)

    # -----------------------------------------------------------------------
    # SimulationAdapter Contract: get_world_state
    # -----------------------------------------------------------------------

    def get_world_state(self) -> WorldState:
        """
        Fetches the current physical state of the simulation as a canonical SĀRTHI WorldState.
        """
        return self._state_reader.read_world_state(
            version=self._world_state_version,
            last_action_outcome=self._last_action_outcome,
        )

    # -----------------------------------------------------------------------
    # SimulationAdapter Contract: execute_action
    # -----------------------------------------------------------------------

    def execute_action(
        self,
        action: Union[CandidateAction, Dict[str, Any]],
    ) -> ActionExecutionResult:
        """
        Executes a candidate action selected by the Decision Engine on the MuJoCo robot.
        Advances physics in bounded, deterministic ticks and evaluates physical verification.
        """
        start_time = self.simulation_time
        prev_version = self._world_state_version

        # Parse action parameters
        if isinstance(action, dict):
            act_type = action.get("action_type")
            action_type = ActionType(act_type) if not isinstance(act_type, ActionType) else act_type
            action_id = str(action.get("action_id", "act_dyn"))
            target_pos = action.get("target_position")
            target_obj = action.get("target_object_id")
            speed = float(action.get("speed_scale", 1.0))
            force = float(action.get("expected_force_n", 5.0))
        elif isinstance(action, CandidateAction):
            action_type = action.action_type
            action_id = action.action_id
            target_pos = action.target_position
            target_obj = action.target_object_id
            speed = action.speed_scale
            force = action.expected_force_n
        else:
            raise TypeError(f"Unsupported action type: {type(action).__name__}")

        # Check safety hold: if robot is stopped, reject motions
        if self._articulation.is_stopped and action_type != ActionType.STOP:
            fail_reason = "Cannot execute action: articulation is in safety STOP state."
            outcome = LastActionOutcome(
                action_type=action_type,
                status=LastActionStatus.FAILURE,
                error_message=fail_reason,
            )
            self._last_action_outcome = outcome
            res = ActionExecutionResult(
                success=False,
                action_type=action_type.value,
                action_id=action_id,
                simulation_time=self.simulation_time,
                previous_world_state_version=prev_version,
                new_world_state_version=prev_version,
                failure_reason=fail_reason,
                details={"error_code": "ROBOT_STOPPED"},
            )
            self._execution_history.append(res)
            return res

        ticks_executed = 0
        failure_reason = None
        action_success = False

        t_ik_ms = 0.0
        t_sim_ms = 0.0
        t_verify_ms = 0.0

        def _timed_solve_ik(pos):
            nonlocal t_ik_ms
            _t0 = time.perf_counter()
            _res = self._articulation.solve_ik(pos)
            t_ik_ms += (time.perf_counter() - _t0) * 1000.0
            return _res

        def _timed_step_traj(target_q, target_pos, steps):
            nonlocal t_sim_ms
            _t0 = time.perf_counter()
            _ticks = self._step_joint_trajectory(target_q, target_pos, steps=steps)
            t_sim_ms += (time.perf_counter() - _t0) * 1000.0
            return _ticks

        def _timed_step(steps):
            nonlocal t_sim_ms
            _t0 = time.perf_counter()
            self._articulation.step(steps)
            t_sim_ms += (time.perf_counter() - _t0) * 1000.0

        def _timed_verify(act, exp=None):
            nonlocal t_verify_ms
            _t0 = time.perf_counter()
            _res = self._verifier.verify(act, exp)
            t_verify_ms += (time.perf_counter() - _t0) * 1000.0
            return _res

        # --- STOP ---
        if action_type == ActionType.STOP:
            self._articulation.stop()
            _timed_step(25)
            ticks_executed = 25
            action_success, failure_reason = _timed_verify(action)

        # --- APPROACH ---
        elif action_type == ActionType.APPROACH:
            self._articulation.open_gripper()
            _timed_step(30)
            ticks_executed += 30

            if target_pos:
                # Standoff approach to avoid clipping table or object from diagonal approach
                if isinstance(target_pos, Point3D):
                    standoff_pos = Point3D(x=target_pos.x, y=target_pos.y, z=target_pos.z + 0.08)
                elif isinstance(target_pos, dict):
                    standoff_pos = Point3D(x=float(target_pos["x"]), y=float(target_pos["y"]), z=float(target_pos["z"]) + 0.08)
                else:
                    standoff_pos = Point3D(x=float(target_pos[0]), y=float(target_pos[1]), z=float(target_pos[2]) + 0.08)

                st_conv, st_q, _ = _timed_solve_ik(standoff_pos)
                if st_conv:
                    ticks_executed += _timed_step_traj(st_q, standoff_pos, steps=100)

                converged, q_sol, dist = _timed_solve_ik(target_pos)
                if not converged:
                    failure_reason = f"IK failed to converge for APPROACH target. Dist: {dist:.4f}m"
                else:
                    ticks_executed += _timed_step_traj(q_sol, target_pos, steps=250)

            if failure_reason is None:
                action_success, failure_reason = _timed_verify(action)
            else:
                action_success = False

        # --- REPOSITION ---
        elif action_type == ActionType.REPOSITION:
            if self._state_reader.is_object_grasped():
                self._articulation.close_gripper()
            else:
                self._articulation.hold_gripper()

            if target_pos:
                converged, q_sol, dist = _timed_solve_ik(target_pos)
                if not converged:
                    failure_reason = f"IK failed to converge for REPOSITION target. Dist: {dist:.4f}m"
                else:
                    ticks_executed += _timed_step_traj(q_sol, target_pos, steps=self._steps_per_action)

            if failure_reason is None:
                action_success, failure_reason = _timed_verify(action)
            else:
                action_success = False

        # --- MOVE ---
        elif action_type == ActionType.MOVE:
            if target_pos:
                # If holding object at elevated height, perform trapezoidal transit (horizontal first, then lower)
                curr_ee = self._state_reader.get_end_effector_position()
                tgt_z = float(target_pos.z if isinstance(target_pos, Point3D) else (target_pos["z"] if isinstance(target_pos, dict) else target_pos[2]))
                tgt_x = float(target_pos.x if isinstance(target_pos, Point3D) else (target_pos["x"] if isinstance(target_pos, dict) else target_pos[0]))
                tgt_y = float(target_pos.y if isinstance(target_pos, Point3D) else (target_pos["y"] if isinstance(target_pos, dict) else target_pos[1]))

                if curr_ee.z > tgt_z + 0.05 and self._state_reader.is_object_grasped():
                    high_tgt = Point3D(x=tgt_x, y=tgt_y, z=curr_ee.z)
                    h_conv, h_q, _ = _timed_solve_ik(high_tgt)
                    if h_conv:
                        ticks_executed += _timed_step_traj(h_q, high_tgt, steps=300)

                converged, q_sol, dist = _timed_solve_ik(target_pos)
                if not converged:
                    failure_reason = f"IK failed to converge for MOVE target. Dist: {dist:.4f}m"
                else:
                    ticks_executed += _timed_step_traj(q_sol, target_pos, steps=self._steps_per_action)

            if failure_reason is None:
                action_success, failure_reason = _timed_verify(action)
            else:
                action_success = False

        # --- GRASP ---
        elif action_type == ActionType.GRASP:
            if not target_obj or not str(target_obj).strip():
                failure_reason = "Missing target_object_id for GRASP."
            else:
                if target_pos:
                    conv_g, q_g, _ = _timed_solve_ik(target_pos)
                    if conv_g:
                        ticks_executed += _timed_step_traj(q_g, target_pos, steps=150)
                self._articulation.close_gripper()
                # Step physics to allow gripper fingers to physically close on object
                _timed_step(self._steps_per_action)
                ticks_executed += self._steps_per_action
                action_success, failure_reason = _timed_verify(action)

        # --- RELEASE ---
        elif action_type == ActionType.RELEASE:
            self._articulation.open_gripper()
            _timed_step(self._steps_per_action)
            ticks_executed = self._steps_per_action
            action_success, failure_reason = _timed_verify(action)

        else:
            failure_reason = f"Unsupported action type: {action_type}"

        # Determine new world state version
        new_version = prev_version + 1 if action_success else prev_version
        if action_success:
            self._world_state_version = new_version

        # Record outcome
        outcome = LastActionOutcome(
            action_type=action_type,
            status=LastActionStatus.SUCCESS if action_success else LastActionStatus.FAILURE,
            error_message=failure_reason,
        )
        self._last_action_outcome = outcome

        details = {
            "start_time": start_time,
            "end_time": self.simulation_time,
            "physics_ticks": ticks_executed,
            "target_position": target_pos,
            "verified": action_success,
            "tau_ik_ms": round(t_ik_ms, 3),
            "tau_sim_ms": round(t_sim_ms, 3),
            "tau_verify_ms": round(t_verify_ms, 3),
        }

        result = ActionExecutionResult(
            success=action_success,
            action_type=action_type.value,
            action_id=action_id,
            simulation_time=self.simulation_time,
            previous_world_state_version=prev_version,
            new_world_state_version=new_version,
            failure_reason=failure_reason,
            details=details,
        )
        self._execution_history.append(result)
        return result

    def _step_joint_trajectory(
        self,
        target_q: Sequence[float],
        target_pos: Optional[Union[Point3D, Dict[str, float], Any]] = None,
        steps: int = 350,
    ) -> int:
        """
        Executes smooth linear interpolation in joint space from current joint angles
        to target_q, followed by a settling phase holding target_q. Prevents sudden
        acceleration and inertial fling of held objects while guaranteeing precise convergence.
        """
        start_q = np.array(self._articulation.get_joint_positions())
        tgt_q = np.array(target_q)
        total_steps = max(10, min(steps, self._maximum_execution_ticks))
        interp_steps = max(5, int(total_steps * 0.85))
        settle_steps = total_steps - interp_steps

        # 1. Smooth interpolation phase
        for s in range(interp_steps):
            alpha = (s + 1) / interp_steps
            interp_q = (1.0 - alpha) * start_q + alpha * tgt_q
            self._articulation.command_joint_positions(interp_q)
            if self._articulation.gripper_state == "CLOSED":
                self._articulation.close_gripper()
            self._articulation.step(1)

        # 2. Settling phase holding target configuration
        for s in range(settle_steps):
            self._articulation.command_joint_positions(tgt_q)
            if self._articulation.gripper_state == "CLOSED":
                self._articulation.close_gripper()
            self._articulation.step(1)
            if target_pos is not None and s > int(settle_steps * 0.5):
                if self._articulation.is_at_position(target_pos, tolerance_m=self._convergence_tolerance):
                    return interp_steps + s + 1

        return total_steps

    def _step_until_reached(
        self,
        target_pos: Union[Point3D, Dict[str, float], Any],
        sub_steps: int = 25,
    ) -> int:
        """
        Advances physics in controlled sub-steps until the end effector
        reaches the target within tolerance or maximum ticks is reached.
        """
        ticks = 0
        while ticks < self._maximum_execution_ticks:
            self._articulation.step(sub_steps)
            ticks += sub_steps
            if self._articulation.is_at_position(target_pos, tolerance_m=self._convergence_tolerance):
                break
        return ticks

    # -----------------------------------------------------------------------
    # SimulationAdapter Contract: inject_disturbance
    # -----------------------------------------------------------------------

    def inject_disturbance(
        self,
        disturbance: Union[DisturbanceEvent, str, Dict[str, Any]],
    ) -> bool:
        """
        Physically activates the dynamic path obstacle in the MuJoCo simulation.
        Translates obstacle into the workspace, modifying the physical scene and WorldState.
        """
        import mujoco

        # Parse disturbance type
        if isinstance(disturbance, str):
            dtype = disturbance
            params = {}
        elif isinstance(disturbance, DisturbanceEvent):
            dtype = disturbance.disturbance_type.value if hasattr(disturbance.disturbance_type, "value") else str(disturbance.disturbance_type)
            params = disturbance.parameters
        elif isinstance(disturbance, dict):
            dtype = disturbance.get("disturbance_type", "PATH_BLOCKED")
            params = disturbance.get("parameters", {})
        else:
            return False

        if dtype == "PATH_BLOCKED" or dtype == DisturbanceType.PATH_BLOCKED.value:
            if self._obstacle_id >= 0:
                pos = params.get("position", self.ACTIVE_OBSTACLE_POS)
                if isinstance(pos, dict):
                    pos_coords = [float(pos.get("x", 0.325)), float(pos.get("y", -0.025)), float(pos.get("z", 0.12))]
                elif isinstance(pos, (list, tuple, np.ndarray)):
                    pos_coords = [float(pos[0]), float(pos[1]), float(pos[2])]
                else:
                    pos_coords = list(self.ACTIVE_OBSTACLE_POS)

                # Physically place obstacle in MuJoCo
                self._model.body_pos[self._obstacle_id] = pos_coords
                mujoco.mj_forward(self._model, self._data)

                self._is_disturbance_active = True
                self._world_state_version += 1
                return True

        return False

    # -----------------------------------------------------------------------
    # SimulationAdapter Contract: verify_action_result
    # -----------------------------------------------------------------------

    def verify_action_result(
        self,
        action: Union[CandidateAction, Dict[str, Any]],
        expected_outcome: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Validates that the executed action achieved the expected physical state change.
        """
        _t0 = time.perf_counter()
        success, _ = self._verifier.verify(action, expected_outcome)
        self._last_verify_latency_ms = (time.perf_counter() - _t0) * 1000.0
        return success
