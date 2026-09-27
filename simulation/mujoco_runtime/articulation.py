"""
SĀRTHI MuJoCo Runtime — Franka Panda Articulation Controller & IK.

Provides a clean, narrow articulation and controller interface for the Franka Emika
Panda manipulator in MuJoCo:
- Damped-least-squares differential inverse kinematics (DLS-IK) for Cartesian position targets
- Actual 7-DOF Panda arm joint and actuator discovery and control
- Gripper actuation via actuator8 and the split tendon mechanism
- Deterministic physics stepping in controlled ticks
- Immediate safety STOP / hold functionality
- Clean CandidateAction dispatch boundary

Strict Architectural Guarantees:
- Contains NO cognitive, replanning, or decision-making logic; SĀRTHI Decision Engine
  remains the sole physical-action authority.
- The Panda moves exclusively through its actual joints and actuators. Direct body
  teleportation is strictly forbidden.
- Offline headless CPU execution; no viewer, GPU, or Isaac Sim required.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from backend.app.decision_engine.models import (
    ActionType,
    CandidateAction,
    Point3D,
)
from simulation.core.events import ActionExecutionResult
from simulation.mujoco_runtime import require_mujoco


@dataclass
class MuJoCoIKConfig:
    """Configuration parameters for Damped-Least-Squares Inverse Kinematics."""
    damping: float = 0.05
    step_size: float = 0.5
    max_joint_delta: float = 0.1  # Maximum joint delta per iteration (radians)
    max_joint_velocity: float = 2.0  # Maximum joint velocity limit (rad/s)
    convergence_tolerance: float = 0.005  # Cartesian tolerance in meters (5mm)
    max_iterations: int = 100


class SarthiMuJoCoArticulation:
    """
    Dedicated articulation controller for the Franka Emika Panda in MuJoCo.
    Executes high-level candidate actions selected by the SĀRTHI Decision Engine.
    """

    ARM_JOINT_NAMES = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]

    ARM_ACTUATOR_NAMES = [
        "actuator1",
        "actuator2",
        "actuator3",
        "actuator4",
        "actuator5",
        "actuator6",
        "actuator7",
    ]

    GRIPPER_ACTUATOR_NAME = "actuator8"
    HAND_BODY_NAME = "hand"
    LEFT_FINGER_NAME = "left_finger"
    RIGHT_FINGER_NAME = "right_finger"

    # End-effector grasp center offset relative to the 'hand' body frame:
    # 0.1034m along local +Z axis aligns with the midpoint of the finger pads.
    GRASP_CENTER_OFFSET = np.array([0.0, 0.0, 0.1034], dtype=np.float64)

    # Gripper control range from panda.xml (ctrlrange="0 255")
    GRIPPER_OPEN_CTRL = 255.0
    GRIPPER_CLOSED_CTRL = 0.0

    def __init__(
        self,
        model: Any,
        data: Any,
        entity_ids: Optional[Dict[str, Dict[str, int]]] = None,
        ik_config: Optional[MuJoCoIKConfig] = None,
    ):
        require_mujoco()
        import mujoco

        self.model = model
        self.data = data
        self.ik_config = ik_config or MuJoCoIKConfig()

        # Controller states
        self._is_stopped: bool = False
        self._gripper_state: str = "OPEN"
        self._is_holding_object: bool = False
        self._holding_object_id: Optional[str] = None
        self._target_joint_positions: Optional[np.ndarray] = None

        # Discover & cache MuJoCo entity indices
        self._discover_entities(entity_ids)

        # Allocate reusable scratch data for IK to avoid mutating primary simulation state
        self._ik_data = mujoco.MjData(self.model)

    def _discover_entities(self, entity_ids: Optional[Dict[str, Dict[str, int]]]) -> None:
        """Discovers and caches joint, actuator, and body IDs in the compiled model."""
        import mujoco

        # Arm Joints
        self._arm_joint_ids: List[int] = []
        self._arm_qposadr: List[int] = []
        self._arm_dofadr: List[int] = []
        self._arm_joint_limits: List[Tuple[float, float]] = []

        for name in self.ARM_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Required Franka Panda joint '{name}' not found in MuJoCo model.")
            self._arm_joint_ids.append(jid)
            self._arm_qposadr.append(int(self.model.jnt_qposadr[jid]))
            self._arm_dofadr.append(int(self.model.jnt_dofadr[jid]))
            j_range = self.model.jnt_range[jid]
            self._arm_joint_limits.append((float(j_range[0]), float(j_range[1])))

        # Arm Actuators
        self._arm_actuator_ids: List[int] = []
        for name in self.ARM_ACTUATOR_NAMES:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise ValueError(f"Required Franka Panda actuator '{name}' not found in MuJoCo model.")
            self._arm_actuator_ids.append(aid)

        # Gripper Actuator
        self._gripper_actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.GRIPPER_ACTUATOR_NAME
        )
        if self._gripper_actuator_id < 0:
            raise ValueError(
                f"Required Franka Panda gripper actuator '{self.GRIPPER_ACTUATOR_NAME}' not found."
            )

        # End-Effector Body
        self._hand_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.HAND_BODY_NAME
        )
        if self._hand_body_id < 0:
            raise ValueError(f"Required Franka Panda body '{self.HAND_BODY_NAME}' not found.")

        # Finger bodies
        self._left_finger_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.LEFT_FINGER_NAME
        )
        self._right_finger_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.RIGHT_FINGER_NAME
        )

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def is_stopped(self) -> bool:
        """True if the robot is in an active safety hold / STOP state."""
        return self._is_stopped

    @property
    def gripper_state(self) -> str:
        """Current gripper state: 'OPEN', 'CLOSED', or 'HOLDING'."""
        return self._gripper_state

    @property
    def is_gripper_closed(self) -> bool:
        """True if gripper is currently in the CLOSED state."""
        return self._gripper_state == "CLOSED"

    @property
    def is_holding_object(self) -> bool:
        """True if gripper is actively holding a physical object."""
        return self._is_holding_object

    @property
    def holding_object_id(self) -> Optional[str]:
        """ID of currently held object, if any."""
        return self._holding_object_id

    @property
    def joint_names(self) -> List[str]:
        """Names of the 7 articulated arm joints."""
        return list(self.ARM_JOINT_NAMES)

    @property
    def actuator_names(self) -> List[str]:
        """Names of the 7 arm actuators."""
        return list(self.ARM_ACTUATOR_NAMES)

    @property
    def joint_limits(self) -> List[Tuple[float, float]]:
        """Min/max angle limits for each arm joint."""
        return list(self._arm_joint_limits)

    # -----------------------------------------------------------------------
    # Joint & Pose Telemetry
    # -----------------------------------------------------------------------

    def get_joint_positions(self) -> List[float]:
        """
        Returns the current joint angles of the 7 arm joints in radians.
        Guarantees all values are finite.
        """
        positions = [float(self.data.qpos[adr]) for adr in self._arm_qposadr]
        for idx, val in enumerate(positions):
            if not math.isfinite(val):
                raise ValueError(f"Non-finite joint position detected at joint {idx + 1}: {val}")
        return positions

    def get_joint_velocities(self) -> List[float]:
        """
        Returns the current joint velocities of the 7 arm joints in rad/s.
        If stopped, returns all zeros.
        """
        if self._is_stopped:
            return [0.0] * 7
        velocities = [float(self.data.qvel[adr]) for adr in self._arm_dofadr]
        for idx, val in enumerate(velocities):
            if not math.isfinite(val):
                raise ValueError(f"Non-finite joint velocity detected at joint {idx + 1}: {val}")
        return velocities

    def get_end_effector_position(self) -> Point3D:
        """
        Computes the Cartesian world coordinates of the Franka Panda end-effector grasp center.
        Evaluated from the 'hand' body frame position and orientation with calibrated offset.
        """
        hand_pos = self.data.xpos[self._hand_body_id]
        hand_mat = self.data.xmat[self._hand_body_id].reshape(3, 3)
        ee_pos = hand_pos + hand_mat @ self.GRASP_CENTER_OFFSET

        if not np.all(np.isfinite(ee_pos)):
            raise ValueError(f"Non-finite end-effector coordinates computed: {ee_pos}")

        return Point3D(x=float(ee_pos[0]), y=float(ee_pos[1]), z=float(ee_pos[2]))

    def get_end_effector_pose(self) -> Tuple[Point3D, Tuple[float, float, float, float]]:
        """
        Returns the end-effector (position, orientation_quaternion) in world frame.
        """
        pos = self.get_end_effector_position()
        quat = self.data.xquat[self._hand_body_id]
        return pos, (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

    def is_at_position(
        self,
        target_pos: Union[Point3D, Dict[str, float], Sequence[float]],
        tolerance_m: float = 0.05,
    ) -> bool:
        """
        Verifies if the measured end-effector is within Cartesian distance tolerance of target.
        """
        if isinstance(target_pos, dict):
            tx, ty, tz = float(target_pos["x"]), float(target_pos["y"]), float(target_pos["z"])
        elif isinstance(target_pos, Point3D):
            tx, ty, tz = target_pos.x, target_pos.y, target_pos.z
        elif isinstance(target_pos, (list, tuple, np.ndarray)) and len(target_pos) >= 3:
            tx, ty, tz = float(target_pos[0]), float(target_pos[1]), float(target_pos[2])
        else:
            return False

        ee = self.get_end_effector_position()
        dist = math.sqrt((ee.x - tx) ** 2 + (ee.y - ty) ** 2 + (ee.z - tz) ** 2)
        return dist <= tolerance_m

    def get_articulation_state(self) -> Dict[str, Any]:
        """Compiles complete articulation telemetry."""
        return {
            "joint_positions": self.get_joint_positions(),
            "joint_velocities": self.get_joint_velocities(),
            "end_effector_position": self.get_end_effector_position(),
            "gripper_state": self.gripper_state,
            "is_gripper_closed": self.is_gripper_closed,
            "is_holding_object": self.is_holding_object,
            "holding_object_id": self.holding_object_id,
            "is_stopped": self.is_stopped,
        }

    # -----------------------------------------------------------------------
    # Gripper Control
    # -----------------------------------------------------------------------

    def command_gripper(self, command: str) -> None:
        """
        Commands the Franka Panda parallel gripper actuator.
        Supported commands: 'OPEN', 'CLOSE', 'HOLD' (case-insensitive).
        """
        if self._is_stopped:
            raise RuntimeError("Cannot command gripper while articulation is in STOP state.")

        cmd = command.strip().upper()
        if cmd == "OPEN":
            self.data.ctrl[self._gripper_actuator_id] = self.GRIPPER_OPEN_CTRL
            self._gripper_state = "OPEN"
            self._is_holding_object = False
            self._holding_object_id = None
        elif cmd == "CLOSE":
            self.data.ctrl[self._gripper_actuator_id] = self.GRIPPER_CLOSED_CTRL
            self._gripper_state = "CLOSED"
        elif cmd == "HOLD":
            self._gripper_state = "HOLDING"
        else:
            raise ValueError(f"Unsupported gripper command '{command}'. Expected OPEN, CLOSE, or HOLD.")

    def open_gripper(self) -> None:
        """Opens the Franka Panda gripper fingers."""
        self.command_gripper("OPEN")

    def close_gripper(self) -> None:
        """Closes the Franka Panda gripper fingers."""
        self.command_gripper("CLOSE")

    def hold_gripper(self) -> None:
        """Maintains current gripper finger separation."""
        self.command_gripper("HOLD")

    # -----------------------------------------------------------------------
    # Inverse Kinematics (Damped Least Squares)
    # -----------------------------------------------------------------------

    def solve_ik(
        self,
        target_position: Union[Point3D, Dict[str, float], Sequence[float]],
        max_iters: Optional[int] = None,
        tolerance: Optional[float] = None,
        damping: Optional[float] = None,
        initial_q: Optional[Sequence[float]] = None,
        align_downward: bool = True,
    ) -> Tuple[bool, List[float], float]:
        """
        Computes 7-DOF arm joint angles to reach a Cartesian target using DLS-IK.
        
        Formula:
            Delta q = J_p^T * (J_p * J_p^T + lambda^2 * I)^(-1) * error
        
        Args:
            target_position: Desired 3D Cartesian coordinates.
            max_iters: Max iterations (defaults to ik_config.max_iterations).
            tolerance: Convergence tolerance in meters (defaults to ik_config.convergence_tolerance).
            damping: Damping factor lambda (defaults to ik_config.damping).
            initial_q: Initial arm joint configuration. If None, uses current arm qpos.
            align_downward: If True, augments DLS formulation with vertical downward orientation constraint.

        Returns:
            Tuple of (converged: bool, joint_positions: List[float], final_error_distance: float)
        """
        import mujoco

        # Parse & validate target position
        if isinstance(target_position, dict):
            tx, ty, tz = float(target_position["x"]), float(target_position["y"]), float(target_position["z"])
        elif isinstance(target_position, Point3D):
            tx, ty, tz = target_position.x, target_position.y, target_position.z
        elif isinstance(target_position, (list, tuple, np.ndarray)) and len(target_position) >= 3:
            tx, ty, tz = float(target_position[0]), float(target_position[1]), float(target_position[2])
        else:
            raise TypeError(f"Invalid target_position type: {type(target_position).__name__}")

        target = np.array([tx, ty, tz], dtype=np.float64)
        if not np.all(np.isfinite(target)):
            return False, self.get_joint_positions(), float("inf")

        # Configuration parameters
        cfg_max_iters = max_iters if max_iters is not None else self.ik_config.max_iterations
        cfg_tol = tolerance if tolerance is not None else self.ik_config.convergence_tolerance
        cfg_damping = damping if damping is not None else self.ik_config.damping
        step_size = self.ik_config.step_size
        max_delta = self.ik_config.max_joint_delta

        # Initialize scratch state
        self._ik_data.qpos[:] = self.data.qpos[:]
        self._ik_data.qvel[:] = 0.0

        if initial_q is not None:
            for idx, adr in enumerate(self._arm_qposadr):
                self._ik_data.qpos[adr] = float(initial_q[idx])

        mujoco.mj_forward(self.model, self._ik_data)

        best_dist = float("inf")
        best_q = [float(self._ik_data.qpos[adr]) for adr in self._arm_qposadr]

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        lambda_sq_eye = (cfg_damping ** 2) * np.eye(3, dtype=np.float64)
        z_des = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        eye6 = np.eye(6, dtype=np.float64)
        lambda_sq_eye6 = (cfg_damping ** 2) * eye6

        for _ in range(cfg_max_iters):
            # Compute current end-effector position in scratch state
            h_pos = self._ik_data.xpos[self._hand_body_id]
            h_mat = self._ik_data.xmat[self._hand_body_id].reshape(3, 3)
            current_ee = h_pos + h_mat @ self.GRASP_CENTER_OFFSET

            err = target - current_ee
            dist = float(np.linalg.norm(err))

            if dist < best_dist:
                best_dist = dist
                best_q = [float(self._ik_data.qpos[adr]) for adr in self._arm_qposadr]

            # Compute Jacobian for the grasp point
            mujoco.mj_jac(self.model, self._ik_data, jacp, jacr, current_ee, self._hand_body_id)
            J_arm = jacp[:, self._arm_dofadr]  # shape (3, 7)

            if align_downward:
                z_curr = h_mat[:, 2]
                err_rot = np.cross(z_curr, z_des)
                rot_err = float(np.linalg.norm(err_rot))
                if dist <= cfg_tol and rot_err <= 0.15:
                    return True, [float(self._ik_data.qpos[adr]) for adr in self._arm_qposadr], dist
                Jr_arm = jacr[:, self._arm_dofadr]
                J_6d = np.vstack([J_arm, Jr_arm])
                err_6d = np.concatenate([err, 0.8 * err_rot])
                try:
                    dq = J_6d.T @ np.linalg.solve(J_6d @ J_6d.T + lambda_sq_eye6, err_6d)
                except np.linalg.LinAlgError:
                    break
            else:
                if dist <= cfg_tol:
                    return True, [float(self._ik_data.qpos[adr]) for adr in self._arm_qposadr], dist
                try:
                    dq = J_arm.T @ np.linalg.solve(J_arm @ J_arm.T + lambda_sq_eye, err)
                except np.linalg.LinAlgError:
                    break

            if not np.all(np.isfinite(dq)):
                break

            # Limit max joint displacement per step
            max_val = np.max(np.abs(dq))
            if max_val > max_delta:
                dq = dq * (max_delta / max_val)

            # Apply candidate update clamped to joint limits
            for idx, adr in enumerate(self._arm_qposadr):
                q_curr = self._ik_data.qpos[adr]
                q_next = q_curr + dq[idx] * step_size
                min_lim, max_lim = self._arm_joint_limits[idx]
                self._ik_data.qpos[adr] = np.clip(q_next, min_lim, max_lim)

            mujoco.mj_forward(self.model, self._ik_data)

        # Final check if best achieved error is within tolerance
        if best_dist <= cfg_tol:
            return True, best_q, best_dist

        return False, best_q, best_dist

    # -----------------------------------------------------------------------
    # Arm Actuator Control & Stepping
    # -----------------------------------------------------------------------

    def command_joint_positions(self, joint_positions: Sequence[float]) -> None:
        """
        Commands the 7 Franka arm actuators with safety checks:
        - Rejects commands if stopped
        - Rejects non-finite values
        - Enforces joint-limit clamping
        """
        if self._is_stopped:
            raise RuntimeError("Cannot command joint positions while articulation is in STOP state.")

        if len(joint_positions) != 7:
            raise ValueError(f"Expected 7 joint positions, got {len(joint_positions)}.")

        clamped = []
        for idx, pos in enumerate(joint_positions):
            val = float(pos)
            if not math.isfinite(val):
                raise ValueError(f"Non-finite joint target at joint {idx + 1}: {val}")
            min_lim, max_lim = self._arm_joint_limits[idx]
            clamped.append(max(min_lim, min(val, max_lim)))

        for idx, aid in enumerate(self._arm_actuator_ids):
            self.data.ctrl[aid] = clamped[idx]

        self._target_joint_positions = np.array(clamped, dtype=np.float64)

    def move_to_position(
        self,
        target_position: Union[Point3D, Dict[str, float], Sequence[float]],
        speed_scale: float = 1.0,
        steps: int = 0,
        tolerance_m: float = 0.05,
    ) -> bool:
        """
        Moves the Franka end-effector toward a Cartesian target:
        1. Solves DLS-IK for the target.
        2. Commands the 7 arm actuators.
        3. Optionally advances physics for a specified number of simulation ticks.
        
        Returns:
            True if IK succeeded (and if steps > 0, whether target was reached).
        """
        if self._is_stopped:
            raise RuntimeError("Cannot command motion while articulation is in STOP state.")

        converged, solved_q, dist = self.solve_ik(target_position)
        if not converged:
            return False

        self.command_joint_positions(solved_q)

        if steps > 0:
            for _ in range(steps):
                self.step(1)
                if self.is_at_position(target_position, tolerance_m=tolerance_m):
                    return True
            return self.is_at_position(target_position, tolerance_m=tolerance_m)

        return True

    def step(self, n_steps: int = 1) -> None:
        """
        Advances the MuJoCo physics simulation in deterministic ticks.
        Guarantees that state values remain finite after each step.
        """
        import mujoco

        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

        # Safety verification: ensure finite state
        for adr in self._arm_qposadr:
            if not math.isfinite(self.data.qpos[adr]):
                raise FloatingPointError("MuJoCo simulation produced non-finite joint position.")

    # -----------------------------------------------------------------------
    # Safety Hold & STOP
    # -----------------------------------------------------------------------

    def stop(self) -> None:
        """
        Immediately puts the robot into a safe hold state:
        - Locks actuator targets to the current joint positions
        - Zeros all joint velocities
        - Freezes gripper state
        - Enters safety stopped state
        """
        self._is_stopped = True

        # Command current joint positions as static hold targets
        for idx, aid in enumerate(self._arm_actuator_ids):
            q_curr = float(self.data.qpos[self._arm_qposadr[idx]])
            self.data.ctrl[aid] = q_curr

        # Zero velocities
        for adr in self._arm_dofadr:
            self.data.qvel[adr] = 0.0

        if self._gripper_state != "CLOSED":
            self._gripper_state = "HOLDING"

    def resume(self) -> None:
        """Clears the safety stopped state, allowing subsequent commanded motions."""
        self._is_stopped = False

    # -----------------------------------------------------------------------
    # CandidateAction Execution Boundary
    # -----------------------------------------------------------------------

    def dispatch_action(
        self,
        action: Union[CandidateAction, Dict[str, Any]],
        sim_time: float = 0.0,
        world_version: int = 1,
    ) -> ActionExecutionResult:
        """
        Executes an action primitive commanded by the SĀRTHI Decision Engine.
        Strict execution boundary: contains NO cognitive or recovery logic.
        """
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
            raise TypeError(f"Unsupported action object type: {type(action).__name__}")

        # STOP action
        if action_type == ActionType.STOP:
            self.stop()
            return ActionExecutionResult(
                success=True,
                action_type=ActionType.STOP.value,
                action_id=action_id,
                simulation_time=sim_time,
                previous_world_state_version=world_version,
                new_world_state_version=world_version + 1,
                failure_reason=None,
                details={"is_stopped": True},
            )

        # Guard against commanding motion while stopped
        if self._is_stopped:
            return ActionExecutionResult(
                success=False,
                action_type=action_type.value,
                action_id=action_id,
                simulation_time=sim_time,
                previous_world_state_version=world_version,
                new_world_state_version=world_version,
                failure_reason=f"Cannot execute {action_type.value}: articulation is in safety STOP state.",
                details={"error_code": "ROBOT_STOPPED"},
            )

        # APPROACH: Open gripper and move toward target Cartesian position
        if action_type == ActionType.APPROACH:
            self.open_gripper()
            if target_pos:
                converged, solved_q, dist = self.solve_ik(target_pos)
                if not converged:
                    return ActionExecutionResult(
                        success=False,
                        action_type=ActionType.APPROACH.value,
                        action_id=action_id,
                        simulation_time=sim_time,
                        previous_world_state_version=world_version,
                        new_world_state_version=world_version,
                        failure_reason=f"IK failed to converge for APPROACH target. Error distance: {dist:.4f}m",
                        details={"error_code": "IK_FAILED", "error_distance_m": dist},
                    )
                self.command_joint_positions(solved_q)

            return ActionExecutionResult(
                success=True,
                action_type=ActionType.APPROACH.value,
                action_id=action_id,
                simulation_time=sim_time,
                previous_world_state_version=world_version,
                new_world_state_version=world_version + 1,
                failure_reason=None,
                details={"target_position": target_pos, "speed_scale": speed},
            )

        # REPOSITION: Hold gripper and move toward recovery target
        if action_type == ActionType.REPOSITION:
            self.hold_gripper()
            if target_pos:
                converged, solved_q, dist = self.solve_ik(target_pos)
                if not converged:
                    return ActionExecutionResult(
                        success=False,
                        action_type=ActionType.REPOSITION.value,
                        action_id=action_id,
                        simulation_time=sim_time,
                        previous_world_state_version=world_version,
                        new_world_state_version=world_version,
                        failure_reason=f"IK failed to converge for REPOSITION target. Error distance: {dist:.4f}m",
                        details={"error_code": "IK_FAILED", "error_distance_m": dist},
                    )
                self.command_joint_positions(solved_q)

            return ActionExecutionResult(
                success=True,
                action_type=ActionType.REPOSITION.value,
                action_id=action_id,
                simulation_time=sim_time,
                previous_world_state_version=world_version,
                new_world_state_version=world_version + 1,
                failure_reason=None,
                details={"target_position": target_pos, "speed_scale": speed},
            )

        # MOVE: Move payload toward destination
        if action_type == ActionType.MOVE:
            if target_pos:
                converged, solved_q, dist = self.solve_ik(target_pos)
                if not converged:
                    return ActionExecutionResult(
                        success=False,
                        action_type=ActionType.MOVE.value,
                        action_id=action_id,
                        simulation_time=sim_time,
                        previous_world_state_version=world_version,
                        new_world_state_version=world_version,
                        failure_reason=f"IK failed to converge for MOVE target. Error distance: {dist:.4f}m",
                        details={"error_code": "IK_FAILED", "error_distance_m": dist},
                    )
                self.command_joint_positions(solved_q)

            return ActionExecutionResult(
                success=True,
                action_type=ActionType.MOVE.value,
                action_id=action_id,
                simulation_time=sim_time,
                previous_world_state_version=world_version,
                new_world_state_version=world_version + 1,
                failure_reason=None,
                details={"target_position": target_pos, "speed_scale": speed},
            )

        # GRASP: Close gripper to grasp payload
        if action_type == ActionType.GRASP:
            if not target_obj or not str(target_obj).strip():
                return ActionExecutionResult(
                    success=False,
                    action_type=ActionType.GRASP.value,
                    action_id=action_id,
                    simulation_time=sim_time,
                    previous_world_state_version=world_version,
                    new_world_state_version=world_version,
                    failure_reason="Cannot execute GRASP: missing target_object_id.",
                    details={"error_code": "MISSING_PAYLOAD"},
                )

            # Reach/proximity check if grasp position provided
            if target_pos:
                grasp_threshold_m = 0.15
                if not self.is_at_position(target_pos, tolerance_m=grasp_threshold_m):
                    current_ee = self.get_end_effector_position()
                    if isinstance(target_pos, dict):
                        gx, gy, gz = float(target_pos["x"]), float(target_pos["y"]), float(target_pos["z"])
                    elif isinstance(target_pos, Point3D):
                        gx, gy, gz = target_pos.x, target_pos.y, target_pos.z
                    else:
                        gx, gy, gz = float(target_pos[0]), float(target_pos[1]), float(target_pos[2])
                    dist = math.sqrt((current_ee.x - gx) ** 2 + (current_ee.y - gy) ** 2 + (current_ee.z - gz) ** 2)
                    return ActionExecutionResult(
                        success=False,
                        action_type=ActionType.GRASP.value,
                        action_id=action_id,
                        simulation_time=sim_time,
                        previous_world_state_version=world_version,
                        new_world_state_version=world_version,
                        failure_reason=(
                            f"Cannot execute GRASP: Object '{target_obj}' is outside grasp reach "
                            f"(distance {dist:.3f}m > threshold {grasp_threshold_m:.3f}m)."
                        ),
                        details={"error_code": "OBJECT_OUT_OF_REACH", "distance_m": dist},
                    )

            self.close_gripper()
            self._is_holding_object = True
            self._holding_object_id = str(target_obj)
            return ActionExecutionResult(
                success=True,
                action_type=ActionType.GRASP.value,
                action_id=action_id,
                simulation_time=sim_time,
                previous_world_state_version=world_version,
                new_world_state_version=world_version + 1,
                failure_reason=None,
                details={"target_object_id": str(target_obj), "force_n": force},
            )

        # RELEASE: Open gripper to release payload
        if action_type == ActionType.RELEASE:
            self.open_gripper()
            return ActionExecutionResult(
                success=True,
                action_type=ActionType.RELEASE.value,
                action_id=action_id,
                simulation_time=sim_time,
                previous_world_state_version=world_version,
                new_world_state_version=world_version + 1,
                failure_reason=None,
                details={"released_object_id": self._holding_object_id},
            )

        raise ValueError(f"Unsupported action type: {action_type}")
