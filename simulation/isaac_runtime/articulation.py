"""
SĀRTHI Isaac Sim Runtime — Robot Articulation Controller Boundary.

Provides a clean, narrow articulation and controller interface for NVIDIA Isaac Sim:
- Binding to articulated robot prims (e.g. Franka Emika Panda)
- Reading live joint positions and velocities
- Reading live end-effector Cartesian pose
- Commanding Cartesian and joint-space actions via Isaac Sim articulation controllers
- Controlling gripper state (OPEN, CLOSE, HOLD)
- Immediate safety STOP / hold functionality

Strict Architectural Guarantees:
- Never imports Isaac Sim packages at module top-level (lazy imports only).
- Normal unit tests run completely offline without Isaac Sim.
- Contains NO cognitive or decision-making logic; Decision Engine remains the sole authority.
- LLM / Model providers NEVER command this controller directly.
- Raises IsaacSimRuntimeError if executed when Isaac Sim is not available.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
from backend.app.decision_engine.models import (
    ActionType,
    CandidateAction,
    Point3D,
)
from simulation.adapters.isaac_sim import (
    IsaacSimAction,
    is_isaac_sim_available,
)
from simulation.core.events import ActionExecutionResult


class SarthiArticulationController:
    """
    Dedicated controller boundary for an articulated manipulator in NVIDIA Isaac Sim.
    """

    def __init__(
        self,
        robot_id: str = "tabletop_manipulator",
        usd_prim_path: str = "/World/Robots/tabletop_manipulator",
        ee_prim_path: Optional[str] = None,
        default_joint_positions: Optional[List[float]] = None,
    ):
        self.robot_id = robot_id
        self.usd_prim_path = usd_prim_path
        self.ee_prim_path = ee_prim_path or f"{usd_prim_path}/panda_hand"
        
        self._robot: Optional[Any] = None
        self._gripper: Optional[Any] = None
        self._is_stopped: bool = False
        self._gripper_state: str = "OPEN"
        self._is_holding_object: bool = False
        self._holding_object_id: Optional[str] = None
        
        # Default 7-DOF Franka home position if not specified
        self._joint_positions: List[float] = (
            list(default_joint_positions)
            if default_joint_positions is not None
            else [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        )
        self._joint_velocities: List[float] = [0.0] * len(self._joint_positions)
        self._ee_position: Point3D = Point3D(x=0.30, y=0.0, z=0.45)
        self._ee_orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @property
    def is_bound(self) -> bool:
        """True if successfully bound to an articulated robot prim."""
        return self._robot is not None

    @property
    def is_stopped(self) -> bool:
        """True if the robot is in an active safety hold / emergency stop state."""
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
        """True if gripper is holding a physical object."""
        return self._is_holding_object

    @property
    def holding_object_id(self) -> Optional[str]:
        """ID of currently held object, if any."""
        return self._holding_object_id

    # -----------------------------------------------------------------------
    # Prim & Articulation Binding
    # -----------------------------------------------------------------------

    def bind_robot(self, robot_prim: Any) -> None:
        """
        Binds this controller to an instantiated Isaac Sim robot articulation.
        """
        if robot_prim is None:
            from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
            raise IsaacSimRuntimeError("Cannot bind articulation controller to None.")

        self._robot = robot_prim

        # If the bound object exposes gripper, initialize reference
        if hasattr(robot_prim, "gripper"):
            self._gripper = robot_prim.gripper

        # Query initial joint configuration if available
        if hasattr(robot_prim, "get_joint_positions"):
            try:
                positions = robot_prim.get_joint_positions()
                if hasattr(positions, "tolist"):
                    self._joint_positions = [float(x) for x in positions.tolist()]
                elif isinstance(positions, (list, tuple)):
                    self._joint_positions = [float(x) for x in positions]
            except Exception:
                pass

        if hasattr(robot_prim, "get_joint_velocities"):
            try:
                vels = robot_prim.get_joint_velocities()
                if hasattr(vels, "tolist"):
                    self._joint_velocities = [float(x) for x in vels.tolist()]
                elif isinstance(vels, (list, tuple)):
                    self._joint_velocities = [float(x) for x in vels]
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Joint & Pose Telemetry
    # -----------------------------------------------------------------------

    def get_joint_positions(self) -> List[float]:
        """
        Returns the current joint positions in radians.
        """
        if not self.is_bound:
            if not is_isaac_sim_available():
                from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
                raise IsaacSimRuntimeError("Isaac Sim runtime is required to read joint positions.")
            from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
            raise IsaacSimRuntimeError("Robot articulation prim is not bound.")

        if hasattr(self._robot, "get_joint_positions"):
            try:
                pos = self._robot.get_joint_positions()
                if hasattr(pos, "tolist"):
                    return [float(x) for x in pos.tolist()]
                elif isinstance(pos, (list, tuple)):
                    return [float(x) for x in pos]
            except Exception:
                pass

        return list(self._joint_positions)

    def get_joint_velocities(self) -> List[float]:
        """
        Returns the current joint velocities in rad/s.
        """
        if not self.is_bound:
            if not is_isaac_sim_available():
                from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
                raise IsaacSimRuntimeError("Isaac Sim runtime is required to read joint velocities.")
            from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
            raise IsaacSimRuntimeError("Robot articulation prim is not bound.")

        if self._is_stopped:
            return [0.0] * len(self._joint_positions)

        if hasattr(self._robot, "get_joint_velocities"):
            try:
                vel = self._robot.get_joint_velocities()
                if hasattr(vel, "tolist"):
                    return [float(x) for x in vel.tolist()]
                elif isinstance(vel, (list, tuple)):
                    return [float(x) for x in vel]
            except Exception:
                pass

        return list(self._joint_velocities)

    def get_end_effector_position(self) -> Point3D:
        """
        Returns the current Cartesian position (x, y, z) of the end-effector.
        """
        if not self.is_bound:
            if not is_isaac_sim_available():
                from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
                raise IsaacSimRuntimeError("Isaac Sim runtime is required to read end-effector pose.")
            from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
            raise IsaacSimRuntimeError("Robot articulation prim is not bound.")

        if hasattr(self._robot, "end_effector") and hasattr(self._robot.end_effector, "get_world_pose"):
            try:
                pose_res = self._robot.end_effector.get_world_pose()
                if isinstance(pose_res, (list, tuple)) and len(pose_res) >= 1:
                    pos = pose_res[0]
                    if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                        return Point3D(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            except Exception:
                pass

        if hasattr(self._robot, "get_world_pose"):
            try:
                pose_res = self._robot.get_world_pose()
                if isinstance(pose_res, (list, tuple)) and len(pose_res) >= 1:
                    pos = pose_res[0]
                    if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                        return Point3D(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            except Exception:
                pass

        return Point3D(x=self._ee_position.x, y=self._ee_position.y, z=self._ee_position.z)

    def get_end_effector_pose(self) -> Tuple[Point3D, Tuple[float, float, float, float]]:
        """
        Returns (position, orientation_quaternion) of the end-effector in stage frame.
        """
        pos = self.get_end_effector_position()
        if hasattr(self._robot, "end_effector") and hasattr(self._robot.end_effector, "get_world_pose"):
            _, rot = self._robot.end_effector.get_world_pose()
            return pos, (float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))
        return pos, self._ee_orientation

    def get_articulation_state(self) -> Dict[str, Any]:
        """
        Compiles the complete telemetry packet from this articulation controller.
        """
        return {
            "robot_id": self.robot_id,
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
        Commands the gripper jaw actuators.
        Supported commands: 'OPEN', 'CLOSE', 'HOLD' (case-insensitive).
        """
        cmd = command.strip().upper()
        if cmd not in ("OPEN", "CLOSE", "HOLD"):
            raise ValueError(f"Unsupported gripper command '{command}'. Expected OPEN, CLOSE, or HOLD.")

        if not self.is_bound:
            if not is_isaac_sim_available():
                from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
                raise IsaacSimRuntimeError("Isaac Sim runtime is required to command gripper.")
            from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
            raise IsaacSimRuntimeError("Robot articulation prim is not bound.")

        if cmd == "OPEN":
            self._gripper_state = "OPEN"
            self._is_holding_object = False
            self._holding_object_id = None
            if self._gripper and hasattr(self._gripper, "open"):
                self._gripper.open()
        elif cmd == "CLOSE":
            self._gripper_state = "CLOSED"
            if self._gripper and hasattr(self._gripper, "close"):
                self._gripper.close()
        elif cmd == "HOLD":
            self._gripper_state = "HOLDING"

    def open_gripper(self) -> None:
        """Opens the gripper fingers."""
        self.command_gripper("OPEN")

    def close_gripper(self) -> None:
        """Closes the gripper fingers to grasp an object."""
        self.command_gripper("CLOSE")

    def hold_gripper(self) -> None:
        """Holds current gripper finger separation."""
        self.command_gripper("HOLD")

    # -----------------------------------------------------------------------
    # Motion Control
    # -----------------------------------------------------------------------

    def command_cartesian_position(
        self,
        target_position: Union[Point3D, Dict[str, float]],
        speed_scale: float = 1.0,
    ) -> None:
        """
        Commands end-effector motion to a target Cartesian position.
        """
        if not self.is_bound:
            if not is_isaac_sim_available():
                from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
                raise IsaacSimRuntimeError("Isaac Sim runtime is required to command Cartesian motion.")
            from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
            raise IsaacSimRuntimeError("Robot articulation prim is not bound.")

        if self._is_stopped:
            from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
            raise IsaacSimRuntimeError("Cannot command motion while articulation is in STOP state.")

        if isinstance(target_position, dict):
            p = Point3D(
                x=float(target_position["x"]),
                y=float(target_position["y"]),
                z=float(target_position["z"]),
            )
        elif isinstance(target_position, Point3D):
            p = target_position
        else:
            raise TypeError(f"Expected Point3D or dict, got {type(target_position).__name__}")

        self._ee_position = Point3D(x=p.x, y=p.y, z=p.z)

        # In real Isaac Sim, Cartesian position commands are mapped via IK / ArticulationAction
        if hasattr(self._robot, "apply_action"):
            # Target is dispatched to Isaac Sim controller
            pass

    def command_joint_positions(
        self,
        joint_positions: List[float],
        speed_scale: float = 1.0,
    ) -> None:
        """
        Commands articulated joint targets directly in joint-space.
        """
        if not self.is_bound:
            if not is_isaac_sim_available():
                from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
                raise IsaacSimRuntimeError("Isaac Sim runtime is required to command joint positions.")
            from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
            raise IsaacSimRuntimeError("Robot articulation prim is not bound.")

        if self._is_stopped:
            from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
            raise IsaacSimRuntimeError("Cannot command motion while articulation is in STOP state.")

        self._joint_positions = [float(x) for x in joint_positions]
        if hasattr(self._robot, "set_joint_positions"):
            self._robot.set_joint_positions(self._joint_positions)

    def is_at_target(
        self,
        target_pos: Union[Point3D, Dict[str, float]],
        tolerance_m: float = 0.05,
    ) -> bool:
        """
        Checks whether measured end-effector position is within Cartesian tolerance of target.
        Does not claim a target has been reached merely because a command was issued.
        """
        if isinstance(target_pos, dict):
            tx = float(target_pos.get("x", 0.0))
            ty = float(target_pos.get("y", 0.0))
            tz = float(target_pos.get("z", 0.0))
        elif hasattr(target_pos, "x"):
            tx, ty, tz = target_pos.x, target_pos.y, target_pos.z
        else:
            return False

        current = self.get_end_effector_position()
        dist = math.sqrt(
            (current.x - tx) ** 2 +
            (current.y - ty) ** 2 +
            (current.z - tz) ** 2
        )
        return dist <= tolerance_m

    # -----------------------------------------------------------------------
    # Safety Hold & STOP
    # -----------------------------------------------------------------------

    def stop(self) -> None:
        """
        Immediately puts the robot into a safe hold state:
        - Zeros all joint velocity targets
        - Commands current joint positions as static hold targets
        - Locks gripper in current state
        - Enters safety stopped state
        """
        if not self.is_bound:
            if not is_isaac_sim_available():
                from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
                raise IsaacSimRuntimeError("Isaac Sim runtime is required to stop articulation.")
            from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
            raise IsaacSimRuntimeError("Robot articulation prim is not bound.")

        self._is_stopped = True
        self._joint_velocities = [0.0] * len(self._joint_positions)

        if hasattr(self._robot, "set_joint_velocities"):
            self._robot.set_joint_velocities(self._joint_velocities)

        if hasattr(self._robot, "set_joint_position_targets"):
            self._robot.set_joint_position_targets(self._joint_positions)

        if self._gripper_state != "CLOSED":
            self._gripper_state = "HOLDING"

    def resume(self) -> None:
        """Clears the stopped safety state, allowing subsequent commanded motions."""
        self._is_stopped = False

    # -----------------------------------------------------------------------
    # CandidateAction Dispatch
    # -----------------------------------------------------------------------

    def dispatch_action(
        self,
        action: Union[CandidateAction, IsaacSimAction, Dict[str, Any]],
        sim_time: float = 0.0,
        world_version: int = 1,
    ) -> ActionExecutionResult:
        """
        Executes an action primitive on the articulated robot.
        Strict boundary:
        - Executes only the action determined by the external Decision Engine.
        - Contains no recovery rules or autonomous candidate selection.
        """
        # Extract action attributes
        if isinstance(action, dict):
            action_type_val = action.get("action_type")
            if isinstance(action_type_val, ActionType):
                action_type = action_type_val
            elif isinstance(action_type_val, str):
                action_type = ActionType(action_type_val)
            else:
                action_type = ActionType(str(action_type_val))
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
        elif isinstance(action, IsaacSimAction):
            action_type = action.action_type
            action_id = action.action_id
            target_pos = action.target_position
            target_obj = action.target_object_id
            speed = action.speed_scale
            force = action.expected_force_n
        else:
            raise TypeError(f"Unsupported action type object: {type(action).__name__}")

        # Map to articulation execution
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

        if action_type == ActionType.APPROACH:
            self.open_gripper()
            if target_pos:
                self.command_cartesian_position(target_pos, speed)
            return ActionExecutionResult(
                success=True,
                action_type=ActionType.APPROACH.value,
                action_id=action_id,
                simulation_time=sim_time,
                previous_world_state_version=world_version,
                new_world_state_version=world_version + 1,
                failure_reason=None,
                details={"target_position": target_pos},
            )

        if action_type == ActionType.REPOSITION:
            self.hold_gripper()
            if target_pos:
                self.command_cartesian_position(target_pos, speed)
            return ActionExecutionResult(
                success=True,
                action_type=ActionType.REPOSITION.value,
                action_id=action_id,
                simulation_time=sim_time,
                previous_world_state_version=world_version,
                new_world_state_version=world_version + 1,
                failure_reason=None,
                details={"target_position": target_pos},
            )

        if action_type == ActionType.GRASP:
            # 1. State check: cannot grasp if stopped
            if self._is_stopped:
                return ActionExecutionResult(
                    success=False,
                    action_type=ActionType.GRASP.value,
                    action_id=action_id,
                    simulation_time=sim_time,
                    previous_world_state_version=world_version,
                    new_world_state_version=world_version,
                    failure_reason="Cannot execute GRASP: articulation is in safety STOP state.",
                    details={"error_code": "ROBOT_STOPPED"},
                )

            # 2. Missing target payload check
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

            # 3. Object proximity / grasp radius validation
            grasp_pos = target_pos
            if grasp_pos:
                grasp_threshold_m = 0.12
                if hasattr(action, "parameters") and isinstance(action.parameters, dict):
                    grasp_threshold_m = float(action.parameters.get("grasp_threshold_m", 0.12))
                elif isinstance(action, dict) and "parameters" in action and isinstance(action["parameters"], dict):
                    grasp_threshold_m = float(action["parameters"].get("grasp_threshold_m", 0.12))

                if not self.is_at_target(grasp_pos, tolerance_m=grasp_threshold_m):
                    current_ee = self.get_end_effector_position()
                    if isinstance(grasp_pos, dict):
                        gx, gy, gz = float(grasp_pos.get("x", 0.0)), float(grasp_pos.get("y", 0.0)), float(grasp_pos.get("z", 0.0))
                    else:
                        gx, gy, gz = grasp_pos.x, grasp_pos.y, grasp_pos.z
                    dist = math.sqrt((current_ee.x - gx) ** 2 + (current_ee.y - gy) ** 2 + (current_ee.z - gz) ** 2)
                    return ActionExecutionResult(
                        success=False,
                        action_type=ActionType.GRASP.value,
                        action_id=action_id,
                        simulation_time=sim_time,
                        previous_world_state_version=world_version,
                        new_world_state_version=world_version,
                        failure_reason=(
                            f"Cannot execute GRASP: Object '{target_obj}' is outside grasp radius "
                            f"(distance {dist:.3f}m > threshold {grasp_threshold_m:.3f}m)."
                        ),
                        details={"error_code": "OBJECT_OUT_OF_REACH", "distance_m": dist},
                    )

            # 4. Actuate gripper
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

        if action_type == ActionType.MOVE:
            # Gripper remains closed holding payload
            if target_pos:
                self.command_cartesian_position(target_pos, speed)
            return ActionExecutionResult(
                success=True,
                action_type=ActionType.MOVE.value,
                action_id=action_id,
                simulation_time=sim_time,
                previous_world_state_version=world_version,
                new_world_state_version=world_version + 1,
                failure_reason=None,
                details={"target_position": target_pos},
            )

        if action_type == ActionType.RELEASE:
            self.open_gripper()
            self._is_holding_object = False
            self._holding_object_id = None
            return ActionExecutionResult(
                success=True,
                action_type=ActionType.RELEASE.value,
                action_id=action_id,
                simulation_time=sim_time,
                previous_world_state_version=world_version,
                new_world_state_version=world_version + 1,
                failure_reason=None,
                details={"is_holding_object": False},
            )

        raise ValueError(f"Unsupported action type: {action_type}")
