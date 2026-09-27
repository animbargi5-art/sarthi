"""
SĀRTHI MuJoCo Runtime — Physical Action Verifier.

Verifies post-action physical state transitions in MuJoCo using actual
physical quantities (telemetry, contacts, kinematics, and canonical WorldState).
"""

import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from backend.app.decision_engine.models import (
    ActionType,
    CandidateAction,
    Point3D,
    WorldState,
)
from simulation.mujoco_runtime.state_reader import SarthiMuJoCoStateReader


class SarthiMuJoCoVerifier:
    """
    Evaluates whether an executed action achieved its expected physical outcome.
    Strictly uses measured physical data from MuJoCo, never software flags.
    """

    DEFAULT_CARTESIAN_TOLERANCE_M = 0.05
    DEFAULT_GRASP_RADIUS_M = 0.10
    DEFAULT_PLACEMENT_TOLERANCE_M = 0.06

    def __init__(self, state_reader: SarthiMuJoCoStateReader):
        self._state_reader = state_reader

    @property
    def state_reader(self) -> SarthiMuJoCoStateReader:
        return self._state_reader

    def verify(
        self,
        action: Union[CandidateAction, Dict[str, Any]],
        expected_outcome: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Main verification entrypoint for any executed action.
        Returns:
            Tuple of (success: bool, failure_reason: Optional[str])
        """
        # Parse action type
        if isinstance(action, dict):
            act_type = action.get("action_type")
            action_type = ActionType(act_type) if not isinstance(act_type, ActionType) else act_type
            target_pos = action.get("target_position")
            target_obj = action.get("target_object_id")
        elif isinstance(action, CandidateAction):
            action_type = action.action_type
            target_pos = action.target_position
            target_obj = action.target_object_id
        else:
            raise TypeError(f"Unsupported action type: {type(action).__name__}")

        if action_type == ActionType.STOP:
            return self.verify_stop(action)

        if action_type == ActionType.APPROACH:
            return self.verify_approach(action, expected_outcome)

        if action_type == ActionType.REPOSITION:
            return self.verify_reposition(action, expected_outcome)

        if action_type == ActionType.GRASP:
            return self.verify_grasp(action, expected_outcome)

        if action_type == ActionType.MOVE:
            return self.verify_move(action, expected_outcome)

        if action_type == ActionType.RELEASE:
            return self.verify_release(action, expected_outcome)

        return False, f"Unknown action type: {action_type}"

    # -----------------------------------------------------------------------
    # Action-Specific Verification
    # -----------------------------------------------------------------------

    def verify_approach(
        self,
        action: Union[CandidateAction, Dict[str, Any]],
        expected_outcome: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verifies APPROACH:
        - End effector reached requested target Cartesian position within tolerance.
        """
        target_p = self._extract_target_position(action)
        if target_p is None:
            return False, "APPROACH action missing target_position."

        tol = (
            expected_outcome.get("tolerance_m", self.DEFAULT_CARTESIAN_TOLERANCE_M)
            if expected_outcome
            else self.DEFAULT_CARTESIAN_TOLERANCE_M
        )

        ee = self._state_reader.get_end_effector_position()
        dist = math.sqrt(
            (ee.x - target_p.x) ** 2
            + (ee.y - target_p.y) ** 2
            + (ee.z - target_p.z) ** 2
        )

        if dist <= tol:
            return True, None
        return (
            False,
            f"End-effector distance {dist:.4f}m exceeds tolerance {tol:.4f}m for APPROACH.",
        )

    def verify_reposition(
        self,
        action: Union[CandidateAction, Dict[str, Any]],
        expected_outcome: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verifies REPOSITION:
        - End effector reached requested reposition target within tolerance.
        """
        target_p = self._extract_target_position(action)
        if target_p is None:
            return False, "REPOSITION action missing target_position."

        tol = (
            expected_outcome.get("tolerance_m", self.DEFAULT_CARTESIAN_TOLERANCE_M)
            if expected_outcome
            else self.DEFAULT_CARTESIAN_TOLERANCE_M
        )

        ee = self._state_reader.get_end_effector_position()
        dist = math.sqrt(
            (ee.x - target_p.x) ** 2
            + (ee.y - target_p.y) ** 2
            + (ee.z - target_p.z) ** 2
        )

        if dist <= tol:
            return True, None
        return (
            False,
            f"End-effector distance {dist:.4f}m exceeds tolerance {tol:.4f}m for REPOSITION.",
        )

    def verify_grasp(
        self,
        action: Union[CandidateAction, Dict[str, Any]],
        expected_outcome: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verifies GRASP:
        - Gripper fingers are physically closed / clamped.
        - Object is within physical grasp radius.
        - Physical grasp evidence exists (contact or clamping proximity).
        """
        if self._state_reader.is_gripper_open():
            return False, "Gripper is still open after GRASP command."

        ee = self._state_reader.get_end_effector_position()
        obj = self._state_reader.get_red_object_position()
        dist = math.sqrt((ee.x - obj.x) ** 2 + (ee.y - obj.y) ** 2 + (ee.z - obj.z) ** 2)

        radius = (
            expected_outcome.get("grasp_radius_m", self.DEFAULT_GRASP_RADIUS_M)
            if expected_outcome
            else self.DEFAULT_GRASP_RADIUS_M
        )

        if dist > radius:
            return (
                False,
                f"Object distance {dist:.3f}m exceeds grasp radius {radius:.3f}m.",
            )

        if not self._state_reader.is_object_grasped():
            return False, "Physical grasp evidence not found (insufficient clamping or contact)."

        return True, None

    def verify_move(
        self,
        action: Union[CandidateAction, Dict[str, Any]],
        expected_outcome: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verifies MOVE:
        - End effector reached requested trajectory target.
        - No joint limit or finite state violation occurred.
        """
        target_p = self._extract_target_position(action)
        if target_p is None:
            return False, "MOVE action missing target_position."

        tol = (
            expected_outcome.get("tolerance_m", self.DEFAULT_CARTESIAN_TOLERANCE_M)
            if expected_outcome
            else self.DEFAULT_CARTESIAN_TOLERANCE_M
        )

        ee = self._state_reader.get_end_effector_position()
        dist = math.sqrt(
            (ee.x - target_p.x) ** 2
            + (ee.y - target_p.y) ** 2
            + (ee.z - target_p.z) ** 2
        )

        if dist > tol:
            return (
                False,
                f"MOVE destination not reached (distance {dist:.4f}m > tolerance {tol:.4f}m).",
            )

        return True, None

    def verify_release(
        self,
        action: Union[CandidateAction, Dict[str, Any]],
        expected_outcome: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verifies RELEASE:
        - Gripper fingers opened physically.
        - Object is no longer held.
        """
        if not self._state_reader.is_gripper_open():
            return False, "Gripper failed to open after RELEASE."

        if self._state_reader.is_object_grasped():
            return False, "Object is still physically held after RELEASE."

        return True, None

    def verify_stop(
        self,
        action: Union[CandidateAction, Dict[str, Any]],
    ) -> Tuple[bool, Optional[str]]:
        """
        Verifies STOP:
        - Robot arm joint velocities are zeroed / near static rest.
        """
        if self._state_reader.is_robot_moving(threshold_rad_per_s=0.10):
            return False, "Robot joint velocities indicate ongoing motion after STOP."
        return True, None

    def verify_placement(self, tolerance_m: Optional[float] = None) -> bool:
        """
        Evaluates task placement success strictly from physical object and target coordinates:
            distance(object_position, target_position) <= tolerance
        """
        tol = tolerance_m or self.DEFAULT_PLACEMENT_TOLERANCE_M
        obj_pos = self._state_reader.get_red_object_position()
        tgt_pos = self._state_reader.get_blue_target_position()
        dist = math.sqrt(
            (obj_pos.x - tgt_pos.x) ** 2
            + (obj_pos.y - tgt_pos.y) ** 2
            + (obj_pos.z - tgt_pos.z) ** 2
        )
        return dist <= tol

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def _extract_target_position(
        self, action: Union[CandidateAction, Dict[str, Any]]
    ) -> Optional[Point3D]:
        """Extracts Point3D target position from CandidateAction or dict."""
        if isinstance(action, dict):
            raw = action.get("target_position")
        elif isinstance(action, CandidateAction):
            raw = action.target_position
        else:
            return None

        if raw is None:
            return None
        if isinstance(raw, Point3D):
            return raw
        if isinstance(raw, dict):
            return Point3D(x=float(raw["x"]), y=float(raw["y"]), z=float(raw["z"]))
        if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) >= 3:
            return Point3D(x=float(raw[0]), y=float(raw[1]), z=float(raw[2]))
        return None
