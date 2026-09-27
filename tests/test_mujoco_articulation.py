"""
Tests for SĀRTHI MuJoCo Franka Panda Articulation Controller & IK (Phase C).
Validates:
1. Articulation initialization
2. Correct Panda joint discovery
3. Correct actuator discovery
4. End-effector position reading
5. Finite joint state
6. IK convergence toward a reachable target
7. IK rejection/failure for clearly invalid or unreachable target
8. Joint-limit safety
9. Gripper open command
10. Gripper close command
11. STOP safe hold behavior
12. Deterministic stepping
"""

import math
import unittest
import numpy as np

from backend.app.decision_engine.models import (
    ActionType,
    CandidateAction,
    Point3D,
)
from simulation.mujoco_runtime import (
    is_mujoco_available,
    SarthiMuJoCoSceneBuilder,
    SarthiMuJoCoArticulation,
    MuJoCoIKConfig,
)


@unittest.skipUnless(is_mujoco_available(), "MuJoCo is required for Phase C articulation tests.")
class TestSarthiMuJoCoArticulation(unittest.TestCase):
    """Phase C test suite for Franka Panda articulation controller & DLS-IK."""

    def setUp(self):
        import mujoco

        self.builder = SarthiMuJoCoSceneBuilder()
        self.model, self.data = self.builder.build()

        # Reset to canonical home keyframe
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.assertGreaterEqual(key_id, 0, "Expected 'home' keyframe in Franka Panda model.")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)

        self.articulation = SarthiMuJoCoArticulation(self.model, self.data)

    def test_01_articulation_initialization(self):
        """1. Articulation initializes cleanly with default properties."""
        self.assertFalse(self.articulation.is_stopped)
        self.assertEqual(self.articulation.gripper_state, "OPEN")
        self.assertFalse(self.articulation.is_gripper_closed)
        self.assertFalse(self.articulation.is_holding_object)
        self.assertIsNone(self.articulation.holding_object_id)
        self.assertEqual(len(self.articulation.joint_names), 7)
        self.assertEqual(len(self.articulation.actuator_names), 7)
        self.assertIsInstance(self.articulation.ik_config, MuJoCoIKConfig)

    def test_02_correct_panda_joint_discovery(self):
        """2. Correct Panda 7-DOF arm joints discovered with valid limits."""
        expected_joints = [f"joint{i}" for i in range(1, 8)]
        self.assertEqual(self.articulation.joint_names, expected_joints)

        limits = self.articulation.joint_limits
        self.assertEqual(len(limits), 7)
        for i, (min_lim, max_lim) in enumerate(limits):
            self.assertLess(min_lim, max_lim, f"Joint {i + 1} lower limit must be less than upper limit.")
            self.assertTrue(math.isfinite(min_lim))
            self.assertTrue(math.isfinite(max_lim))

        # Check specific Panda joint 4 asymmetric range (-3.0718 to -0.0698)
        min_j4, max_j4 = limits[3]
        self.assertAlmostEqual(min_j4, -3.0718, places=3)
        self.assertAlmostEqual(max_j4, -0.0698, places=3)

    def test_03_correct_actuator_discovery(self):
        """3. Correct Franka Panda arm actuators and gripper actuator discovered."""
        expected_actuators = [f"actuator{i}" for i in range(1, 8)]
        self.assertEqual(self.articulation.actuator_names, expected_actuators)
        self.assertEqual(self.articulation.GRIPPER_ACTUATOR_NAME, "actuator8")

    def test_04_end_effector_position_reading(self):
        """4. End-effector Cartesian position reading returns finite coordinates."""
        ee_pos = self.articulation.get_end_effector_position()
        self.assertIsInstance(ee_pos, Point3D)
        self.assertTrue(math.isfinite(ee_pos.x))
        self.assertTrue(math.isfinite(ee_pos.y))
        self.assertTrue(math.isfinite(ee_pos.z))

        # At home keyframe, EE should be roughly x~0.55m, y~0.0m, z~0.52m
        self.assertAlmostEqual(ee_pos.x, 0.5545, delta=0.05)
        self.assertAlmostEqual(ee_pos.y, 0.0, delta=0.05)
        self.assertAlmostEqual(ee_pos.z, 0.5211, delta=0.05)

        # Pose reading (position + orientation)
        pos, quat = self.articulation.get_end_effector_pose()
        self.assertEqual(len(quat), 4)
        for q in quat:
            self.assertTrue(math.isfinite(q))

    def test_05_finite_joint_state(self):
        """5. Joint positions and velocities are strictly finite."""
        positions = self.articulation.get_joint_positions()
        velocities = self.articulation.get_joint_velocities()

        self.assertEqual(len(positions), 7)
        self.assertEqual(len(velocities), 7)

        for p in positions:
            self.assertTrue(math.isfinite(p))
        for v in velocities:
            self.assertTrue(math.isfinite(v))

        # Compiles complete state dictionary cleanly
        state = self.articulation.get_articulation_state()
        self.assertIn("joint_positions", state)
        self.assertIn("joint_velocities", state)
        self.assertIn("end_effector_position", state)
        self.assertIn("gripper_state", state)

    def test_06_ik_convergence_reachable_target(self):
        """6. DLS-IK converges cleanly to a reachable Cartesian workspace target."""
        # Target position above red object: (0.25, 0.15, 0.25)
        target = Point3D(x=0.25, y=0.15, z=0.25)

        converged, solved_q, dist = self.articulation.solve_ik(target)
        self.assertTrue(converged, f"IK failed to converge. Dist: {dist:.4f}m")
        self.assertLessEqual(dist, 0.005, "Convergence distance should be within 5mm.")
        self.assertEqual(len(solved_q), 7)

        # Verify solved angles respect joint limits
        for i, q in enumerate(solved_q):
            min_lim, max_lim = self.articulation.joint_limits[i]
            self.assertGreaterEqual(q, min_lim - 1e-4)
            self.assertLessEqual(q, max_lim + 1e-4)

    def test_07_ik_rejection_invalid_target(self):
        """7. IK gracefully rejects unreachable or non-finite Cartesian targets."""
        # 1. Unreachable target far outside physical reach
        unreachable = Point3D(x=2.5, y=2.5, z=2.5)
        converged, _, dist = self.articulation.solve_ik(unreachable)
        self.assertFalse(converged)
        self.assertGreater(dist, 1.0)

        # 2. NaN target
        nan_target = Point3D(x=float("nan"), y=0.15, z=0.25)
        converged_nan, _, _ = self.articulation.solve_ik(nan_target)
        self.assertFalse(converged_nan)

        # 3. move_to_position returns False for unreachable target
        res = self.articulation.move_to_position(unreachable)
        self.assertFalse(res)

    def test_08_joint_limit_safety(self):
        """8. Joint targets outside hardware limits are safely clamped."""
        out_of_bounds_target = [10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0]
        self.articulation.command_joint_positions(out_of_bounds_target)

        # Verify applied actuator control values are clamped to limits
        for i, aid in enumerate(self.articulation._arm_actuator_ids):
            ctrl_val = float(self.data.ctrl[aid])
            min_lim, max_lim = self.articulation.joint_limits[i]
            self.assertGreaterEqual(ctrl_val, min_lim - 1e-5)
            self.assertLessEqual(ctrl_val, max_lim + 1e-5)

        # Reject invalid length
        with self.assertRaises(ValueError):
            self.articulation.command_joint_positions([0.0] * 5)

        # Reject NaN
        with self.assertRaises(ValueError):
            self.articulation.command_joint_positions([float("nan")] * 7)

    def test_09_gripper_open_command(self):
        """9. Gripper open command sets actuator8 to 255.0 and updates state."""
        self.articulation.close_gripper()
        self.assertEqual(self.articulation.gripper_state, "CLOSED")

        self.articulation.open_gripper()
        self.assertEqual(self.articulation.gripper_state, "OPEN")
        self.assertFalse(self.articulation.is_gripper_closed)
        gripper_aid = self.articulation._gripper_actuator_id
        self.assertEqual(self.data.ctrl[gripper_aid], 255.0)

    def test_10_gripper_close_command(self):
        """10. Gripper close command sets actuator8 to 0.0 and updates state."""
        self.articulation.close_gripper()
        self.assertEqual(self.articulation.gripper_state, "CLOSED")
        self.assertTrue(self.articulation.is_gripper_closed)
        gripper_aid = self.articulation._gripper_actuator_id
        self.assertEqual(self.data.ctrl[gripper_aid], 0.0)

    def test_11_stop_behavior(self):
        """11. STOP action puts robot into safe hold and rejects motion commands."""
        self.articulation.stop()
        self.assertTrue(self.articulation.is_stopped)

        # Velocities zeroed
        velocities = self.articulation.get_joint_velocities()
        self.assertEqual(velocities, [0.0] * 7)

        # Motion commands rejected while stopped
        with self.assertRaises(RuntimeError):
            self.articulation.command_joint_positions([0.0] * 7)

        with self.assertRaises(RuntimeError):
            self.articulation.move_to_position(Point3D(x=0.25, y=0.15, z=0.25))

        # Dispatching action fails gracefully
        action = CandidateAction(
            action_type=ActionType.APPROACH,
            action_id="act_approach_test",
            target_position=Point3D(x=0.25, y=0.15, z=0.25),
        )
        result = self.articulation.dispatch_action(action)
        self.assertFalse(result.success)
        self.assertEqual(result.details.get("error_code"), "ROBOT_STOPPED")

        # Resume restores normal operation
        self.articulation.resume()
        self.assertFalse(self.articulation.is_stopped)

    def test_12_deterministic_stepping(self):
        """12. Physics stepping advances deterministically across identical rollouts."""
        import mujoco

        # Stepping advances time deterministically
        t_start = float(self.data.time)
        self.articulation.step(50)
        self.assertAlmostEqual(float(self.data.time) - t_start, 50 * self.model.opt.timestep)

        # Reset and compare two rollouts from home
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")

        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        self.articulation.command_joint_positions([0.1, -0.2, 0.1, -1.8, 0.0, 1.6, -0.5])
        self.articulation.step(100)
        qpos_rollout_1 = list(self.articulation.get_joint_positions())

        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        self.articulation.command_joint_positions([0.1, -0.2, 0.1, -1.8, 0.0, 1.6, -0.5])
        self.articulation.step(100)
        qpos_rollout_2 = list(self.articulation.get_joint_positions())

        self.assertEqual(qpos_rollout_1, qpos_rollout_2, "Two identical runs must produce identical joint states.")

    def test_13_candidate_action_dispatch(self):
        """13. Clean execution boundary for all SĀRTHI candidate action types."""
        # 1. APPROACH
        approach = CandidateAction(
            action_type=ActionType.APPROACH,
            action_id="act_app_1",
            target_position=Point3D(x=0.25, y=0.15, z=0.25),
        )
        res_app = self.articulation.dispatch_action(approach)
        self.assertTrue(res_app.success)
        self.assertEqual(res_app.action_type, "APPROACH")
        self.assertEqual(self.articulation.gripper_state, "OPEN")

        # 2. GRASP
        grasp = CandidateAction(
            action_type=ActionType.GRASP,
            action_id="act_grasp_1",
            target_object_id="red_object",
            target_position=Point3D(x=0.25, y=0.15, z=0.25),
        )
        # Advance simulation to bring arm close
        self.articulation.step(250)
        res_grasp = self.articulation.dispatch_action(grasp)
        self.assertTrue(res_grasp.success)
        self.assertEqual(res_grasp.action_type, "GRASP")
        self.assertTrue(self.articulation.is_holding_object)
        self.assertEqual(self.articulation.holding_object_id, "red_object")

        # 3. REPOSITION
        reposition = CandidateAction(
            action_type=ActionType.REPOSITION,
            action_id="act_repo_1",
            target_position=Point3D(x=0.30, y=0.0, z=0.35),
        )
        res_repo = self.articulation.dispatch_action(reposition)
        self.assertTrue(res_repo.success)
        self.assertEqual(res_repo.action_type, "REPOSITION")

        # 4. MOVE
        move = CandidateAction(
            action_type=ActionType.MOVE,
            action_id="act_move_1",
            target_position=Point3D(x=0.40, y=-0.20, z=0.25),
        )
        res_move = self.articulation.dispatch_action(move)
        self.assertTrue(res_move.success)
        self.assertEqual(res_move.action_type, "MOVE")

        # 5. RELEASE
        release = CandidateAction(
            action_type=ActionType.RELEASE,
            action_id="act_rel_1",
        )
        res_rel = self.articulation.dispatch_action(release)
        self.assertTrue(res_rel.success)
        self.assertEqual(res_rel.action_type, "RELEASE")
        self.assertEqual(self.articulation.gripper_state, "OPEN")

        # 6. STOP
        stop_act = CandidateAction(
            action_type=ActionType.STOP,
            action_id="act_stop_1",
        )
        res_stop = self.articulation.dispatch_action(stop_act)
        self.assertTrue(res_stop.success)
        self.assertEqual(res_stop.action_type, "STOP")
        self.assertTrue(self.articulation.is_stopped)


if __name__ == "__main__":
    unittest.main()
