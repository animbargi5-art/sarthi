"""
Tests for SĀRTHI MuJoCo Runtime Adapter & Action Execution (Phase E).
Validates:
1. Runtime initialization
2. SimulationAdapter compatibility
3. get_world_state()
4. APPROACH execution
5. REPOSITION execution
6. GRASP command execution
7. RELEASE command execution
8. STOP command execution
9. Bounded physics stepping
10. Timeout / budget handling
11. Verification success
12. Verification failure
13. Actual object position after movement
14. PATH_BLOCKED disturbance changes WorldState
15. No teleportation
16. Deterministic execution
"""

import math
import unittest
import numpy as np

from backend.app.decision_engine.models import (
    ActionType,
    CandidateAction,
    Point3D,
    WorldState,
)
from simulation.adapters.base import SimulationAdapter
from simulation.core.events import ActionExecutionResult, DisturbanceEvent, DisturbanceType
from simulation.mujoco_runtime import (
    is_mujoco_available,
    SarthiMuJoCoRuntime,
    SarthiMuJoCoVerifier,
)


@unittest.skipUnless(is_mujoco_available(), "MuJoCo is required for Phase E runtime tests.")
class TestSarthiMuJoCoRuntime(unittest.TestCase):
    """Phase E test suite for MuJoCo action execution, physics stepping, and verification."""

    def setUp(self):
        self.runtime = SarthiMuJoCoRuntime(with_disturbance=False)

    def test_01_runtime_initialization(self):
        """1. Runtime initializes all sub-components with deterministic settings."""
        self.assertIsNotNone(self.runtime.model)
        self.assertIsNotNone(self.runtime.data)
        self.assertIsNotNone(self.runtime.articulation)
        self.assertIsNotNone(self.runtime.state_reader)
        self.assertIsNotNone(self.runtime.verifier)
        self.assertEqual(self.runtime.world_state_version, 1)
        self.assertFalse(self.runtime.is_disturbance_active)
        self.assertEqual(len(self.runtime.execution_history), 0)

    def test_02_simulation_adapter_compatibility(self):
        """2. SarthiMuJoCoRuntime strictly conforms to SimulationAdapter contract."""
        self.assertIsInstance(self.runtime, SimulationAdapter)
        self.assertTrue(callable(getattr(self.runtime, "get_world_state", None)))
        self.assertTrue(callable(getattr(self.runtime, "execute_action", None)))
        self.assertTrue(callable(getattr(self.runtime, "inject_disturbance", None)))
        self.assertTrue(callable(getattr(self.runtime, "verify_action_result", None)))

    def test_03_get_world_state(self):
        """3. get_world_state returns fully valid canonical WorldState model."""
        ws = self.runtime.get_world_state()
        self.assertIsInstance(ws, WorldState)
        self.assertEqual(ws.version, 1)
        self.assertIsInstance(ws.robot.position, Point3D)
        self.assertTrue(ws.robot.gripper_open)
        self.assertEqual(len(ws.objects), 1)  # Only red_object when disturbance is inactive
        self.assertEqual(ws.objects[0].id, "red_object_01")
        self.assertFalse(ws.environment.dynamic_obstacles_detected)

    def test_04_approach_execution(self):
        """4. Executes APPROACH action and verifies end-effector reaches target."""
        target = Point3D(x=0.25, y=0.15, z=0.25)
        action = CandidateAction(
            action_type=ActionType.APPROACH,
            action_id="act_app_test",
            target_position=target,
        )
        result = self.runtime.execute_action(action)

        self.assertIsInstance(result, ActionExecutionResult)
        self.assertTrue(result.success)
        self.assertEqual(result.action_type, "APPROACH")
        self.assertIsNone(result.failure_reason)
        self.assertEqual(result.new_world_state_version, 2)

        # End effector verified at target
        ee = self.runtime.state_reader.get_end_effector_position()
        dist = math.sqrt((ee.x - target.x) ** 2 + (ee.y - target.y) ** 2 + (ee.z - target.z) ** 2)
        self.assertLessEqual(dist, 0.05)

    def test_05_reposition_execution(self):
        """5. Executes REPOSITION action to an obstacle detour target."""
        target = Point3D(x=0.30, y=0.0, z=0.35)
        action = CandidateAction(
            action_type=ActionType.REPOSITION,
            action_id="act_repo_test",
            target_position=target,
        )
        result = self.runtime.execute_action(action)

        self.assertTrue(result.success)
        self.assertEqual(result.action_type, "REPOSITION")
        self.assertIsNone(result.failure_reason)

        ee = self.runtime.state_reader.get_end_effector_position()
        dist = math.sqrt((ee.x - target.x) ** 2 + (ee.y - target.y) ** 2 + (ee.z - target.z) ** 2)
        self.assertLessEqual(dist, 0.05)

    def test_06_grasp_command_execution(self):
        """6. Approaches object and executes GRASP, verifying physical closure."""
        # First approach above object
        approach_action = CandidateAction(
            action_type=ActionType.APPROACH,
            action_id="act_app_pregrasp",
            target_position=Point3D(x=0.25, y=0.15, z=0.25),
        )
        self.runtime.execute_action(approach_action)

        grasp_action = CandidateAction(
            action_type=ActionType.GRASP,
            action_id="act_grasp_test",
            target_object_id="red_object_01",
            target_position=Point3D(x=0.25, y=0.15, z=0.25),
        )
        result = self.runtime.execute_action(grasp_action)

        self.assertTrue(result.success)
        self.assertEqual(result.action_type, "GRASP")
        self.assertFalse(self.runtime.state_reader.is_gripper_open())

    def test_07_release_command_execution(self):
        """7. Executes RELEASE, opening gripper fingers."""
        # Close gripper first
        self.runtime.articulation.close_gripper()
        self.runtime.articulation.step(150)
        self.assertFalse(self.runtime.state_reader.is_gripper_open())

        release_action = CandidateAction(
            action_type=ActionType.RELEASE,
            action_id="act_release_test",
        )
        result = self.runtime.execute_action(release_action)

        self.assertTrue(result.success)
        self.assertEqual(result.action_type, "RELEASE")
        self.assertTrue(self.runtime.state_reader.is_gripper_open())

    def test_08_stop_command_execution(self):
        """8. Executes STOP, engaging safe hold and rejecting motion."""
        stop_action = CandidateAction(
            action_type=ActionType.STOP,
            action_id="act_stop_test",
        )
        result = self.runtime.execute_action(stop_action)

        self.assertTrue(result.success)
        self.assertEqual(result.action_type, "STOP")
        self.assertTrue(self.runtime.articulation.is_stopped)

        # Subsequent motion action is rejected safely
        move_action = CandidateAction(
            action_type=ActionType.APPROACH,
            action_id="act_after_stop",
            target_position=Point3D(x=0.3, y=0.1, z=0.3),
        )
        fail_res = self.runtime.execute_action(move_action)
        self.assertFalse(fail_res.success)
        self.assertEqual(fail_res.details.get("error_code"), "ROBOT_STOPPED")

    def test_09_bounded_physics_stepping(self):
        """9. Physics advances within bounded ticks per action execution."""
        start_time = self.runtime.simulation_time
        action = CandidateAction(
            action_type=ActionType.APPROACH,
            action_id="act_bounded_step",
            target_position=Point3D(x=0.25, y=0.15, z=0.25),
        )
        result = self.runtime.execute_action(action)

        delta_time = self.runtime.simulation_time - start_time
        self.assertGreater(delta_time, 0.0)
        # 500 max ticks * 0.002s = 1.0s maximum
        self.assertLessEqual(delta_time, 1.2)
        self.assertIn("physics_ticks", result.details)
        self.assertLessEqual(result.details["physics_ticks"], 550)

    def test_10_timeout_handling(self):
        """10. Unreachable target times out / rejects without hanging in infinite loop."""
        unreachable_action = CandidateAction(
            action_type=ActionType.APPROACH,
            action_id="act_unreachable",
            target_position=Point3D(x=2.5, y=2.5, z=2.5),
        )
        result = self.runtime.execute_action(unreachable_action)

        self.assertFalse(result.success)
        self.assertIn("IK failed", result.failure_reason)

    def test_11_verification_success(self):
        """11. verify_action_result returns True when physical conditions are satisfied."""
        action = CandidateAction(
            action_type=ActionType.APPROACH,
            action_id="act_v_ok",
            target_position=Point3D(x=0.25, y=0.15, z=0.25),
        )
        self.runtime.execute_action(action)
        verified = self.runtime.verify_action_result(action)
        self.assertTrue(verified)

    def test_12_verification_failure(self):
        """12. verify_action_result returns False when tolerance expectations are violated."""
        action = CandidateAction(
            action_type=ActionType.APPROACH,
            action_id="act_v_fail",
            target_position=Point3D(x=0.25, y=0.15, z=0.25),
        )
        self.runtime.execute_action(action)

        # Require an absurdly strict tolerance (e.g. 0.0001m)
        verified = self.runtime.verify_action_result(action, expected_outcome={"tolerance_m": 0.00001})
        self.assertFalse(verified)

    def test_13_actual_object_position_after_movement(self):
        """13. Object positions reflect actual physical simulation state."""
        import mujoco

        # Move the object directly in physics
        self.runtime.data.qpos[9] = 0.38
        self.runtime.data.qpos[10] = 0.18
        self.runtime.data.qpos[11] = 0.22
        mujoco.mj_forward(self.runtime.model, self.runtime.data)

        ws = self.runtime.get_world_state()
        red_obj = next(o for o in ws.objects if o.is_target)
        self.assertAlmostEqual(red_obj.position.x, 0.38, places=3)
        self.assertAlmostEqual(red_obj.position.y, 0.18, places=3)
        self.assertAlmostEqual(red_obj.position.z, 0.22, places=3)

    def test_14_path_blocked_disturbance_changes_world_state(self):
        """14. Injecting PATH_BLOCKED disturbance physically activates obstacle in WorldState."""
        ws_before = self.runtime.get_world_state()
        self.assertFalse(ws_before.environment.dynamic_obstacles_detected)
        self.assertEqual(len([o for o in ws_before.objects if o.is_obstacle]), 0)

        # Inject disturbance
        event = DisturbanceEvent(
            event_id="dist_block_1",
            disturbance_type=DisturbanceType.PATH_BLOCKED,
            parameters={"position": {"x": 0.325, "y": -0.025, "z": 0.21}},
        )
        injected = self.runtime.inject_disturbance(event)
        self.assertTrue(injected)

        ws_after = self.runtime.get_world_state()
        self.assertTrue(ws_after.environment.dynamic_obstacles_detected)
        obstacles = [o for o in ws_after.objects if o.is_obstacle]
        self.assertEqual(len(obstacles), 1)
        self.assertEqual(obstacles[0].id, "blocking_barrier_01")
        self.assertAlmostEqual(obstacles[0].position.x, 0.325, places=3)

        # Active constraint keep-out created
        self.assertGreaterEqual(len(ws_after.active_constraints), 1)
        self.assertEqual(ws_after.active_constraints[0].constraint_id, "c_obstacle_keepout")

    def test_15_no_teleportation(self):
        """15. Robot moves via actuator controls and forward integration, not teleportation."""
        init_pos = self.runtime.state_reader.get_end_effector_position()
        target = Point3D(x=0.25, y=0.15, z=0.25)

        action = CandidateAction(
            action_type=ActionType.APPROACH,
            action_id="act_physics_motion",
            target_position=target,
        )
        self.runtime.execute_action(action)

        final_pos = self.runtime.state_reader.get_end_effector_position()
        self.assertNotEqual(init_pos.x, final_pos.x)
        self.assertGreater(self.runtime.simulation_time, 0.0)

    def test_16_deterministic_execution(self):
        """16. Two independent runtime runs execute identically down to float precision."""
        rt1 = SarthiMuJoCoRuntime()
        rt2 = SarthiMuJoCoRuntime()

        action = CandidateAction(
            action_type=ActionType.APPROACH,
            action_id="act_det",
            target_position=Point3D(x=0.25, y=0.15, z=0.25),
        )

        res1 = rt1.execute_action(action)
        res2 = rt2.execute_action(action)

        self.assertEqual(res1.success, res2.success)
        self.assertAlmostEqual(res1.simulation_time, res2.simulation_time, places=4)
        qpos1 = rt1.state_reader.get_robot_joint_positions()
        qpos2 = rt2.state_reader.get_robot_joint_positions()
        self.assertEqual(qpos1, qpos2)


if __name__ == "__main__":
    unittest.main()
