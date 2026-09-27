"""
Tests for SĀRTHI MuJoCo Physical World-State Reader (Phase D).
Validates:
1. State reader initialization
2. Robot joint positions
3. Robot joint velocities
4. End-effector position
5. Robot base position
6. Red object position
7. Blue target position
8. Obstacle position
9. Target tolerance
10. Gripper state derived physically
11. State conversion into canonical WorldState
12. Read-only behavior (no mutation of mjModel, mjData, or sim state)
13. Deterministic repeated reads
14. Physical state changes after stepping MuJoCo
15. Object position reflects actual mjData rather than scenario constants
"""

import math
import unittest
import numpy as np

from backend.app.decision_engine.models import (
    ObjectState,
    Point3D,
    WorldState,
)
from simulation.mujoco_runtime import (
    is_mujoco_available,
    SarthiMuJoCoSceneBuilder,
    SarthiMuJoCoArticulation,
    SarthiMuJoCoStateReader,
)


@unittest.skipUnless(is_mujoco_available(), "MuJoCo is required for Phase D state reader tests.")
class TestSarthiMuJoCoStateReader(unittest.TestCase):
    """Phase D test suite for physical WorldState extraction from MuJoCo."""

    def setUp(self):
        import mujoco

        self.builder = SarthiMuJoCoSceneBuilder()
        self.model, self.data = self.builder.build()

        # Reset Panda joints to home keyframe while preserving red object initial pos
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.assertGreaterEqual(key_id, 0)
        
        # Save red_object initial qpos
        red_obj_id = self.model.body("red_object").id
        init_obj_qpos = self.data.qpos[9:16].copy()

        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        # Restore red object position if keyframe had zeros
        self.data.qpos[9:16] = init_obj_qpos
        mujoco.mj_forward(self.model, self.data)

        self.reader = SarthiMuJoCoStateReader(self.model, self.data)

    def test_01_state_reader_initialization(self):
        """1. State reader initializes with valid references and default state."""
        self.assertIsNotNone(self.reader)
        self.assertEqual(self.reader.simulation_time, 0.0)
        self.assertEqual(len(self.reader._arm_joint_ids), 7)
        self.assertEqual(len(self.reader._finger_joint_ids), 2)
        self.assertGreaterEqual(self.reader._base_body_id, 0)
        self.assertGreaterEqual(self.reader._hand_body_id, 0)
        self.assertGreaterEqual(self.reader._red_object_body_id, 0)
        self.assertGreaterEqual(self.reader._blue_target_body_id, 0)
        self.assertGreaterEqual(self.reader._obstacle_body_id, 0)

    def test_02_robot_joint_positions(self):
        """2. Arm joint positions are read from live mjData.qpos."""
        positions = self.reader.get_robot_joint_positions()
        self.assertEqual(len(positions), 7)
        for i, val in enumerate(positions):
            self.assertTrue(math.isfinite(val))
            # At home keyframe: joint4 is -1.57079, joint6 is 1.57079, joint7 is -0.7853
            expected = float(self.data.qpos[self.reader._arm_qposadr[i]])
            self.assertAlmostEqual(val, expected, places=5)

    def test_03_robot_joint_velocities(self):
        """3. Arm joint velocities are read from live mjData.qvel."""
        velocities = self.reader.get_robot_joint_velocities()
        self.assertEqual(len(velocities), 7)
        for i, val in enumerate(velocities):
            self.assertTrue(math.isfinite(val))
            expected = float(self.data.qvel[self.reader._arm_dofadr[i]])
            self.assertAlmostEqual(val, expected, places=5)

    def test_04_end_effector_position(self):
        """4. End-effector Cartesian position matches calculated grasp center."""
        ee = self.reader.get_end_effector_position()
        self.assertIsInstance(ee, Point3D)
        self.assertTrue(math.isfinite(ee.x))
        self.assertTrue(math.isfinite(ee.y))
        self.assertTrue(math.isfinite(ee.z))

        # At home keyframe, grasp center is roughly (0.5545, 0.0, 0.5211)
        self.assertAlmostEqual(ee.x, 0.5545, delta=0.05)
        self.assertAlmostEqual(ee.y, 0.0, delta=0.05)
        self.assertAlmostEqual(ee.z, 0.5211, delta=0.05)

        # Pose reading
        pos, quat = self.reader.get_end_effector_pose()
        self.assertEqual(len(quat), 4)

    def test_05_robot_base(self):
        """5. Robot base (link0) position is read from mjData."""
        base = self.reader.get_robot_base_position()
        self.assertIsInstance(base, Point3D)
        self.assertAlmostEqual(base.x, 0.0, places=4)
        self.assertAlmostEqual(base.y, 0.0, places=4)
        self.assertAlmostEqual(base.z, 0.0, places=4)

    def test_06_red_object_position(self):
        """6. Red object Cartesian position is read from mjData.xpos."""
        obj_pos = self.reader.get_red_object_position()
        self.assertIsInstance(obj_pos, Point3D)
        # Canonical initial position in tabletop scenario is (0.25, 0.15, 0.20)
        self.assertAlmostEqual(obj_pos.x, 0.25, delta=0.02)
        self.assertAlmostEqual(obj_pos.y, 0.15, delta=0.02)
        self.assertAlmostEqual(obj_pos.z, 0.20, delta=0.02)

    def test_07_blue_target_position(self):
        """7. Blue target zone position is read from mjData.xpos."""
        tgt_pos = self.reader.get_blue_target_position()
        self.assertIsInstance(tgt_pos, Point3D)
        # Canonical target position in tabletop scenario is (0.40, -0.20, 0.171)
        self.assertAlmostEqual(tgt_pos.x, 0.40, places=3)
        self.assertAlmostEqual(tgt_pos.y, -0.20, places=3)
        self.assertAlmostEqual(tgt_pos.z, 0.171, places=3)

    def test_08_obstacle_position(self):
        """8. Dynamic path obstacle position is read from mjData.xpos."""
        obs_pos = self.reader.get_obstacle_position()
        self.assertIsInstance(obs_pos, Point3D)
        # Canonical obstacle position in tabletop scenario is (0.325, -0.025, 0.21)
        self.assertAlmostEqual(obs_pos.x, 0.325, places=3)
        self.assertAlmostEqual(obs_pos.y, -0.025, places=3)
        self.assertAlmostEqual(obs_pos.z, 0.21, places=3)

        # Dimensions: (0.06, 0.06, 0.08)
        dims = self.reader.get_obstacle_dimensions()
        self.assertAlmostEqual(dims[0], 0.06, places=3)
        self.assertAlmostEqual(dims[1], 0.06, places=3)
        self.assertAlmostEqual(dims[2], 0.08, places=3)

        self.assertTrue(self.reader.is_obstacle_active())

    def test_09_target_tolerance(self):
        """9. Blue target tolerance radius is extracted accurately."""
        tol = self.reader.get_blue_target_tolerance()
        self.assertAlmostEqual(tol, 0.06, delta=0.01)

    def test_10_gripper_state(self):
        """10. Gripper state is derived physically from finger joint separation."""
        import mujoco

        # At home keyframe: fingers are at 0.04 each -> separation 0.08m
        self.assertAlmostEqual(self.reader.get_gripper_separation(), 0.08, places=3)
        self.assertTrue(self.reader.is_gripper_open())

        # Manually close fingers in data
        f1_adr, f2_adr = self.reader._finger_qposadr
        self.data.qpos[f1_adr] = 0.001
        self.data.qpos[f2_adr] = 0.001
        mujoco.mj_forward(self.model, self.data)

        self.assertLess(self.reader.get_gripper_separation(), 0.01)
        self.assertFalse(self.reader.is_gripper_open())

    def test_11_state_conversion_canonical_world_state(self):
        """11. Translates physical state into canonical SĀRTHI WorldState."""
        ws = self.reader.read_world_state(version=42)

        self.assertIsInstance(ws, WorldState)
        self.assertEqual(ws.version, 42)
        self.assertIsInstance(ws.robot.position, Point3D)
        self.assertTrue(ws.robot.gripper_open)
        self.assertIsNone(ws.robot.holding_object_id)
        self.assertEqual(ws.robot.payload_mass_kg, 0.0)

        # Objects (red_object + obstacle)
        self.assertEqual(len(ws.objects), 2)
        red_obj = next(o for o in ws.objects if o.is_target)
        self.assertEqual(red_obj.name, self.reader._scenario.red_object.name)
        self.assertTrue(red_obj.is_target)
        self.assertFalse(red_obj.is_obstacle)
        self.assertEqual(red_obj.state, ObjectState.FREE)

        obstacle = next(o for o in ws.objects if o.is_obstacle)
        self.assertEqual(obstacle.id, self.reader._scenario.obstacle.obstacle_id)
        self.assertTrue(obstacle.is_obstacle)

        # Target zone
        self.assertEqual(ws.target.id, self.reader._scenario.blue_target.target_id)
        self.assertAlmostEqual(ws.target.position.x, 0.40, places=2)

        # Environment
        self.assertTrue(ws.environment.dynamic_obstacles_detected)

        # Constraints
        self.assertGreaterEqual(len(ws.active_constraints), 1)
        self.assertEqual(ws.active_constraints[0].constraint_id, "c_obstacle_keepout")

    def test_12_read_only_behavior(self):
        """12. Calling state reader methods does NOT mutate MuJoCo simulation state."""
        qpos_before = self.data.qpos.copy()
        qvel_before = self.data.qvel.copy()
        ctrl_before = self.data.ctrl.copy()
        time_before = float(self.data.time)

        # Perform multiple telemetry and world state reads
        _ = self.reader.get_robot_joint_positions()
        _ = self.reader.get_robot_joint_velocities()
        _ = self.reader.get_end_effector_position()
        _ = self.reader.get_red_object_position()
        _ = self.reader.is_object_grasped()
        _ = self.reader.read_world_state(version=10)

        # Verify exact bitwise preservation of physics data
        np.testing.assert_array_equal(self.data.qpos, qpos_before)
        np.testing.assert_array_equal(self.data.qvel, qvel_before)
        np.testing.assert_array_equal(self.data.ctrl, ctrl_before)
        self.assertEqual(float(self.data.time), time_before)

    def test_13_deterministic_repeated_reads(self):
        """13. Repeated reads without physics stepping yield identical WorldState data."""
        ws1 = self.reader.read_world_state(version=1)
        ws2 = self.reader.read_world_state(version=1)

        self.assertEqual(ws1.robot.position.x, ws2.robot.position.x)
        self.assertEqual(ws1.robot.position.y, ws2.robot.position.y)
        self.assertEqual(ws1.robot.position.z, ws2.robot.position.z)
        self.assertEqual(ws1.robot.gripper_open, ws2.robot.gripper_open)
        self.assertEqual(ws1.objects[0].position.x, ws2.objects[0].position.x)
        self.assertEqual(ws1.objects[0].position.y, ws2.objects[0].position.y)
        self.assertEqual(ws1.objects[0].position.z, ws2.objects[0].position.z)

    def test_14_physical_state_changes_after_stepping(self):
        """14. Stepping physics produces observable physical state changes in reader."""
        import mujoco

        art = SarthiMuJoCoArticulation(self.model, self.data)
        initial_ee = self.reader.get_end_effector_position()

        # Command a new target position and step physics
        target = Point3D(x=0.30, y=0.10, z=0.35)
        art.move_to_position(target)
        art.step(100)

        new_ee = self.reader.get_end_effector_position()
        self.assertNotEqual(initial_ee.x, new_ee.x)
        self.assertGreater(float(self.reader.simulation_time), 0.0)

        # Verified through canonical WorldState as well
        ws_after = self.reader.read_world_state()
        self.assertAlmostEqual(ws_after.robot.position.x, new_ee.x)
        self.assertAlmostEqual(ws_after.robot.position.z, new_ee.z)

    def test_15_object_position_reflects_actual_mjdata(self):
        """
        15. CRITICAL TEST: Direct modification of object qpos in mjData is immediately
        reflected in the WorldState output, proving reader does NOT use scenario constants.
        """
        import mujoco

        # Directly displace the red object in MuJoCo physics state
        new_x, new_y, new_z = 0.42, 0.28, 0.33
        self.data.qpos[9] = new_x
        self.data.qpos[10] = new_y
        self.data.qpos[11] = new_z

        # Update forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # Direct telemetry check
        measured_pos = self.reader.get_red_object_position()
        self.assertAlmostEqual(measured_pos.x, new_x, places=4)
        self.assertAlmostEqual(measured_pos.y, new_y, places=4)
        self.assertAlmostEqual(measured_pos.z, new_z, places=4)

        # Canonical WorldState check
        ws = self.reader.read_world_state()
        red_obj = next(o for o in ws.objects if o.is_target)
        self.assertAlmostEqual(red_obj.position.x, new_x, places=4)
        self.assertAlmostEqual(red_obj.position.y, new_y, places=4)
        self.assertAlmostEqual(red_obj.position.z, new_z, places=4)


if __name__ == "__main__":
    unittest.main()
