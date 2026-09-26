"""
SĀRTHI Phase 7A — Isaac Sim Articulation Boundary & Controller Tests.

IMPORTANT:
- ZERO dependencies on NVIDIA Isaac Sim packages.
- Tests execute completely offline without Isaac Sim installed.
- Verifies lazy imports, robot articulation binding, joint telemetry,
  Cartesian end-effector tracking, gripper commands, action dispatch,
  safety STOP hold behavior, asset configuration validation, and
  conversion to the canonical SĀRTHI WorldState schema.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

from backend.app.decision_engine.models import (
    ActionType,
    CandidateAction,
    Point3D,
    WorldState,
)
from simulation.isaac_runtime.articulation import SarthiArticulationController
from simulation.isaac_runtime.robot_scene import RobotSceneConfig, SarthiRobotPrim
from simulation.isaac_runtime.runtime import IsaacSimRuntimeError, SarthiIsaacRuntime
from simulation.scenarios.tabletop_pick_place import create_default_scenario


class TestIsaacArticulation(unittest.TestCase):
    """
    Unit test suite covering Phase 7A Isaac Sim Articulation Boundary.
    """

    def setUp(self):
        self.scenario = create_default_scenario()

    # 1. Lazy import behavior
    def test_01_lazy_import_behavior(self):
        """Verify articulation controller is importable without requiring Isaac Sim packages."""
        self.assertNotIn("isaacsim", sys.modules)
        self.assertNotIn("omni.isaac.core", sys.modules)

        controller = SarthiArticulationController(
            robot_id="tabletop_manipulator",
            usd_prim_path="/World/Robots/tabletop_manipulator",
        )
        self.assertEqual(controller.robot_id, "tabletop_manipulator")
        self.assertFalse(controller.is_bound)
        self.assertFalse(controller.is_stopped)
        self.assertEqual(controller.gripper_state, "OPEN")

    # 2. Isaac Sim unavailable behavior
    def test_02_isaac_sim_unavailable_behavior(self):
        """Verify controller methods raise IsaacSimRuntimeError when executed outside Isaac Sim without binding."""
        controller = SarthiArticulationController()

        with self.assertRaises(IsaacSimRuntimeError) as ctx:
            controller.get_joint_positions()
        self.assertIn("Isaac Sim runtime is required", str(ctx.exception))

        with self.assertRaises(IsaacSimRuntimeError):
            controller.get_joint_velocities()

        with self.assertRaises(IsaacSimRuntimeError):
            controller.get_end_effector_position()

        with self.assertRaises(IsaacSimRuntimeError):
            controller.command_gripper("OPEN")

        with self.assertRaises(IsaacSimRuntimeError):
            controller.command_cartesian_position(Point3D(x=0.3, y=0.0, z=0.2))

        with self.assertRaises(IsaacSimRuntimeError):
            controller.command_joint_positions([0.0] * 7)

        with self.assertRaises(IsaacSimRuntimeError):
            controller.stop()

    # 3. Robot binding
    def test_03_robot_binding(self):
        """Verify articulation controller binds to instantiated robot prim and reads initial state."""
        controller = SarthiArticulationController()

        # Binding None must raise
        with self.assertRaises(IsaacSimRuntimeError):
            controller.bind_robot(None)

        mock_robot = MagicMock()
        mock_robot.get_joint_positions.return_value = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7]
        mock_robot.get_joint_velocities.return_value = [0.0] * 7
        mock_gripper = MagicMock()
        mock_robot.gripper = mock_gripper

        controller.bind_robot(mock_robot)
        self.assertTrue(controller.is_bound)
        self.assertEqual(len(controller.get_joint_positions()), 7)
        self.assertAlmostEqual(controller.get_joint_positions()[1], -0.5)

    # 4. Joint state retrieval
    def test_04_joint_state_retrieval(self):
        """Verify joint positions and velocities are retrieved accurately as floats."""
        controller = SarthiArticulationController()
        mock_robot = MagicMock()
        mock_robot.get_joint_positions.return_value = [0.1, -0.7, 0.2, -1.8, 0.3, 1.2, 0.8]
        mock_robot.get_joint_velocities.return_value = [0.01, -0.02, 0.03, -0.04, 0.05, 0.06, 0.07]
        controller.bind_robot(mock_robot)

        positions = controller.get_joint_positions()
        velocities = controller.get_joint_velocities()

        self.assertEqual(len(positions), 7)
        self.assertAlmostEqual(positions[0], 0.1)
        self.assertAlmostEqual(velocities[1], -0.02)
        for p in positions:
            self.assertIsInstance(p, float)
        for v in velocities:
            self.assertIsInstance(v, float)

    # 5. End-effector state retrieval
    def test_05_end_effector_state_retrieval(self):
        """Verify Cartesian end-effector position and orientation quaternion retrieval."""
        controller = SarthiArticulationController()
        mock_robot = MagicMock()
        mock_ee = MagicMock()
        mock_ee.get_world_pose.return_value = ([0.42, -0.15, 0.28], [1.0, 0.0, 0.0, 0.0])
        mock_robot.end_effector = mock_ee
        controller.bind_robot(mock_robot)

        pos = controller.get_end_effector_position()
        self.assertIsInstance(pos, Point3D)
        self.assertAlmostEqual(pos.x, 0.42)
        self.assertAlmostEqual(pos.y, -0.15)
        self.assertAlmostEqual(pos.z, 0.28)

        full_pos, rot = controller.get_end_effector_pose()
        self.assertEqual(full_pos.x, 0.42)
        self.assertEqual(rot, (1.0, 0.0, 0.0, 0.0))

    # 6. Gripper commands
    def test_06_gripper_commands(self):
        """Verify gripper OPEN, CLOSE, and HOLD commands and state transitions."""
        controller = SarthiArticulationController()
        mock_robot = MagicMock()
        mock_gripper = MagicMock()
        mock_robot.gripper = mock_gripper
        controller.bind_robot(mock_robot)

        # CLOSE
        controller.close_gripper()
        self.assertEqual(controller.gripper_state, "CLOSED")
        self.assertTrue(controller.is_gripper_closed)
        mock_gripper.close.assert_called_once()

        # HOLD
        controller.hold_gripper()
        self.assertEqual(controller.gripper_state, "HOLDING")
        self.assertFalse(controller.is_gripper_closed)

        # OPEN
        controller.open_gripper()
        self.assertEqual(controller.gripper_state, "OPEN")
        self.assertFalse(controller.is_gripper_closed)
        mock_gripper.open.assert_called_once()

        # Unsupported command raises ValueError
        with self.assertRaises(ValueError):
            controller.command_gripper("INVALID_CMD")

    # 7. Action dispatch: APPROACH, REPOSITION, MOVE
    def test_07_action_dispatch_motion(self):
        """Verify CandidateAction dispatch for APPROACH, REPOSITION, and MOVE."""
        controller = SarthiArticulationController()
        mock_robot = MagicMock()
        controller.bind_robot(mock_robot)

        # APPROACH
        approach = CandidateAction(
            action_id="act_approach",
            action_type=ActionType.APPROACH,
            target_object_id="red_object_01",
            target_position=Point3D(x=0.25, y=0.15, z=0.20),
        )
        res_approach = controller.dispatch_action(approach)
        self.assertTrue(res_approach.success)
        self.assertEqual(res_approach.action_type, "APPROACH")
        self.assertEqual(controller.gripper_state, "OPEN")
        self.assertAlmostEqual(controller.get_end_effector_position().x, 0.25)

        # REPOSITION
        reposition = CandidateAction(
            action_id="act_reposition",
            action_type=ActionType.REPOSITION,
            target_position=Point3D(x=0.25, y=-0.15, z=0.35),
        )
        res_repo = controller.dispatch_action(reposition)
        self.assertTrue(res_repo.success)
        self.assertEqual(res_repo.action_type, "REPOSITION")
        self.assertEqual(controller.gripper_state, "HOLDING")
        self.assertAlmostEqual(controller.get_end_effector_position().y, -0.15)

        # MOVE
        move = CandidateAction(
            action_id="act_move",
            action_type=ActionType.MOVE,
            target_position=Point3D(x=0.35, y=0.20, z=0.20),
        )
        res_move = controller.dispatch_action(move)
        self.assertTrue(res_move.success)
        self.assertEqual(res_move.action_type, "MOVE")
        self.assertAlmostEqual(controller.get_end_effector_position().x, 0.35)

    # 8. Action dispatch: GRASP and RELEASE
    def test_08_action_dispatch_grasp_and_release(self):
        """Verify CandidateAction dispatch for GRASP and RELEASE with holding tracking."""
        controller = SarthiArticulationController()
        mock_robot = MagicMock()
        controller.bind_robot(mock_robot)

        # GRASP
        grasp = CandidateAction(
            action_id="act_grasp",
            action_type=ActionType.GRASP,
            target_object_id="red_object_01",
            expected_force_n=6.0,
        )
        res_grasp = controller.dispatch_action(grasp)
        self.assertTrue(res_grasp.success)
        self.assertEqual(res_grasp.action_type, "GRASP")
        self.assertEqual(controller.gripper_state, "CLOSED")
        self.assertTrue(controller.is_holding_object)
        self.assertEqual(controller.holding_object_id, "red_object_01")

        # RELEASE
        release = CandidateAction(
            action_id="act_release",
            action_type=ActionType.RELEASE,
            target_object_id="red_object_01",
        )
        res_release = controller.dispatch_action(release)
        self.assertTrue(res_release.success)
        self.assertEqual(res_release.action_type, "RELEASE")
        self.assertEqual(controller.gripper_state, "OPEN")
        self.assertFalse(controller.is_holding_object)
        self.assertIsNone(controller.holding_object_id)

    # 9. Safety STOP behavior
    def test_09_safety_stop_behavior(self):
        """Verify STOP immediately halts articulation, zeros velocities, and blocks commands until resumed."""
        controller = SarthiArticulationController()
        mock_robot = MagicMock()
        mock_robot.get_joint_positions.return_value = [0.0] * 7
        mock_robot.get_joint_velocities.return_value = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        controller.bind_robot(mock_robot)

        # Trigger STOP
        stop_action = CandidateAction(
            action_id="act_stop",
            action_type=ActionType.STOP,
        )
        res_stop = controller.dispatch_action(stop_action)
        self.assertTrue(res_stop.success)
        self.assertEqual(res_stop.action_type, "STOP")
        self.assertTrue(controller.is_stopped)

        # Verify joint velocities were zeroed
        self.assertTrue(all(v == 0.0 for v in controller.get_joint_velocities()))
        mock_robot.set_joint_velocities.assert_called_with([0.0] * 7)

        # Attempting Cartesian motion while stopped must raise IsaacSimRuntimeError
        with self.assertRaises(IsaacSimRuntimeError) as ctx:
            controller.command_cartesian_position(Point3D(x=0.1, y=0.1, z=0.1))
        self.assertIn("STOP state", str(ctx.exception))

        # Attempting joint motion while stopped must raise IsaacSimRuntimeError
        with self.assertRaises(IsaacSimRuntimeError) as ctx_joint:
            controller.command_joint_positions([0.1] * 7)
        self.assertIn("STOP state", str(ctx_joint.exception))

        # Resume clears stop state
        controller.resume()
        self.assertFalse(controller.is_stopped)

    # 10. Missing or invalid asset failure
    def test_10_missing_or_invalid_asset_failure(self):
        """Verify missing or nonexistent robot asset path raises IsaacSimRuntimeError clearly."""
        # 1. Missing / None asset path
        cfg_none = RobotSceneConfig(robot_id="franka", robot_asset_path=None)
        prim_none = SarthiRobotPrim(config=cfg_none)
        with patch("simulation.isaac_runtime.robot_scene.is_isaac_sim_available", return_value=True):
            with self.assertRaises(IsaacSimRuntimeError) as ctx_none:
                prim_none.create()
            self.assertIn("Missing robot asset path", str(ctx_none.exception))

        # 2. Empty string asset path
        cfg_empty = RobotSceneConfig(robot_id="franka", robot_asset_path="   ")
        prim_empty = SarthiRobotPrim(config=cfg_empty)
        with patch("simulation.isaac_runtime.robot_scene.is_isaac_sim_available", return_value=True):
            with self.assertRaises(IsaacSimRuntimeError) as ctx_empty:
                prim_empty.create()
            self.assertIn("Missing robot asset path", str(ctx_empty.exception))

        # 3. Nonexistent local file path
        cfg_bad_local = RobotSceneConfig(robot_id="franka", robot_asset_path="/nonexistent/franka.usd")
        prim_bad = SarthiRobotPrim(config=cfg_bad_local)
        with patch("simulation.isaac_runtime.robot_scene.is_isaac_sim_available", return_value=True):
            with self.assertRaises(IsaacSimRuntimeError) as ctx_bad:
                prim_bad.create()
            self.assertIn("not found on disk", str(ctx_bad.exception))

    # 11. Live world state conversion
    def test_11_live_world_state_conversion(self):
        """Verify live Isaac Sim telemetry is cleanly converted to canonical SĀRTHI WorldState."""
        runtime = SarthiIsaacRuntime(scenario=self.scenario)
        mock_robot = MagicMock()
        mock_robot.get_joint_positions.return_value = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        mock_robot.get_joint_velocities.return_value = [0.0] * 7
        mock_ee = MagicMock()
        mock_ee.get_world_pose.return_value = ([0.25, 0.15, 0.20], [1.0, 0.0, 0.0, 0.0])
        mock_robot.end_effector = mock_ee

        # Bind controller
        runtime.articulation_controller.bind_robot(mock_robot)

        with patch("simulation.isaac_runtime.runtime.is_isaac_sim_available", return_value=True):
            runtime._is_initialized = True
            world_state = runtime.read_live_world_state()

        self.assertIsInstance(world_state, WorldState)
        self.assertAlmostEqual(world_state.robot.position.x, 0.25)
        self.assertEqual(len(world_state.objects), 1)
        self.assertEqual(world_state.objects[0].id, self.scenario.red_object.object_id)
        self.assertEqual(world_state.target.id, self.scenario.blue_target.target_id)
        self.assertFalse(world_state.environment.dynamic_obstacles_detected)

        # Telemetry query
        telemetry = runtime.get_telemetry()
        self.assertEqual(len(telemetry["joint_positions"]), 7)
        self.assertAlmostEqual(telemetry["end_effector_position"].x, 0.25)

    # 12. No decision logic in articulation controller
    def test_12_no_decision_logic_in_articulation_controller(self):
        """Verify articulation controller contains zero cognitive / decision logic."""
        controller = SarthiArticulationController()
        forbidden_methods = [
            "decide",
            "evaluate_candidates",
            "score_candidate",
            "check_constraints",
            "plan_detour",
            "select_action",
        ]
        for m in forbidden_methods:
            self.assertFalse(
                hasattr(controller, m),
                f"SarthiArticulationController should not implement decision method '{m}'",
            )
