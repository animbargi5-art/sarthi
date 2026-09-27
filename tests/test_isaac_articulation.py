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
    ObjectState,
    Point3D,
    WorldState,
)
from simulation.adapters.isaac_sim import (
    IsaacSimAction,
    IsaacSimAdapter,
    IsaacSimDisturbance,
)
from simulation.core.events import DisturbanceEvent
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

    # 13. Object pose persists after release
    def test_13_object_pose_persists_after_release(self):
        """Verify object pose follows payload while held and remains at released coordinates after RELEASE."""
        runtime = SarthiIsaacRuntime(scenario=self.scenario)
        mock_robot = MagicMock()
        mock_robot.get_joint_positions.return_value = [0.0] * 7
        mock_robot.get_joint_velocities.return_value = [0.0] * 7
        mock_ee = MagicMock()
        initial_ee = ([0.25, 0.15, 0.20], [1.0, 0.0, 0.0, 0.0])
        mock_ee.get_world_pose.return_value = initial_ee
        mock_robot.end_effector = mock_ee

        runtime.articulation_controller.bind_robot(mock_robot)

        with patch("simulation.isaac_runtime.runtime.is_isaac_sim_available", return_value=True):
            runtime._is_initialized = True

            # 1. Execute GRASP at initial object coordinates
            act_grasp = CandidateAction(
                action_type=ActionType.GRASP,
                action_id="act_grasp_1",
                target_object_id=self.scenario.red_object.object_id,
                target_position=Point3D(x=0.25, y=0.15, z=0.20),
            )
            res_grasp = runtime.execute_action(act_grasp)
            self.assertTrue(res_grasp.success)

            # Object follows end-effector while held
            ws_grasp = runtime.read_live_world_state()
            self.assertEqual(ws_grasp.objects[0].position, Point3D(x=0.25, y=0.15, z=0.20))
            self.assertEqual(ws_grasp.objects[0].state, ObjectState.GRASPED)

            # 2. Move to blue target destination
            target_pos = self.scenario.blue_target.target_pose
            mock_ee.get_world_pose.return_value = ([target_pos.x, target_pos.y, target_pos.z], [1.0, 0.0, 0.0, 0.0])
            act_move = CandidateAction(
                action_type=ActionType.MOVE,
                action_id="act_move_1",
                target_position=target_pos,
            )
            res_move = runtime.execute_action(act_move)
            self.assertTrue(res_move.success)

            ws_move = runtime.read_live_world_state()
            self.assertEqual(ws_move.objects[0].position, target_pos)

            # 3. Execute RELEASE at destination
            act_release = CandidateAction(
                action_type=ActionType.RELEASE,
                action_id="act_rel_1",
                target_object_id=self.scenario.red_object.object_id,
            )
            res_rel = runtime.execute_action(act_release)
            self.assertTrue(res_rel.success)

            # 4. Robot moves away to a home waypoint
            mock_ee.get_world_pose.return_value = ([0.0, 0.0, 0.30], [1.0, 0.0, 0.0, 0.0])
            act_reposition = CandidateAction(
                action_type=ActionType.REPOSITION,
                action_id="act_repo_1",
                target_position=Point3D(x=0.0, y=0.0, z=0.30),
            )
            runtime.execute_action(act_reposition)

            # 5. Subsequent WorldState must contain released destination, NOT initial pose
            ws_subsequent = runtime.read_live_world_state()
            self.assertEqual(ws_subsequent.objects[0].position, target_pos)
            self.assertEqual(ws_subsequent.objects[0].state, ObjectState.PLACED)
            self.assertNotEqual(ws_subsequent.objects[0].position, self.scenario.red_object.initial_pose)

    # 14. Action verification: APPROACH success and failure
    def test_14_runtime_action_verification_approach_success_and_failure(self):
        """Verify explicit position tolerance verification for APPROACH."""
        runtime = SarthiIsaacRuntime(scenario=self.scenario)
        mock_robot = MagicMock()
        mock_robot.get_joint_positions.return_value = [0.0] * 7
        mock_robot.get_joint_velocities.return_value = [0.0] * 7
        mock_ee = MagicMock()
        mock_ee.get_world_pose.return_value = ([0.25, 0.15, 0.20], [1.0, 0.0, 0.0, 0.0])
        mock_robot.end_effector = mock_ee

        runtime.articulation_controller.bind_robot(mock_robot)

        with patch("simulation.isaac_runtime.runtime.is_isaac_sim_available", return_value=True):
            runtime._is_initialized = True

            # Matching target -> success
            act_success = CandidateAction(
                action_type=ActionType.APPROACH,
                action_id="act_app_ok",
                target_position=Point3D(x=0.25, y=0.15, z=0.20),
            )
            runtime.execute_action(act_success)
            self.assertTrue(runtime.verify_action_result(act_success))
            self.assertIsNone(runtime.last_verification_failure)

            # Far target outside tolerance -> failure with useful diagnostic reason
            act_fail = CandidateAction(
                action_type=ActionType.APPROACH,
                action_id="act_app_fail",
                target_position=Point3D(x=0.60, y=0.60, z=0.50),
            )
            self.assertFalse(runtime.verify_action_result(act_fail))
            self.assertIsNotNone(runtime.last_verification_failure)
            self.assertIn("APPROACH verification failed", runtime.last_verification_failure)

    # 15. Action verification: GRASP success and failure
    def test_15_runtime_action_verification_grasp_success_and_failure(self):
        """Verify explicit physical verification for GRASP (gripper closed, attached payload, radius)."""
        runtime = SarthiIsaacRuntime(scenario=self.scenario)
        mock_robot = MagicMock()
        mock_robot.get_joint_positions.return_value = [0.0] * 7
        mock_robot.get_joint_velocities.return_value = [0.0] * 7
        mock_ee = MagicMock()
        mock_ee.get_world_pose.return_value = ([0.25, 0.15, 0.20], [1.0, 0.0, 0.0, 0.0])
        mock_robot.end_effector = mock_ee

        runtime.articulation_controller.bind_robot(mock_robot)

        with patch("simulation.isaac_runtime.runtime.is_isaac_sim_available", return_value=True):
            runtime._is_initialized = True

            act = CandidateAction(
                action_type=ActionType.GRASP,
                action_id="act_g_1",
                target_object_id=self.scenario.red_object.object_id,
                target_position=Point3D(x=0.25, y=0.15, z=0.20),
            )
            runtime.execute_action(act)
            self.assertTrue(runtime.verify_action_result(act))

            # Failure condition 1: Gripper is unexpectedly open
            runtime.articulation_controller.open_gripper()
            self.assertFalse(runtime.verify_action_result(act))
            self.assertIn("Gripper is not closed", runtime.last_verification_failure)

            # Failure condition 2: Wrong payload ID expected
            runtime.articulation_controller.close_gripper()
            runtime.articulation_controller._is_holding_object = True
            runtime.articulation_controller._holding_object_id = "wrong_payload"
            self.assertFalse(runtime.verify_action_result(act, {"carrying_object_id": self.scenario.red_object.object_id}))
            self.assertIn("Attached object 'wrong_payload' != expected", runtime.last_verification_failure)

    # 16. Action verification: RELEASE success
    def test_16_runtime_action_verification_release_success(self):
        """Verify explicit physical verification for RELEASE."""
        runtime = SarthiIsaacRuntime(scenario=self.scenario)
        mock_robot = MagicMock()
        mock_robot.get_joint_positions.return_value = [0.0] * 7
        mock_robot.get_joint_velocities.return_value = [0.0] * 7
        mock_ee = MagicMock()
        target_pos = self.scenario.blue_target.target_pose
        mock_ee.get_world_pose.return_value = ([target_pos.x, target_pos.y, target_pos.z], [1.0, 0.0, 0.0, 0.0])
        mock_robot.end_effector = mock_ee

        runtime.articulation_controller.bind_robot(mock_robot)

        with patch("simulation.isaac_runtime.runtime.is_isaac_sim_available", return_value=True):
            runtime._is_initialized = True

            # First hold object
            act_grasp = CandidateAction(
                action_type=ActionType.GRASP,
                action_id="act_g",
                target_object_id=self.scenario.red_object.object_id,
                target_position=target_pos,
            )
            runtime.execute_action(act_grasp)

            # Execute RELEASE
            act_rel = CandidateAction(
                action_type=ActionType.RELEASE,
                action_id="act_rel",
                target_object_id=self.scenario.red_object.object_id,
                target_position=target_pos,
            )
            res = runtime.execute_action(act_rel)
            self.assertTrue(res.success)
            self.assertTrue(runtime.verify_action_result(act_rel))

            # Failure condition: gripper still closed
            runtime.articulation_controller._gripper_state = "CLOSED"
            self.assertFalse(runtime.verify_action_result(act_rel))
            self.assertIn("Gripper is still closed", runtime.last_verification_failure)

    # 17. Structured grasp failure
    def test_17_grasp_failure_returns_structured_error(self):
        """Verify grasp failures return explicit ActionExecutionResult(success=False) without state corruption."""
        controller = SarthiArticulationController()
        mock_robot = MagicMock()
        mock_robot.get_joint_positions.return_value = [0.0] * 7
        mock_robot.get_joint_velocities.return_value = [0.0] * 7
        mock_ee = MagicMock()
        mock_ee.get_world_pose.return_value = ([0.0, 0.0, 0.20], [1.0, 0.0, 0.0, 0.0])
        mock_robot.end_effector = mock_ee
        controller.bind_robot(mock_robot)

        # 1. Robot stopped failure
        controller.stop()
        act_stopped = CandidateAction(
            action_type=ActionType.GRASP,
            action_id="act_fail_stop",
            target_object_id="red_box",
        )
        res_stopped = controller.dispatch_action(act_stopped, sim_time=1.0, world_version=5)
        self.assertFalse(res_stopped.success)
        self.assertEqual(res_stopped.new_world_state_version, 5)
        self.assertIn("safety STOP", res_stopped.failure_reason)

        # 2. Missing payload failure
        controller.resume()
        act_missing = CandidateAction(
            action_type=ActionType.GRASP,
            action_id="act_fail_missing",
            target_object_id="",
        )
        res_missing = controller.dispatch_action(act_missing, sim_time=2.0, world_version=5)
        self.assertFalse(res_missing.success)
        self.assertIn("missing target_object_id", res_missing.failure_reason)

        # 3. Object outside grasp radius failure
        act_out_of_reach = CandidateAction(
            action_type=ActionType.GRASP,
            action_id="act_fail_radius",
            target_object_id="red_box",
            target_position=Point3D(x=0.50, y=0.50, z=0.20),
        )
        res_radius = controller.dispatch_action(act_out_of_reach, sim_time=3.0, world_version=5)
        self.assertFalse(res_radius.success)
        self.assertIn("outside grasp radius", res_radius.failure_reason)
        self.assertEqual(res_radius.details.get("error_code"), "OBJECT_OUT_OF_REACH")
        self.assertFalse(controller.is_holding_object)

    # 18. Disturbance parameter alignment
    def test_18_disturbance_injection_accepts_both_event_and_isaac_disturbance(self):
        """Verify disturbance interface accepts DisturbanceEvent and IsaacSimDisturbance interchangeably."""
        runtime = SarthiIsaacRuntime(scenario=self.scenario)
        runtime.disturbance.spawn = MagicMock()

        # 1. DisturbanceEvent
        from simulation.core.geometry import SimPoint3D
        event = DisturbanceEvent.create_path_blocked(
            event_id="dist_align_1",
            obstacle_id="obs_align_1",
            position=SimPoint3D(x=0.25, y=0.05, z=0.20),
        )
        with patch("simulation.isaac_runtime.runtime.is_isaac_sim_available", return_value=True):
            runtime._is_initialized = True
            ok1 = runtime.inject_disturbance(event)
        self.assertTrue(ok1)
        self.assertEqual(runtime.disturbance.position.y, 0.05)

        # 2. IsaacSimDisturbance
        isaac_dist = IsaacSimDisturbance(
            event_id="dist_align_2",
            disturbance_type="PATH_BLOCKED",
            usd_prim_path="/World/Obstacles/obs_align_2",
            position={"x": 0.28, "y": -0.04, "z": 0.20},
            dimensions={"length_x": 0.08, "width_y": 0.08, "height_z": 0.20},
        )
        with patch("simulation.isaac_runtime.runtime.is_isaac_sim_available", return_value=True):
            ok2 = runtime.inject_disturbance(isaac_dist)
        self.assertTrue(ok2)
        self.assertEqual(runtime.disturbance.position.x, 0.28)

        # 3. Adapter boundary accepts both
        mock_backend = MagicMock()
        mock_backend.inject_disturbance.return_value = True
        adapter = IsaacSimAdapter(sim_backend=mock_backend)
        self.assertTrue(adapter.inject_disturbance(event))
        self.assertTrue(adapter.inject_disturbance(isaac_dist))
        self.assertEqual(mock_backend.inject_disturbance.call_count, 2)

    # 19. Full closed-loop validation with Isaac runtime backend
    def test_19_full_closed_loop_validation_with_isaac_runtime_backend(self):
        """Verify full nominal task sequence across IsaacSimAdapter backed by SarthiIsaacRuntime."""
        runtime = SarthiIsaacRuntime(scenario=self.scenario)
        mock_robot = MagicMock()
        mock_robot.get_joint_positions.return_value = [0.0] * 7
        mock_robot.get_joint_velocities.return_value = [0.0] * 7
        mock_ee = MagicMock()
        mock_ee.get_world_pose.return_value = ([0.0, 0.0, 0.20], [1.0, 0.0, 0.0, 0.0])
        mock_robot.end_effector = mock_ee

        runtime.articulation_controller.bind_robot(mock_robot)

        with patch("simulation.isaac_runtime.runtime.is_isaac_sim_available", return_value=True):
            runtime._is_initialized = True
            adapter = IsaacSimAdapter(sim_backend=runtime)

            # Step 1: APPROACH
            red_pos = self.scenario.red_object.initial_pose
            act_app = CandidateAction(
                action_type=ActionType.APPROACH,
                action_id="seq_app",
                target_position=red_pos,
            )
            res_app = adapter.execute_action(act_app)
            self.assertTrue(res_app.success)
            mock_ee.get_world_pose.return_value = ([red_pos.x, red_pos.y, red_pos.z], [1.0, 0.0, 0.0, 0.0])
            self.assertTrue(adapter.verify_action_result(act_app))

            # Step 2: GRASP
            act_grasp = CandidateAction(
                action_type=ActionType.GRASP,
                action_id="seq_grasp",
                target_object_id=self.scenario.red_object.object_id,
                target_position=red_pos,
            )
            res_grasp = adapter.execute_action(act_grasp)
            self.assertTrue(res_grasp.success)
            self.assertTrue(adapter.verify_action_result(act_grasp))

            # Step 3: MOVE to blue target
            blue_pos = self.scenario.blue_target.target_pose
            act_move = CandidateAction(
                action_type=ActionType.MOVE,
                action_id="seq_move",
                target_position=blue_pos,
            )
            res_move = adapter.execute_action(act_move)
            self.assertTrue(res_move.success)
            mock_ee.get_world_pose.return_value = ([blue_pos.x, blue_pos.y, blue_pos.z], [1.0, 0.0, 0.0, 0.0])
            self.assertTrue(adapter.verify_action_result(act_move))

            # Step 4: RELEASE
            act_rel = CandidateAction(
                action_type=ActionType.RELEASE,
                action_id="seq_rel",
                target_object_id=self.scenario.red_object.object_id,
                target_position=blue_pos,
            )
            res_rel = adapter.execute_action(act_rel)
            self.assertTrue(res_rel.success)
            self.assertTrue(adapter.verify_action_result(act_rel))

            # Verify final WorldState meets scenario success criteria
            final_ws = adapter.get_world_state()
            self.assertEqual(final_ws.objects[0].position, blue_pos)
            self.assertEqual(final_ws.objects[0].state, ObjectState.PLACED)
            is_done, failure_reason = self.scenario.check_success_conditions(final_ws)
            self.assertTrue(is_done, f"Scenario success criteria failed: {failure_reason}")

    # 20. Configurable physics stepping in execute_action
    def test_20_runtime_physics_stepping_configuration_and_execution(self):
        """Verify physics_steps_per_action is configurable and advances simulation ticks during execute_action."""
        runtime = SarthiIsaacRuntime(scenario=self.scenario, physics_steps_per_action=45)
        self.assertEqual(runtime.physics_steps_per_action, 45)

        mock_robot = MagicMock()
        mock_robot.get_joint_positions.return_value = [0.0] * 7
        mock_robot.get_joint_velocities.return_value = [0.0] * 7
        mock_ee = MagicMock()
        mock_ee.get_world_pose.return_value = ([0.0, 0.0, 0.20], [1.0, 0.0, 0.0, 0.0])
        mock_robot.end_effector = mock_ee
        runtime.articulation_controller.bind_robot(mock_robot)

        mock_world = MagicMock()
        runtime._world = mock_world

        with patch("simulation.isaac_runtime.runtime.is_isaac_sim_available", return_value=True):
            runtime._is_initialized = True

            act = CandidateAction(
                action_type=ActionType.APPROACH,
                action_id="act_step_test",
                target_position=Point3D(x=0.25, y=0.15, z=0.20),
            )
            # Execute with default configured steps (45 ticks)
            res = runtime.execute_action(act)
            self.assertTrue(res.success)
            self.assertEqual(mock_world.step.call_count, 45)
            self.assertAlmostEqual(runtime._simulation_time, 45 * (1.0 / 60.0), places=4)

            # Execute with explicit override (10 ticks)
            mock_world.step.reset_mock()
            res2 = runtime.execute_action(act, physics_steps=10)
            self.assertTrue(res2.success)
            self.assertEqual(mock_world.step.call_count, 10)

    # 21. Articulation controller Cartesian forwarding to robot controller
    def test_21_articulation_controller_cartesian_forwarding_to_robot(self):
        """Verify command_cartesian_position forwards target coordinates to robot controller prim."""
        controller = SarthiArticulationController()
        mock_robot = MagicMock()
        mock_ctrl = MagicMock()
        mock_robot.controller = mock_ctrl
        controller.bind_robot(mock_robot)

        controller.command_cartesian_position(Point3D(x=0.25, y=0.15, z=0.20))
        mock_ctrl.forward.assert_called_once_with(target_position=[0.25, 0.15, 0.20])
        mock_robot.apply_action.assert_called_once()

    # 22. SarthiTaskRunner + IsaacSimAdapter APPROACH milestone
    def test_22_task_runner_isaac_sim_adapter_approach_milestone(self):
        """Verify SarthiTaskRunner with IsaacSimAdapter executes APPROACH and stops after step 1 when max_steps=1."""
        from backend.app.model.mock_provider import MockModelProvider
        from backend.app.orchestration.task_runner import SarthiTaskRunner

        runtime = SarthiIsaacRuntime(scenario=self.scenario)
        mock_robot = MagicMock()
        mock_robot.get_joint_positions.return_value = [0.0] * 7
        mock_robot.get_joint_velocities.return_value = [0.0] * 7
        mock_ee = MagicMock()
        target_pos = self.scenario.red_object.initial_pose
        # Initial robot pose is origin
        mock_ee.get_world_pose.return_value = ([0.0, 0.0, 0.20], [1.0, 0.0, 0.0, 0.0])
        mock_robot.end_effector = mock_ee
        runtime.articulation_controller.bind_robot(mock_robot)

        with patch("simulation.isaac_runtime.runtime.is_isaac_sim_available", return_value=True):
            runtime._is_initialized = True
            adapter = IsaacSimAdapter(sim_backend=runtime)

            runner = SarthiTaskRunner(adapter=adapter, max_steps=1)

            # When execute_action is called for APPROACH, update mock_ee to target_pos so verification passes
            def on_approach(act):
                mock_ee.get_world_pose.return_value = ([target_pos.x, target_pos.y, target_pos.z], [1.0, 0.0, 0.0, 0.0])
            mock_robot.apply_action.side_effect = on_approach

            result = runner.run_instruction(
                instruction="Move the red object to the blue target.",
                model_provider=MockModelProvider(),
            )

            # Verify exactly 1 step executed (APPROACH) and cleanly stopped
            self.assertEqual(result.executed_actions, [ActionType.APPROACH.value])
            self.assertEqual(len(result.verification_results), 1)
            self.assertTrue(result.verification_results[0].verified)
            self.assertEqual(result.run_status, "MAX_STEPS_REACHED")
            self.assertFalse(result.completed)  # Milestone stops after Step 1 without running full task
