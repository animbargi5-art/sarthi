"""
SĀRTHI Phase 6C — Isaac Sim Runtime Boundary Tests.

IMPORTANT:
- ZERO dependencies on NVIDIA Isaac Sim packages.
- Tests do NOT launch Isaac Sim.
- Verifies runtime package structure, configuration validation, missing runtime guards,
  sourcing of constants from TabletopPickPlaceScenario, and absence of Decision Engine logic.
"""

import inspect
import sys
import unittest
from unittest.mock import patch

from backend.app.decision_engine.models import WorldState
import simulation.isaac_runtime as isaac_pkg
from simulation.isaac_runtime.disturbances import PathBlockedDisturbance
from simulation.isaac_runtime.objects import BlueTargetPrim, RedObjectPrim, TablePrim
from simulation.isaac_runtime.robot_scene import RobotSceneConfig, SarthiRobotPrim
from simulation.isaac_runtime.runtime import IsaacSimRuntimeError, SarthiIsaacRuntime
from simulation.isaac_runtime.scene_builder import SarthiSceneBuilder, SceneCameraConfig
from simulation.scenarios.tabletop_pick_place import create_default_scenario


class TestIsaacRuntimeBoundary(unittest.TestCase):
    """
    Unit test suite covering Phase 6C Isaac Sim Runtime Boundary.
    """

    # 1. Runtime package structure
    def test_01_runtime_package_structure(self):
        """Verify package exports all required runtime, builder, and prim classes."""
        expected_exports = [
            "IsaacSimRuntimeError",
            "SarthiIsaacRuntime",
            "SarthiSceneBuilder",
            "SceneCameraConfig",
            "SarthiRobotPrim",
            "RobotSceneConfig",
            "RedObjectPrim",
            "BlueTargetPrim",
            "TablePrim",
            "PathBlockedDisturbance",
        ]
        for symbol in expected_exports:
            self.assertTrue(
                hasattr(isaac_pkg, symbol),
                f"simulation.isaac_runtime missing expected export: '{symbol}'",
            )

    # 2. Configuration validation
    def test_02_configuration_validation(self):
        """Verify camera, robot, and table configuration properties."""
        cam_config = SceneCameraConfig(
            fov_deg=60.0,
            focal_length_mm=35.0,
        )
        self.assertEqual(cam_config.fov_deg, 60.0)
        self.assertEqual(cam_config.focal_length_mm, 35.0)
        self.assertEqual(cam_config.usd_prim_path, "/World/Camera")

        robot_cfg = RobotSceneConfig(
            robot_id="custom_franka",
            robot_asset_path="omniverse://localhost/Isaac/Robots/Franka/franka.usd",
        )
        self.assertEqual(robot_cfg.robot_id, "custom_franka")
        self.assertIn("custom_franka", repr(robot_cfg))

        table = TablePrim(surface_z=0.20, height_z=0.04)
        self.assertAlmostEqual(table.center_z, 0.18, places=4)

    # 3. Missing Isaac runtime detection
    def test_03_missing_isaac_runtime_detection(self):
        """Verify that runtime methods raise IsaacSimRuntimeError when executed outside Isaac Sim."""
        runtime = SarthiIsaacRuntime()
        self.assertFalse(runtime.is_initialized)

        # initialize() outside Isaac Sim must raise IsaacSimRuntimeError
        with self.assertRaises(IsaacSimRuntimeError) as ctx:
            runtime.initialize()
        self.assertIn("Isaac Sim runtime is required", str(ctx.exception))

        # Other lifecycle operations without initialization must raise
        with self.assertRaises(IsaacSimRuntimeError):
            runtime.load_scenario()

        with self.assertRaises(IsaacSimRuntimeError):
            runtime.step()

        with self.assertRaises(IsaacSimRuntimeError):
            runtime.inject_disturbance()

        with self.assertRaises(IsaacSimRuntimeError):
            runtime.get_world_state()

        with self.assertRaises(IsaacSimRuntimeError):
            runtime.execute_action({"action_type": "APPROACH"})

        # SceneBuilder.create_world() without Isaac Sim must raise
        builder = SarthiSceneBuilder()
        with self.assertRaises(IsaacSimRuntimeError):
            builder.create_world()

    # 4. Scenario constants are sourced from TabletopPickPlaceScenario
    def test_04_scenario_constants_sourced_from_scenario(self):
        """Verify that scene builder sources all physical properties from locked scenario."""
        scenario = create_default_scenario()
        builder = SarthiSceneBuilder(scenario=scenario)

        # Red object
        self.assertEqual(builder.red_object_prim.object_id, scenario.red_object.object_id)
        self.assertEqual(builder.red_object_prim.position, scenario.red_object.initial_pose)
        self.assertEqual(builder.red_object_prim.mass_kg, scenario.red_object.mass_kg)
        self.assertEqual(builder.red_object_prim.dimensions, scenario.red_object.dimensions_m)

        # Blue target
        self.assertEqual(builder.blue_target_prim.target_id, scenario.blue_target.target_id)
        self.assertEqual(builder.blue_target_prim.position, scenario.blue_target.target_pose)
        self.assertEqual(builder.blue_target_prim.tolerance_radius_m, scenario.blue_target.tolerance_radius_m)

        # Robot
        self.assertEqual(builder.robot_prim.robot_id, scenario.robot.robot_id)
        self.assertEqual(builder.robot_prim.base_pose, scenario.robot.base_pose)

    # 5. Robot asset path is configurable
    def test_05_robot_asset_path_is_configurable(self):
        """Verify robot asset path is configurable and missing path raises clear error."""
        cfg = RobotSceneConfig(
            robot_id="tabletop_manipulator",
            robot_asset_path="/custom/assets/manipulator.usd",
        )
        prim = SarthiRobotPrim(config=cfg)
        self.assertEqual(prim.robot_asset_path, "/custom/assets/manipulator.usd")
        self.assertEqual(prim.robot_id, "tabletop_manipulator")

        # When asset path is missing, and Isaac Sim is simulated as available,
        # create() must raise error forbidding silent substitution
        with patch("simulation.isaac_runtime.robot_scene.is_isaac_sim_available", return_value=True):
            missing_asset_prim = SarthiRobotPrim(
                config=RobotSceneConfig(robot_asset_path=None)
            )
            with self.assertRaises(IsaacSimRuntimeError) as ctx:
                missing_asset_prim.create()
            self.assertIn("Missing robot asset path", str(ctx.exception))
            self.assertIn("Silent substitution", str(ctx.exception))

    # 6. Disturbance configuration is deterministic
    def test_06_disturbance_configuration_is_deterministic(self):
        """Verify dynamic obstacle prim parameters are deterministic and initially inactive."""
        scenario = create_default_scenario()
        dist1 = PathBlockedDisturbance(scenario=scenario)
        dist2 = PathBlockedDisturbance(scenario=scenario)

        self.assertFalse(dist1.is_active)
        self.assertEqual(dist1.obstacle_id, scenario.obstacle.obstacle_id)
        self.assertEqual(dist1.position, scenario.obstacle.position)
        self.assertEqual(dist1.dimensions, scenario.obstacle.dimensions_m)
        self.assertEqual(dist1.mass_kg, scenario.obstacle.mass_kg)
        self.assertEqual(dist1.usd_prim_path, f"/World/Obstacles/{scenario.obstacle.obstacle_id}")

        self.assertEqual(dist1.position, dist2.position)
        self.assertEqual(dist1.dimensions, dist2.dimensions)

    # 7. No duplicate WorldState model exists
    def test_07_no_duplicate_world_state_model_exists(self):
        """Verify runtime strictly reuses backend.app.decision_engine.models.WorldState."""
        from simulation.isaac_runtime import runtime as rt_module

        # Ensure WorldState imported in runtime module is identical to canonical WorldState
        self.assertIs(rt_module.WorldState, WorldState)

        # Inspect classes defined in runtime module to ensure no second WorldState is defined
        for name, cls in inspect.getmembers(rt_module, inspect.isclass):
            if name.endswith("WorldState") and cls.__module__ == rt_module.__name__:
                self.fail(f"Duplicate WorldState class '{name}' found in runtime module!")

    # 8. Runtime does not contain Decision Engine logic
    def test_08_runtime_does_not_contain_decision_engine_logic(self):
        """Verify SarthiIsaacRuntime contains zero decision engine or action selection logic."""
        runtime = SarthiIsaacRuntime()

        forbidden_cognitive_methods = [
            "decide",
            "evaluate_candidates",
            "calculate_responsibility",
            "calculate_relevance",
            "evaluate_consequences",
            "check_constraints",
            "select_action",
            "choose_action",
            "propose_action",
        ]
        for method in forbidden_cognitive_methods:
            self.assertFalse(
                hasattr(runtime, method),
                f"SarthiIsaacRuntime violates architectural boundary: contains '{method}'",
            )


if __name__ == "__main__":
    unittest.main()
