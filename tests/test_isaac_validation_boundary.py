"""
SĀRTHI Phase 6D — Isaac Sim Real Runtime Validation Boundary Tests.

IMPORTANT:
- ZERO dependencies on NVIDIA Isaac Sim packages.
- Tests do NOT launch Isaac Sim.
- Verifies script importability, configuration parsing, runtime guard,
  absence of duplicated constants, schema validation, and closed-loop validation flow.
"""

import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from backend.app.decision_engine.engine import SarthiDecisionEngine
from backend.app.decision_engine.models import ActionType, Point3D
from scripts.isaac_sim.run_sarthi_validation import (
    SarthiIsaacValidator,
    ValidationTraceEntry,
    load_validation_config,
    log_phase,
    main as validation_main,
)
from simulation.adapters.base import LocalSimulationAdapter
from simulation.core.geometry import SimDimensions3D, SimPoint3D, WorkspaceBounds
from simulation.core.objects import SimulatedObject, SimulatedTargetZone
from simulation.core.robot import SimulatedRobot
from simulation.core.world import SimulationWorld
from simulation.scenarios.tabletop_pick_place import create_default_scenario


def _make_validation_test_world():
    """Builds standard world matching tabletop_pick_and_place_mvp scenario."""
    scen = create_default_scenario()
    bounds = WorkspaceBounds(
        min_x=scen.workspace.min_x,
        max_x=scen.workspace.max_x,
        min_y=scen.workspace.min_y,
        max_y=scen.workspace.max_y,
        min_z=scen.workspace.min_z,
        max_z=scen.workspace.max_z,
    )
    robot = SimulatedRobot(
        robot_id=scen.robot.robot_id,
        position=SimPoint3D(
            x=scen.robot.base_pose.x,
            y=scen.robot.base_pose.y,
            z=scen.robot.base_pose.z,
        ),
        max_reach=scen.robot.max_reach_m,
        max_payload_kg=scen.robot.max_payload_kg,
        workspace_limits=bounds,
    )
    world = SimulationWorld(robot=robot, workspace_bounds=bounds)

    red_obj = SimulatedObject(
        id=scen.red_object.object_id,
        name=scen.red_object.name,
        position=SimPoint3D(
            x=scen.red_object.initial_pose.x,
            y=scen.red_object.initial_pose.y,
            z=scen.red_object.initial_pose.z,
        ),
        dimensions=SimDimensions3D(
            length_x=scen.red_object.dimensions_m.x,
            width_y=scen.red_object.dimensions_m.y,
            height_z=scen.red_object.dimensions_m.z,
        ),
        mass=scen.red_object.mass_kg,
        is_target=True,
    )
    blue_target = SimulatedTargetZone(
        id=scen.blue_target.target_id,
        name=scen.blue_target.name,
        position=SimPoint3D(
            x=scen.blue_target.target_pose.x,
            y=scen.blue_target.target_pose.y,
            z=scen.blue_target.target_pose.z,
        ),
        tolerance_radius=scen.blue_target.tolerance_radius_m,
    )
    world.add_object(red_obj)
    world.add_target_zone(blue_target)
    adapter = LocalSimulationAdapter(world)
    return world, adapter


class TestIsaacValidationBoundary(unittest.TestCase):
    """
    Unit test suite covering Phase 6D Validation Boundary.
    """

    # 1. Validation script imports correctly without Isaac Sim
    def test_01_validation_script_imports_without_isaac_sim(self):
        """Verify run_sarthi_validation can be imported without requiring Isaac Sim packages."""
        self.assertNotIn("isaacsim", sys.modules)
        self.assertTrue(callable(load_validation_config))
        self.assertTrue(callable(validation_main))

    # 2. Configuration loads correctly
    def test_02_configuration_loads(self):
        """Verify configs/isaac_sim_validation.yaml parses expected parameters."""
        cfg = load_validation_config()
        self.assertEqual(cfg["scenario_id"], "tabletop_pick_and_place_mvp")
        self.assertTrue(cfg["headless"])
        self.assertEqual(cfg["max_steps"], 10)
        self.assertTrue(cfg["enable_trace"])
        self.assertEqual(cfg["output_directory"], "logs/isaac_validation")

    # 3. Scenario ID is valid and matches locked benchmark
    def test_03_scenario_id_is_valid(self):
        """Verify scenario_id matches TabletopPickPlaceScenario."""
        cfg = load_validation_config()
        scenario = create_default_scenario()
        self.assertEqual(cfg["scenario_id"], scenario.scenario_id)

    # 4. Runtime guard still works and exits non-zero outside Isaac Sim
    def test_04_runtime_guard_fails_outside_isaac_sim(self):
        """Verify validation script fails fast with exit code 1 outside Isaac Sim."""
        stderr_capture = io.StringIO()
        with patch("sys.argv", ["run_sarthi_validation.py"]):
            with patch("sys.stderr", stderr_capture):
                with self.assertRaises(SystemExit) as ctx:
                    validation_main()
                self.assertEqual(ctx.exception.code, 1)

        err_output = stderr_capture.getvalue()
        self.assertIn("Isaac Sim runtime is required", err_output)

    # 5. No Decision Engine logic inside Isaac runtime
    def test_05_no_decision_engine_logic_inside_isaac_runtime(self):
        """Verify Decision Engine is external to runtime and validator."""
        from simulation.isaac_runtime.runtime import SarthiIsaacRuntime

        runtime = SarthiIsaacRuntime()
        forbidden_methods = ["decide", "evaluate_candidates", "score_candidate", "check_constraints"]
        for m in forbidden_methods:
            self.assertFalse(hasattr(runtime, m))

        # Validator delegates decision making to SarthiDecisionEngine instance
        validator = SarthiIsaacValidator(
            runtime=None,
            adapter=None,
        )
        self.assertIsInstance(validator.engine, SarthiDecisionEngine)

    # 6. No physical constants duplicated in configuration
    def test_06_no_physical_constants_duplicated_in_config(self):
        """Verify YAML config contains zero physical constants (coordinates, masses, dims)."""
        cfg = load_validation_config()
        forbidden_keys = [
            "position", "dimensions", "mass", "mass_kg",
            "reach", "payload", "bounds", "coordinates",
            "x", "y", "z", "tolerance", "obstacle_pos",
        ]
        for key in forbidden_keys:
            self.assertNotIn(
                key,
                cfg,
                f"Physical constant '{key}' leaked into configuration file!",
            )

    # 7. Validation output schema is deterministic
    def test_07_validation_output_schema_deterministic(self):
        """Verify ValidationTraceEntry validates and serializes to JSON deterministically."""
        trace = ValidationTraceEntry(
            step_number=1,
            world_state_version_before=1,
            selected_action={"action_type": "APPROACH", "action_id": "act_1"},
            isaac_action={"action_type": "APPROACH", "primitive_name": "cartesian_approach"},
            execution_result={"success": True, "action_type": "APPROACH", "simulation_time": 0.05},
            world_state_version_after=2,
            verification_result=True,
            disturbance_state={"is_active": False, "type": None},
            final_status="STEP_OK",
        )
        dump = trace.model_dump()
        self.assertEqual(dump["step_number"], 1)
        self.assertEqual(dump["final_status"], "STEP_OK")
        self.assertEqual(dump["selected_action"]["action_type"], "APPROACH")
        self.assertEqual(dump["isaac_action"]["primitive_name"], "cartesian_approach")

    # 8. Validator executes closed-loop recovery flow with local adapter
    def test_08_validator_closed_loop_flow(self):
        """Verify closed-loop sequence: APPROACH -> GRASP -> DISTURBANCE -> REPOSITION -> MOVE -> RELEASE."""
        world, adapter = _make_validation_test_world()
        scenario = create_default_scenario()

        validator = SarthiIsaacValidator(
            runtime=None,
            adapter=adapter,
            scenario=scenario,
            max_steps=10,
            enable_trace=True,
        )

        stdout_capture = io.StringIO()
        with patch("sys.stdout", stdout_capture):
            success, msg = validator.run_validation()

        self.assertTrue(success, f"Validation failed: {msg}")
        self.assertEqual(len(validator.trace), 5)  # APPROACH, GRASP, REPOSITION, MOVE, RELEASE

        # Verify phase logging
        log_output = stdout_capture.getvalue()
        self.assertIn("[ISAAC] phase=TASK_LOADED", log_output)
        self.assertIn("[ISAAC] phase=APPROACH", log_output)
        self.assertIn("[ISAAC] phase=GRASP", log_output)
        self.assertIn("[ISAAC] phase=DISTURBANCE_INJECTED", log_output)
        self.assertIn("[ISAAC] phase=RECOVERY_DECISION", log_output)
        self.assertIn("[ISAAC] phase=REPOSITION", log_output)
        self.assertIn("[ISAAC] phase=MOVE", log_output)
        self.assertIn("[ISAAC] phase=RELEASE", log_output)
        self.assertIn("[ISAAC] phase=FINAL_VERIFICATION", log_output)
        self.assertIn("[ISAAC] phase=SUCCESS", log_output)


if __name__ == "__main__":
    unittest.main()
