"""
SĀRTHI Phase V3-8: Physics Validation Framework Unit Tests.

Covers the 20 required validation dimensions and architectural invariants:
1. validation result schema
2. timestep validation
3. joint-limit validation
4. joint-velocity validation
5. end-effector repeatability
6. contact stability
7. placement repeatability
8. collision/clearance
9. gravity/settling
10. IK convergence
11. numerical stability
12. deterministic replay
13. physics regression
14. failure reporting
15. tolerance handling
16. no AI provider access
17. no secret leakage
18. no robot execution outside the existing runtime
19. deterministic validation result structure
20. existing V3 safety authority remains unchanged
"""

import math
import unittest

from backend.app.decision_engine.engine import SarthiDecisionEngine
from backend.app.decision_engine.models import (
    ActionType,
    CandidateAction,
    Point3D,
    WorldState,
)
from backend.app.validation.framework import SarthiPhysicsValidator
from backend.app.validation.models import (
    PhysicsValidationResult,
    PhysicsValidationSuiteReport,
    ValidationDimension,
    sanitize_secrets,
)
from simulation.mujoco_runtime import require_mujoco
from simulation.mujoco_runtime.runtime import SarthiMuJoCoRuntime


class TestV38PhysicsValidation(unittest.TestCase):
    """
    V3-8 Physics Validation Suite Tests.
    """

    @classmethod
    def setUpClass(cls):
        require_mujoco()
        cls.validator = SarthiPhysicsValidator()

    # -----------------------------------------------------------------------
    # Test 1: Validation result schema
    # -----------------------------------------------------------------------
    def test_01_validation_result_schema(self):
        """1. Validates PhysicsValidationResult and SuiteReport schema conformance."""
        res = PhysicsValidationResult(
            validation_id="val_test_schema_01",
            dimension=ValidationDimension.TIMESTEP_SENSITIVITY.value,
            scenario="tabletop_pick_and_place_mvp",
            parameter_configuration={"dt": 0.002},
            run_count=1,
            pass_fail=True,
            measured_values={"placement_error_m": 0.015},
            expected_bounds={"max_error_m": 0.06},
            failure_reasons=[],
            reproducibility_information={"engine": "mujoco"},
            metadata={"test": "schema"},
        )
        self.assertEqual(res.validation_id, "val_test_schema_01")
        self.assertTrue(res.passed)
        self.assertEqual(res.dimension, "timestep_sensitivity")
        d = res.to_dict()
        self.assertIn("validation_id", d)
        self.assertIn("pass_fail", d)
        self.assertIn("measured_values", d)

    # -----------------------------------------------------------------------
    # Test 2: Timestep validation
    # -----------------------------------------------------------------------
    def test_02_timestep_validation(self):
        """2. Validates timestep sensitivity under controlled timesteps [0.0019, 0.002, 0.0025]."""
        res = self.validator.validate_timestep_sensitivity([0.0019, 0.002, 0.0025])
        self.assertTrue(res.pass_fail)
        self.assertEqual(res.dimension, ValidationDimension.TIMESTEP_SENSITIVITY.value)
        self.assertEqual(res.run_count, 3)
        self.assertIn("dt_0.0020s", res.measured_values)
        self.assertTrue(res.measured_values["dt_0.0020s"]["within_tolerance"])

    # -----------------------------------------------------------------------
    # Test 3: Joint-limit validation
    # -----------------------------------------------------------------------
    def test_03_joint_limit_validation(self):
        """3. Verifies that joint limits are strictly maintained and clamping is active."""
        res = self.validator.validate_joint_limits()
        self.assertTrue(res.pass_fail)
        self.assertEqual(res.measured_values["nominal_violations"], 0)
        self.assertTrue(res.measured_values["clamping_protection_active"])
        self.assertTrue(res.measured_values["boundary_target_within_limits"])

    # -----------------------------------------------------------------------
    # Test 4: Joint-velocity validation
    # -----------------------------------------------------------------------
    def test_04_joint_velocity_validation(self):
        """4. Verifies joint velocities remain within safe operational bounds."""
        res = self.validator.validate_joint_velocities(max_allowed_vel=2.5)
        self.assertTrue(res.pass_fail)
        self.assertFalse(res.measured_values["violation_detected"])
        self.assertLessEqual(res.measured_values["max_observed_velocity_rad_s"], 2.5)

    # -----------------------------------------------------------------------
    # Test 5: End-effector repeatability
    # -----------------------------------------------------------------------
    def test_05_end_effector_repeatability(self):
        """5. Measures repeatability error across multiple approach runs."""
        res = self.validator.validate_end_effector_repeatability(runs=3)
        self.assertTrue(res.pass_fail)
        self.assertEqual(res.run_count, 3)
        self.assertLessEqual(res.measured_values["max_deviation_m"], 0.002)

    # -----------------------------------------------------------------------
    # Test 6: Contact stability
    # -----------------------------------------------------------------------
    def test_06_contact_stability(self):
        """6. Validates grasp retention during transport and clean separation on release."""
        res = self.validator.validate_contact_stability()
        self.assertTrue(res.pass_fail)
        self.assertTrue(res.measured_values["post_grasp_grasped"])
        self.assertTrue(res.measured_values["transport_reposition_grasped"])
        self.assertTrue(res.measured_values["transport_move_grasped"])
        self.assertFalse(res.measured_values["post_release_grasped"])
        self.assertTrue(res.measured_values["post_release_gripper_open"])

    # -----------------------------------------------------------------------
    # Test 7: Placement repeatability
    # -----------------------------------------------------------------------
    def test_07_placement_repeatability(self):
        """7. Evaluates placement repeatability across multiple full runs."""
        res = self.validator.validate_placement_repeatability(runs=3)
        self.assertTrue(res.pass_fail)
        self.assertEqual(res.measured_values["success_rate"], 1.0)
        self.assertLessEqual(res.measured_values["max_placement_error_m"], 0.06)

    # -----------------------------------------------------------------------
    # Test 8: Collision/clearance
    # -----------------------------------------------------------------------
    def test_08_collision_clearance(self):
        """8. Validates direct blocked trajectory rejection and elevated collision-free recovery."""
        res = self.validator.validate_collision_clearance()
        self.assertTrue(res.pass_fail)
        self.assertTrue(res.measured_values["direct_trajectory_rejected"])
        self.assertFalse(res.measured_values["obstacle_collision_detected"])
        self.assertGreater(res.measured_values["vertical_clearance_m"], 0.05)

    # -----------------------------------------------------------------------
    # Test 9: Gravity/settling
    # -----------------------------------------------------------------------
    def test_09_gravity_settling(self):
        """9. Verifies object settling dynamics under gravity without teleportation."""
        res = self.validator.validate_gravity_settling(settle_ticks=100)
        self.assertTrue(res.pass_fail)
        self.assertLessEqual(res.measured_values["residual_linear_velocity_m_s"], 0.05)
        self.assertLessEqual(res.measured_values["settling_displacement_m"], 0.05)

    # -----------------------------------------------------------------------
    # Test 10: IK convergence
    # -----------------------------------------------------------------------
    def test_10_ik_convergence(self):
        """10. Tests DLS-IK convergence on reachable waypoints and safe handling of unreachable targets."""
        res = self.validator.validate_ik_convergence()
        self.assertTrue(res.pass_fail)
        self.assertEqual(res.measured_values["reachable_converged"], res.measured_values["reachable_count"])
        unreachable_res = res.measured_values["waypoint_results"]["unreachable_extreme"]
        self.assertFalse(unreachable_res["converged"])
        self.assertTrue(unreachable_res["finite_values"])

    # -----------------------------------------------------------------------
    # Test 11: Numerical stability
    # -----------------------------------------------------------------------
    def test_11_numerical_stability(self):
        """11. Continuously monitors physics state for NaN, Inf, and explosions."""
        res = self.validator.validate_numerical_stability()
        self.assertTrue(res.pass_fail)
        self.assertEqual(res.measured_values["instability_events"], 0)
        self.assertEqual(res.measured_values["nan_events"], 0)
        self.assertEqual(res.measured_values["explosion_events"], 0)

    # -----------------------------------------------------------------------
    # Test 12: Deterministic replay
    # -----------------------------------------------------------------------
    def test_12_deterministic_replay(self):
        """12. Confirms exact deterministic reproducibility across identical simulation runs."""
        res = self.validator.validate_deterministic_replay()
        self.assertTrue(res.pass_fail)
        self.assertTrue(res.measured_values["action_sequence_matched"])
        self.assertLessEqual(res.measured_values["euclidean_distance_difference_m"], 1e-4)

    # -----------------------------------------------------------------------
    # Test 13: Physics regression
    # -----------------------------------------------------------------------
    def test_13_physics_regression(self):
        """13. Verifies that all 8 established stages of the canonical scenario succeed without regression."""
        res = self.validator.validate_physics_regression()
        self.assertTrue(res.pass_fail)
        self.assertTrue(res.measured_values["all_milestones_passed"])
        self.assertEqual(len(res.measured_values["milestones"]), 8)

    # -----------------------------------------------------------------------
    # Test 14: Failure reporting
    # -----------------------------------------------------------------------
    def test_14_failure_reporting(self):
        """14. Tests that an intentionally unachievable velocity bound reports a structured failure."""
        unreasonable_vel = 0.001  # 1 mm/s will fail during normal arm motion
        res = self.validator.validate_joint_velocities(max_allowed_vel=unreasonable_vel)
        self.assertFalse(res.pass_fail)
        self.assertGreater(len(res.failure_reasons), 0)
        self.assertTrue(any("exceeded limit" in r for r in res.failure_reasons))

    # -----------------------------------------------------------------------
    # Test 15: Tolerance handling
    # -----------------------------------------------------------------------
    def test_15_tolerance_handling(self):
        """15. Confirms configurable parameters and bounds are preserved in validation results."""
        res = self.validator.validate_end_effector_repeatability(runs=2)
        self.assertIn("max_repeatability_error_m", res.expected_bounds)
        self.assertEqual(res.expected_bounds["max_repeatability_error_m"], 0.002)

    # -----------------------------------------------------------------------
    # Test 16: No AI provider access
    # -----------------------------------------------------------------------
    def test_16_no_ai_provider_access(self):
        """16. Proves validation engine does not import or invoke external AI models."""
        import sys
        validator_module = sys.modules[self.validator.__module__]
        forbidden_providers = ["nebius", "openai", "anthropic", "nemotron", "laya", "typesafe"]
        for name in dir(validator_module):
            for f in forbidden_providers:
                self.assertNotIn(f, name.lower())

    # -----------------------------------------------------------------------
    # Test 17: No secret leakage
    # -----------------------------------------------------------------------
    def test_17_no_secret_leakage(self):
        """17. Verifies secret sanitization scrubs bearer tokens and sensitive keys."""
        dirty = {
            "api_key": "sk-secret123456",
            "Authorization": "Bearer secret_token_abc",
            "normal_data": 42,
            "nested": {"token_value": "secret999"},
        }
        cleaned = sanitize_secrets(dirty)
        self.assertEqual(cleaned["api_key"], "[REDACTED]")
        self.assertEqual(cleaned["Authorization"], "[REDACTED]")
        self.assertEqual(cleaned["normal_data"], 42)
        self.assertEqual(cleaned["nested"]["token_value"], "[REDACTED]")

    # -----------------------------------------------------------------------
    # Test 18: No robot execution outside existing runtime
    # -----------------------------------------------------------------------
    def test_18_no_robot_execution_outside_existing_runtime(self):
        """18. Verifies validator constructs physical executions strictly through SarthiMuJoCoRuntime."""
        runtime = self.validator._create_runtime()
        self.assertIsInstance(runtime, SarthiMuJoCoRuntime)
        self.assertTrue(hasattr(runtime, "execute_action"))
        self.assertTrue(hasattr(runtime, "inject_disturbance"))

    # -----------------------------------------------------------------------
    # Test 19: Deterministic validation result structure
    # -----------------------------------------------------------------------
    def test_19_deterministic_validation_result_structure(self):
        """19. Tests that suite report aggregates all 12 dimensions deterministically."""
        report = self.validator.run_all_validations()
        self.assertIsInstance(report, PhysicsValidationSuiteReport)
        self.assertEqual(report.total_validations, 12)
        self.assertEqual(report.passed_validations, 12)
        self.assertEqual(report.failed_validations, 0)
        self.assertTrue(report.all_passed)
        for dim in ValidationDimension:
            self.assertIn(dim.value, report.results)

    # -----------------------------------------------------------------------
    # Test 20: Existing V3 safety authority remains unchanged
    # -----------------------------------------------------------------------
    def test_20_existing_v3_safety_authority_remains_unchanged(self):
        """20. Confirms SarthiDecisionEngine retains sole deterministic authority over actions."""
        engine = SarthiDecisionEngine()
        runtime = self.validator._create_runtime()
        ws = runtime.get_world_state()
        decision = engine.decide(ws)
        self.assertIsNotNone(decision.selected_action)
        self.assertEqual(decision.selected_action.action_type, ActionType.APPROACH)
        # Attempt an invalid action exceeding reach limit
        invalid_candidate = CandidateAction(
            action_type=ActionType.APPROACH,
            action_id="act_invalid_reach",
            target_position=Point3D(x=5.0, y=5.0, z=0.20),
        )
        decision_invalid = engine.decide(ws, candidate_actions=[invalid_candidate])
        self.assertEqual(len(decision_invalid.candidate_evaluations), 1)
        eval_res = decision_invalid.candidate_evaluations[0]
        self.assertFalse(eval_res.is_valid)
        self.assertTrue(any("reach" in r.lower() or "workspace" in r.lower() for r in eval_res.rejection_reasons))


if __name__ == "__main__":
    unittest.main()
