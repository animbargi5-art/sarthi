"""
SĀRTHI V3-3 — Decision Context Builder Unit Test Suite.
Verifies deterministic synthesis of compact, bounded DecisionContext payloads,
strict context minimization, isolation from simulation internals, credential sanitization,
invariance to irrelevant field modifications, and sensitivity to decision-critical state changes.

Non-Negotiable Architecture Invariants:
1. AI models propose or select bounded semantic candidates; deterministic software validates
   physical feasibility; only validated actions reach the robot controller.
2. Jev must never bypass deterministic safety validation.
3. DecisionContext contains only minimal decision-critical fields; never exposes raw simulator
   internals, meshes, or credentials.
"""

import json
import unittest

from backend.app.decision_engine.models import (
    ActionType,
    ActiveConstraint,
    CandidateAction,
    EnvironmentState,
    LastActionOutcome,
    LastActionStatus,
    ObjectState,
    Point3D,
    RobotState,
    TargetZone,
    TaskObjective,
    WorldObject,
    WorldState,
)
from backend.app.model.models import TaskUnderstanding
from backend.app.model.v3_models import (
    DecisionContext,
    HumanInstruction,
    PhysicalSituation,
)
from backend.app.orchestration.context_builder import DecisionContextBuilder


class TestDecisionContextBuilder(unittest.TestCase):
    """Test suite for deterministic DecisionContextBuilder (Phase V3-3)."""

    def setUp(self):
        """Set up canonical state fixtures for context builder testing."""
        self.builder = DecisionContextBuilder(coordinate_precision=3)

        self.robot = RobotState(
            position=Point3D(x=0.55012, y=-0.00045, z=0.52099),
            gripper_open=True,
            holding_object_id=None,
            payload_mass_kg=0.0,
            is_moving=False,
            max_payload_kg=3.0,
            max_reach_m=0.85,
        )
        self.target_zone = TargetZone(
            id="zone_drop",
            position=Point3D(x=0.4001, y=-0.1902, z=0.1203),
            tolerance_radius_m=0.05,
        )
        self.object_red = WorldObject(
            id="obj_red_01",
            name="red_cube",
            position=Point3D(x=0.4501, y=0.1004, z=0.1206),
            bounding_radius_m=0.025,
            mass_kg=0.12,
            state=ObjectState.FREE,
            is_target=True,
            is_obstacle=False,
        )
        self.environment = EnvironmentState(
            friction_coefficient=0.85,
            slip_risk_level=0.1,
            dynamic_obstacles_detected=False,
        )
        self.constraint_clearance = ActiveConstraint(
            constraint_id="c_clearance",
            description="Maintain clearance >= 0.08m",
            required_clearance_m=0.08,
        )
        self.constraint_speed = ActiveConstraint(
            constraint_id="c_speed",
            description="Max velocity 0.25 m/s",
            max_speed_mps=0.25,
        )
        self.last_outcome_success = LastActionOutcome(
            action_type=ActionType.APPROACH,
            status=LastActionStatus.SUCCESS,
        )
        self.world_state = WorldState(
            version=1,
            timestamp_ns=1_000_000,
            robot=self.robot,
            objects=[self.object_red],
            target=self.target_zone,
            environment=self.environment,
            task_objective=TaskObjective.PICK_AND_PLACE,
            active_constraints=[self.constraint_speed, self.constraint_clearance],
            last_action_outcome=self.last_outcome_success,
        )
        self.situation = PhysicalSituation(
            situation_id="sit_nominal_01",
            world_state=self.world_state,
            active_constraints=[self.constraint_speed, self.constraint_clearance],
            last_action_outcome=self.last_outcome_success,
            timestamp_ns=1_000_000,
        )

    # 1. Nominal context construction
    def test_nominal_context_construction_from_situation(self):
        """Verify nominal DecisionContext construction from PhysicalSituation."""
        candidates = ["APPROACH", "GRASP", "MOVE", "RELEASE"]
        ctx = self.builder.build(
            context_id="ctx_001",
            situation=self.situation,
            candidate_actions=candidates,
        )

        self.assertIsInstance(ctx, DecisionContext)
        self.assertEqual(ctx.context_id, "ctx_001")
        self.assertEqual(ctx.task_objective, "PICK_AND_PLACE")
        self.assertEqual(ctx.timestamp_ns, 1_000_000)

        # Robot state verification (rounded to 3 decimal places)
        self.assertEqual(ctx.robot_state_summary["position"], [0.55, -0.0, 0.521])
        self.assertTrue(ctx.robot_state_summary["gripper_open"])
        self.assertIsNone(ctx.robot_state_summary["holding_object_id"])
        self.assertEqual(ctx.robot_state_summary["payload_mass_kg"], 0.0)

        # Target object verification
        self.assertIsNotNone(ctx.target_object_summary)
        self.assertEqual(ctx.target_object_summary["id"], "obj_red_01")
        self.assertEqual(ctx.target_object_summary["name"], "red_cube")
        self.assertEqual(ctx.target_object_summary["position"], [0.45, 0.1, 0.121])
        self.assertEqual(ctx.target_object_summary["state"], "FREE")

        # Destination verification
        self.assertIsNotNone(ctx.destination_summary)
        self.assertEqual(ctx.destination_summary["id"], "zone_drop")
        self.assertEqual(ctx.destination_summary["position"], [0.4, -0.19, 0.12])
        self.assertEqual(ctx.destination_summary["tolerance_radius_m"], 0.05)

        # Candidate actions
        self.assertEqual(ctx.candidate_actions, ["APPROACH", "GRASP", "MOVE", "RELEASE"])

        # Previous action outcome
        self.assertEqual(ctx.previous_action_outcome, "APPROACH:SUCCESS")

    def test_nominal_context_construction_from_world_state(self):
        """Verify direct construction from WorldState without PhysicalSituation wrapper."""
        ctx = self.builder.build(
            context_id="ctx_direct_ws",
            world_state=self.world_state,
            candidate_actions=[ActionType.APPROACH, ActionType.STOP],
        )
        self.assertEqual(ctx.context_id, "ctx_direct_ws")
        self.assertEqual(ctx.task_objective, "PICK_AND_PLACE")
        self.assertEqual(ctx.candidate_actions, ["APPROACH", "STOP"])
        self.assertEqual(ctx.previous_action_outcome, "APPROACH:SUCCESS")

    # 2. Obstacle / PATH_BLOCKED context
    def test_obstacle_path_blocked_context(self):
        """Verify context construction for obstacle disturbance and recovery."""
        keepout_constraint = ActiveConstraint(
            constraint_id="c_keepout_obs",
            description="Exclusion zone: obstacle_cylinder radius 0.07m",
            keep_out_center=Point3D(x=0.50, y=0.05, z=0.15),
            keep_out_radius=0.07,
        )
        failed_outcome = LastActionOutcome(
            action_type=ActionType.MOVE,
            status=LastActionStatus.FAILURE,
            error_message="Proximity threshold violation: obstacle_cylinder",
        )
        disturbance = {
            "type": "PATH_BLOCKED",
            "detected_obstacle": "obstacle_cylinder",
            "clearance_margin_m": 0.012,
        }

        ctx = self.builder.build(
            context_id="ctx_blocked_01",
            world_state=self.world_state,
            active_constraints=[keepout_constraint],
            last_action_outcome=failed_outcome,
            candidate_actions=["REPOSITION", "APPROACH", "STOP"],
            disturbance_info=disturbance,
        )

        self.assertIn("Exclusion zone: obstacle_cylinder radius 0.07m", ctx.active_constraints)
        self.assertEqual(
            ctx.previous_action_outcome,
            "MOVE:FAILURE (Proximity threshold violation: obstacle_cylinder)",
        )
        self.assertIsNotNone(ctx.disturbance_info)
        self.assertEqual(ctx.disturbance_info["type"], "PATH_BLOCKED")
        self.assertEqual(ctx.disturbance_info["detected_obstacle"], "obstacle_cylinder")
        self.assertEqual(ctx.candidate_actions, ["REPOSITION", "APPROACH", "STOP"])

    # 3. Previous-action context variants
    def test_previous_action_context_no_previous_action(self):
        """Verify context construction when no previous action has been executed."""
        none_outcome = LastActionOutcome(status=LastActionStatus.NONE)
        ctx = self.builder.build(
            context_id="ctx_initial",
            world_state=self.world_state,
            last_action_outcome=none_outcome,
        )
        self.assertIsNone(ctx.previous_action_outcome)

    def test_previous_action_context_with_failure_message(self):
        """Verify error message formatting in previous action outcome."""
        failed_outcome = LastActionOutcome(
            action_type=ActionType.GRASP,
            status=LastActionStatus.FAILURE,
            error_message="Slip detected during grasp closure",
        )
        ctx = self.builder.build(
            context_id="ctx_slip_fail",
            world_state=self.world_state,
            last_action_outcome=failed_outcome,
        )
        self.assertEqual(
            ctx.previous_action_outcome,
            "GRASP:FAILURE (Slip detected during grasp closure)",
        )

    # 4. Missing optional information
    def test_missing_optional_information(self):
        """Verify context builder handles empty objects, no candidates, and missing disturbance."""
        empty_ws = WorldState(
            version=2,
            timestamp_ns=2_000_000,
            robot=self.robot,
            objects=[],  # No objects
            target=self.target_zone,
            active_constraints=[],
            last_action_outcome=LastActionOutcome(status=LastActionStatus.NONE),
        )
        ctx = self.builder.build(
            context_id="ctx_sparse",
            world_state=empty_ws,
        )

        self.assertIsNone(ctx.target_object_summary)
        self.assertEqual(ctx.active_constraints, [])
        self.assertIsNone(ctx.previous_action_outcome)
        self.assertEqual(ctx.candidate_actions, [])
        self.assertIsNone(ctx.disturbance_info)
        self.assertEqual(ctx.task_objective, "PICK_AND_PLACE")

    # 5. Deterministic serialization
    def test_deterministic_serialization(self):
        """Verify identical inputs produce identical byte-for-byte serialized JSON."""
        ctx_a = self.builder.build(
            context_id="ctx_determinism",
            situation=self.situation,
            candidate_actions=["APPROACH", "GRASP", "STOP"],
            disturbance_info={"severity": "mild", "anomaly": "minor_drift"},
            timestamp_ns=42_000,
        )
        ctx_b = self.builder.build(
            context_id="ctx_determinism",
            situation=self.situation,
            candidate_actions=["APPROACH", "GRASP", "STOP"],
            disturbance_info={"severity": "mild", "anomaly": "minor_drift"},
            timestamp_ns=42_000,
        )

        self.assertEqual(ctx_a, ctx_b)
        json_a = ctx_a.model_dump_json()
        json_b = ctx_b.model_dump_json()
        self.assertEqual(json_a, json_b)

    # 6. Irrelevant simulation data is NOT exposed
    def test_irrelevant_simulation_data_is_not_exposed(self):
        """Verify simulator internal arrays, meshes, and dynamics are strictly excluded."""
        ctx = self.builder.build(
            context_id="ctx_minimal",
            situation=self.situation,
        )
        raw_json = ctx.model_dump_json()
        parsed = json.loads(raw_json)

        # Robot state must only have compact decision fields
        robot_summary = parsed["robot_state_summary"]
        self.assertEqual(set(robot_summary.keys()), {"position", "gripper_open", "holding_object_id", "payload_mass_kg"})
        self.assertNotIn("joint_positions", robot_summary)
        self.assertNotIn("joint_velocities", robot_summary)
        self.assertNotIn("actuator_forces", robot_summary)
        self.assertNotIn("max_reach_m", robot_summary)
        self.assertNotIn("max_payload_kg", robot_summary)
        self.assertNotIn("is_moving", robot_summary)
        self.assertNotIn("qpos", raw_json)
        self.assertNotIn("qvel", raw_json)
        self.assertNotIn("mujoco", raw_json.lower())
        self.assertNotIn("mesh", raw_json.lower())
        self.assertNotIn("renderer", raw_json.lower())

        # Target object must only have id, name, position, state
        target_summary = parsed["target_object_summary"]
        self.assertEqual(set(target_summary.keys()), {"id", "name", "position", "state"})
        self.assertNotIn("friction_coefficient", target_summary)
        self.assertNotIn("slip_risk_level", target_summary)
        self.assertNotIn("bounding_radius_m", target_summary)
        self.assertNotIn("mass_kg", target_summary)
        self.assertNotIn("is_obstacle", target_summary)

    # 7. Credentials and secrets cannot enter context
    def test_credentials_and_secrets_cannot_enter_context(self):
        """Verify that API keys, tokens, and authorization credentials are fully redacted."""
        secret_key = "sk-nebius-1234567890abcdef1234567890"
        secret_token = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"

        # Inject secrets into disturbance_info
        disturbance_with_secret = {
            "api_key": secret_key,
            "auth_token": secret_token,
            "diagnostic_note": f"Auth failed with token {secret_key}",
            "normal_field": "unaffected_value",
        }

        # Inject secrets into active constraint description
        tainted_constraint = ActiveConstraint(
            constraint_id="c_tainted",
            description=f"Security check using password {secret_key}",
        )

        # Inject secrets into previous action error message
        tainted_outcome = LastActionOutcome(
            action_type=ActionType.APPROACH,
            status=LastActionStatus.FAILURE,
            error_message=f"Connection refused by endpoint using key {secret_key}",
        )

        ctx = self.builder.build(
            context_id="ctx_secrets_test",
            world_state=self.world_state,
            active_constraints=[tainted_constraint],
            last_action_outcome=tainted_outcome,
            disturbance_info=disturbance_with_secret,
            task_objective=f"Pick cube with API_KEY={secret_key}",
        )

        raw_json = ctx.model_dump_json()

        # The actual secret string must NEVER be present anywhere in serialized JSON
        self.assertNotIn("sk-nebius", raw_json)
        self.assertNotIn("eyJhbGciOi", raw_json)

        # Verify redactions in specific structures
        self.assertEqual(ctx.disturbance_info["api_key"], "[REDACTED]")
        self.assertEqual(ctx.disturbance_info["auth_token"], "[REDACTED]")
        self.assertEqual(ctx.disturbance_info["normal_field"], "unaffected_value")
        self.assertIn("[REDACTED]", ctx.active_constraints[0])
        self.assertIn("[REDACTED]", ctx.previous_action_outcome)
        self.assertIn("[REDACTED]", ctx.task_objective)

    # 8. Bounded context size and shape
    def test_bounded_context_size_and_shape(self):
        """Verify the serialized context size is compact and strictly bounded (< 1500 bytes)."""
        ctx = self.builder.build(
            context_id="ctx_size_test",
            situation=self.situation,
            candidate_actions=["APPROACH", "GRASP", "MOVE", "RELEASE", "REPOSITION", "STOP"],
            disturbance_info={"anomaly": "minor_deviation", "magnitude": 0.005},
        )
        serialized_bytes = len(ctx.model_dump_json().encode("utf-8"))
        # Must be well below 1500 bytes (typically ~500-700 bytes)
        self.assertLess(serialized_bytes, 1500)
        self.assertGreater(serialized_bytes, 200)

    # 9. Correct mapping from canonical WorldState
    def test_correct_mapping_from_canonical_world_state(self):
        """Verify exact precision mapping from WorldState fields."""
        ctx = self.builder.build(
            context_id="ctx_map_test",
            world_state=self.world_state,
        )
        # Check position coordinate rounding
        self.assertAlmostEqual(ctx.robot_state_summary["position"][0], 0.550, places=3)
        self.assertAlmostEqual(ctx.robot_state_summary["position"][1], 0.000, places=3)
        self.assertAlmostEqual(ctx.robot_state_summary["position"][2], 0.521, places=3)

        # Check target object position
        self.assertAlmostEqual(ctx.target_object_summary["position"][0], 0.450, places=3)
        self.assertAlmostEqual(ctx.target_object_summary["position"][1], 0.100, places=3)
        self.assertAlmostEqual(ctx.target_object_summary["position"][2], 0.121, places=3)

        # Check target zone
        self.assertEqual(ctx.destination_summary["id"], "zone_drop")
        self.assertAlmostEqual(ctx.destination_summary["tolerance_radius_m"], 0.05, places=4)

    # 10. Correct mapping and deterministic sorting of active constraints
    def test_correct_mapping_and_sorting_of_active_constraints(self):
        """Verify active constraints are sorted alphabetically and deduplicated."""
        c1 = ActiveConstraint(constraint_id="c_z", description="Zebra boundary limit")
        c2 = ActiveConstraint(constraint_id="c_a", description="Alpha velocity limit")
        c3 = ActiveConstraint(constraint_id="c_m", description="Middle clearance threshold")
        c_dup = ActiveConstraint(constraint_id="c_a2", description="Alpha velocity limit")

        # Pass in scrambled order with duplicates
        ctx = self.builder.build(
            context_id="ctx_constraint_sort",
            world_state=self.world_state,
            active_constraints=[c1, c2, c3, c_dup],
        )

        expected = [
            "Alpha velocity limit",
            "Middle clearance threshold",
            "Zebra boundary limit",
        ]
        self.assertEqual(ctx.active_constraints, expected)

    # 11. Changing an IRRELEVANT field does NOT change the decision-critical context
    def test_irrelevant_field_change_does_not_change_context(self):
        """Modifying physics simulation internals (friction, bounding radius, mass) must not alter DecisionContext."""
        ctx_baseline = self.builder.build(
            context_id="ctx_const_test",
            world_state=self.world_state,
            candidate_actions=["APPROACH", "GRASP"],
            timestamp_ns=5000,
        )

        # Create modified WorldObject with changed bounding radius, mass, and is_obstacle
        modified_object = WorldObject(
            id="obj_red_01",
            name="red_cube",
            position=Point3D(x=0.4501, y=0.1004, z=0.1206),  # Unchanged
            bounding_radius_m=0.089,                         # CHANGED (irrelevant)
            mass_kg=0.999,                                   # CHANGED (irrelevant)
            state=ObjectState.FREE,                          # Unchanged
            is_target=True,                                  # Unchanged
            is_obstacle=False,
        )
        # Create modified Environment with different friction and slip risk
        modified_env = EnvironmentState(
            friction_coefficient=0.12,                       # CHANGED (irrelevant)
            slip_risk_level=0.95,                            # CHANGED (irrelevant)
            dynamic_obstacles_detected=False,
        )
        # Create modified Robot with changed reaches and moving flag
        modified_robot = RobotState(
            position=self.robot.position,                    # Unchanged
            gripper_open=self.robot.gripper_open,            # Unchanged
            holding_object_id=self.robot.holding_object_id,  # Unchanged
            payload_mass_kg=self.robot.payload_mass_kg,      # Unchanged
            is_moving=True,                                  # CHANGED (irrelevant)
            max_payload_kg=10.0,                             # CHANGED (irrelevant)
            max_reach_m=1.20,                                # CHANGED (irrelevant)
        )
        ws_modified = WorldState(
            version=self.world_state.version + 1,            # CHANGED (irrelevant)
            timestamp_ns=self.world_state.timestamp_ns,
            robot=modified_robot,
            objects=[modified_object],
            target=self.target_zone,
            environment=modified_env,
            task_objective=self.world_state.task_objective,
            active_constraints=self.world_state.active_constraints,
            last_action_outcome=self.world_state.last_action_outcome,
        )

        ctx_modified = self.builder.build(
            context_id="ctx_const_test",
            world_state=ws_modified,
            candidate_actions=["APPROACH", "GRASP"],
            timestamp_ns=5000,
        )

        self.assertEqual(ctx_baseline, ctx_modified)
        self.assertEqual(ctx_baseline.model_dump_json(), ctx_modified.model_dump_json())

    # 12. Changing a DECISION-CRITICAL field DOES change the context
    def test_decision_critical_robot_position_change_alters_context(self):
        """Altering robot end-effector position must change DecisionContext."""
        ctx_baseline = self.builder.build(
            context_id="ctx_crit_test",
            world_state=self.world_state,
            timestamp_ns=5000,
        )

        # Alter robot position significantly
        moved_robot = RobotState(
            position=Point3D(x=0.850, y=0.200, z=0.600),
            gripper_open=True,
            holding_object_id=None,
            payload_mass_kg=0.0,
        )
        ws_moved = WorldState(
            version=self.world_state.version + 1,
            timestamp_ns=self.world_state.timestamp_ns,
            robot=moved_robot,
            objects=self.world_state.objects,
            target=self.target_zone,
            task_objective=self.world_state.task_objective,
            active_constraints=self.world_state.active_constraints,
            last_action_outcome=self.world_state.last_action_outcome,
        )

        ctx_moved = self.builder.build(
            context_id="ctx_crit_test",
            world_state=ws_moved,
            timestamp_ns=5000,
        )

        self.assertNotEqual(ctx_baseline, ctx_moved)
        self.assertNotEqual(ctx_baseline.robot_state_summary["position"], ctx_moved.robot_state_summary["position"])

    def test_decision_critical_gripper_state_change_alters_context(self):
        """Altering gripper state from open to closed must change DecisionContext."""
        ctx_baseline = self.builder.build(
            context_id="ctx_crit_test",
            world_state=self.world_state,
            timestamp_ns=5000,
        )

        closed_robot = RobotState(
            position=self.robot.position,
            gripper_open=False,
            holding_object_id="obj_red_01",
            payload_mass_kg=0.12,
        )
        ws_closed = WorldState(
            version=self.world_state.version + 1,
            timestamp_ns=self.world_state.timestamp_ns,
            robot=closed_robot,
            objects=self.world_state.objects,
            target=self.target_zone,
            task_objective=self.world_state.task_objective,
            active_constraints=self.world_state.active_constraints,
            last_action_outcome=self.world_state.last_action_outcome,
        )

        ctx_closed = self.builder.build(
            context_id="ctx_crit_test",
            world_state=ws_closed,
            timestamp_ns=5000,
        )

        self.assertNotEqual(ctx_baseline, ctx_closed)
        self.assertFalse(ctx_closed.robot_state_summary["gripper_open"])
        self.assertEqual(ctx_closed.robot_state_summary["holding_object_id"], "obj_red_01")

    def test_decision_critical_constraint_change_alters_context(self):
        """Adding a new active constraint must alter DecisionContext."""
        ctx_baseline = self.builder.build(
            context_id="ctx_crit_test",
            world_state=self.world_state,
            timestamp_ns=5000,
        )

        new_constraint = ActiveConstraint(
            constraint_id="c_force_limit",
            description="Force threshold 10N",
            max_force_newtons=10.0,
        )

        ctx_constrained = self.builder.build(
            context_id="ctx_crit_test",
            world_state=self.world_state,
            active_constraints=list(self.world_state.active_constraints) + [new_constraint],
            timestamp_ns=5000,
        )

        self.assertNotEqual(ctx_baseline, ctx_constrained)
        self.assertIn("Force threshold 10N", ctx_constrained.active_constraints)

    # 13. TaskUnderstanding and HumanInstruction resolution
    def test_task_understanding_and_instruction_resolution(self):
        """Verify target object and objective resolution from TaskUnderstanding and HumanInstruction."""
        understanding = TaskUnderstanding(
            task_id="task_99",
            objective="MOVE_CUBE_CAREFULLY",
            target_object="red_cube",
            target_location="zone_drop",
            confidence=0.95,
            reasoning_summary="Pick up red cube carefully",
        )
        instruction = HumanInstruction(
            instruction="Move the red cube safely to the drop zone.",
            task_id="task_99",
        )

        ctx = self.builder.build(
            context_id="ctx_understanding",
            world_state=self.world_state,
            task_understanding=understanding,
            instruction=instruction,
        )

        self.assertEqual(ctx.task_objective, "MOVE_CUBE_CAREFULLY")
        self.assertEqual(ctx.target_object_summary["name"], "red_cube")

    # 14. Error handling: missing both situation and world_state
    def test_missing_state_raises_value_error(self):
        """Verify calling build without situation or world_state raises ValueError."""
        with self.assertRaises(ValueError):
            self.builder.build(context_id="ctx_invalid")


if __name__ == "__main__":
    unittest.main()
