"""
SĀRTHI Decision Engine — Unit Test Suite.
Validates deterministic evaluation, constraint filtering, blocked-path rejection,
alternative selection, and repeatability without stochastic external dependencies.
"""

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
from backend.app.decision_engine.engine import SarthiDecisionEngine
from backend.app.decision_engine.constraints import ConstraintValidator


class TestSarthiDecisionEngine(unittest.TestCase):
    """Unit test cases verifying core requirements of the SĀRTHI Decision Engine."""

    def setUp(self):
        self.engine = SarthiDecisionEngine()

        # Baseline standard robot state
        self.standard_robot = RobotState(
            position=Point3D(x=0.0, y=0.0, z=0.2),
            gripper_open=True,
            holding_object_id=None,
            payload_mass_kg=0.0,
            is_moving=False,
            max_payload_kg=3.0,
            max_reach_m=0.85,
        )

        # Baseline target zone
        self.standard_target = TargetZone(
            id="target_zone_A",
            position=Point3D(x=0.4, y=-0.2, z=0.1),
            tolerance_radius_m=0.05,
        )

        # Baseline environment
        self.standard_env = EnvironmentState(
            min_x=-0.8, max_x=0.8,
            min_y=-0.8, max_y=0.8,
            min_z=0.0, max_z=1.0,
            dynamic_obstacles_detected=False,
            slip_risk_level=0.1,
            friction_coefficient=0.6,
        )

    def test_feasible_action_selection(self):
        """Verify that in a nominal clear state, the engine selects a feasible, goal-advancing action."""
        target_obj = WorldObject(
            id="obj_cylinder_01",
            name="Cylinder Alpha",
            position=Point3D(x=0.3, y=0.2, z=0.2),
            bounding_radius_m=0.04,
            mass_kg=0.5,
            state=ObjectState.FREE,
            is_target=True,
            is_obstacle=False,
        )

        world_state = WorldState(
            version="ws_101",
            timestamp_ns=1000000,
            robot=self.standard_robot,
            objects=[target_obj],
            target=self.standard_target,
            environment=self.standard_env,
            task_objective=TaskObjective.PICK_AND_PLACE,
            active_constraints=[],
            last_action_outcome=LastActionOutcome(status=LastActionStatus.SUCCESS),
        )

        decision = self.engine.decide(world_state)

        # In this clear state, the robot is far from target object and not holding anything.
        # APPROACH should be valid, highly relevant, and selected over premature GRASP/RELEASE.
        self.assertIsNotNone(decision.selected_action)
        self.assertEqual(decision.world_state_version, "ws_101")
        self.assertEqual(decision.selected_action.action_type, ActionType.APPROACH)
        self.assertGreater(decision.decision_factors["overall_score"], 0.6)
        self.assertGreater(len(decision.candidate_evaluations), 0)

    def test_blocked_path_rejection(self):
        """Verify that actions traversing directly through an obstacle are rejected with BlockedPath reason."""
        target_obj = WorldObject(
            id="obj_cylinder_01",
            name="Cylinder Alpha",
            position=Point3D(x=0.5, y=0.0, z=0.2),
            bounding_radius_m=0.04,
            mass_kg=0.5,
            state=ObjectState.FREE,
            is_target=True,
            is_obstacle=False,
        )

        # Obstacle placed directly along segment from (0,0,0.2) to (0.5,0,0.2) at (0.25, 0, 0.2)
        obstacle = WorldObject(
            id="obstacle_block_01",
            name="Interfering Block",
            position=Point3D(x=0.25, y=0.0, z=0.2),
            bounding_radius_m=0.08,
            mass_kg=2.0,
            state=ObjectState.FREE,
            is_target=False,
            is_obstacle=True,
        )

        world_state = WorldState(
            version="ws_102",
            timestamp_ns=2000000,
            robot=self.standard_robot,
            objects=[target_obj, obstacle],
            target=self.standard_target,
            environment=self.standard_env,
            task_objective=TaskObjective.PICK_AND_PLACE,
            active_constraints=[],
            last_action_outcome=LastActionOutcome(status=LastActionStatus.SUCCESS),
        )

        blocked_action = CandidateAction(
            action_id="act_direct_blocked_move",
            action_type=ActionType.MOVE,
            target_position=Point3D(x=0.5, y=0.0, z=0.2),
            speed_scale=0.5,
        )

        is_valid, reasons = ConstraintValidator.validate(blocked_action, world_state)
        self.assertFalse(is_valid)
        self.assertTrue(any("BlockedPath" in r for r in reasons), f"Expected BlockedPath in reasons: {reasons}")

        # Engine decision must reject the blocked action
        decision = self.engine.decide(world_state, candidate_actions=[blocked_action])
        self.assertIn("act_direct_blocked_move", decision.rejection_reasons)
        self.assertTrue(any("BlockedPath" in r for r in decision.rejection_reasons["act_direct_blocked_move"]))
        # Fallback to safe stop
        self.assertEqual(decision.selected_action.action_type, ActionType.STOP)

    def test_constraint_violation(self):
        """Verify deterministic rejection for workspace boundaries, reachability, payload, and force limits."""
        heavy_obj = WorldObject(
            id="heavy_crate",
            name="Overweight Crate",
            position=Point3D(x=0.05, y=0.0, z=0.2),
            bounding_radius_m=0.05,
            mass_kg=8.5,  # Exceeds max_payload_kg of 3.0kg
            state=ObjectState.FREE,
            is_target=True,
            is_obstacle=False,
        )

        force_constraint = ActiveConstraint(
            constraint_id="c_force_limit",
            description="Delicate part force limit",
            max_force_newtons=12.0,
        )

        world_state = WorldState(
            version="ws_103",
            timestamp_ns=3000000,
            robot=self.standard_robot,
            objects=[heavy_obj],
            target=self.standard_target,
            environment=self.standard_env,
            task_objective=TaskObjective.PICK_AND_PLACE,
            active_constraints=[force_constraint],
            last_action_outcome=LastActionOutcome(status=LastActionStatus.SUCCESS),
        )

        # 1. Test Workspace Boundary Violation
        outside_action = CandidateAction(
            action_id="act_outside",
            action_type=ActionType.MOVE,
            target_position=Point3D(x=10.0, y=0.0, z=0.2),
        )
        is_valid, reasons = ConstraintValidator.validate(outside_action, world_state)
        self.assertFalse(is_valid)
        self.assertTrue(any("WorkspaceViolation" in r for r in reasons))

        # 2. Test Reachability Violation
        unreachable_action = CandidateAction(
            action_id="act_unreachable",
            action_type=ActionType.MOVE,
            target_position=Point3D(x=0.7, y=0.6, z=0.5),  # Distance ~1.04m > 0.85m
        )
        is_valid, reasons = ConstraintValidator.validate(unreachable_action, world_state)
        self.assertFalse(is_valid)
        self.assertTrue(any("ReachabilityViolation" in r for r in reasons))

        # 3. Test Payload Violation on Grasp
        overload_grasp = CandidateAction(
            action_id="act_grasp_heavy",
            action_type=ActionType.GRASP,
            target_object_id="heavy_crate",
            expected_force_n=8.0,
        )
        is_valid, reasons = ConstraintValidator.validate(overload_grasp, world_state)
        self.assertFalse(is_valid)
        self.assertTrue(any("PayloadViolation" in r for r in reasons))

        # 4. Test Force Constraint Breach
        excess_force_action = CandidateAction(
            action_id="act_force_violation",
            action_type=ActionType.REPOSITION,
            target_position=Point3D(x=0.2, y=0.0, z=0.25),
            expected_force_n=25.0,  # Exceeds 12.0N limit
        )
        is_valid, reasons = ConstraintValidator.validate(excess_force_action, world_state)
        self.assertFalse(is_valid)
        self.assertTrue(any("ForceLimitViolation" in r for r in reasons))

    def test_alternative_action_selection(self):
        """Verify that when a direct action is blocked, the engine selects a viable alternative."""
        target_obj = WorldObject(
            id="obj_target",
            name="Target Part",
            position=Point3D(x=0.4, y=0.0, z=0.2),
            bounding_radius_m=0.04,
            mass_kg=0.5,
            state=ObjectState.FREE,
            is_target=True,
            is_obstacle=False,
        )

        obstacle = WorldObject(
            id="obs_direct",
            name="Blocking Obstacle",
            position=Point3D(x=0.2, y=0.0, z=0.2),
            bounding_radius_m=0.06,
            mass_kg=1.0,
            state=ObjectState.FREE,
            is_target=False,
            is_obstacle=True,
        )

        world_state = WorldState(
            version="ws_104",
            timestamp_ns=4000000,
            robot=self.standard_robot,
            objects=[target_obj, obstacle],
            target=self.standard_target,
            environment=self.standard_env,
            task_objective=TaskObjective.PICK_AND_PLACE,
            active_constraints=[],
            last_action_outcome=LastActionOutcome(status=LastActionStatus.SUCCESS),
        )

        # Action A: Direct blocked trajectory
        act_blocked = CandidateAction(
            action_id="act_direct_approach",
            action_type=ActionType.APPROACH,
            target_position=Point3D(x=0.4, y=0.0, z=0.2),
            target_object_id="obj_target",
        )

        # Action B: Alternative detour waypoint above obstacle (Clear)
        act_detour = CandidateAction(
            action_id="act_clear_reposition",
            action_type=ActionType.REPOSITION,
            target_position=Point3D(x=0.0, y=0.0, z=0.45),  # Vertical lift avoids obstacle
            speed_scale=0.3,
        )

        # Action C: Safe stop
        act_stop = CandidateAction(
            action_id="act_stop",
            action_type=ActionType.STOP,
        )

        decision = self.engine.decide(
            world_state=world_state,
            candidate_actions=[act_blocked, act_detour, act_stop]
        )

        # The blocked action must be rejected
        self.assertIn("act_direct_approach", decision.rejection_reasons)

        # The engine must successfully select the viable detour alternative
        self.assertEqual(decision.selected_action.action_id, "act_clear_reposition")
        self.assertEqual(decision.selected_action.action_type, ActionType.REPOSITION)
        self.assertGreater(decision.decision_factors["overall_score"], 0.0)

    def test_deterministic_repeated_decision(self):
        """Verify that multiple consecutive invocations on identical input yield bit-for-bit identical decisions."""
        target_obj = WorldObject(
            id="obj_cube_01",
            name="Target Cube",
            position=Point3D(x=0.35, y=0.15, z=0.2),
            bounding_radius_m=0.04,
            mass_kg=0.8,
            state=ObjectState.FREE,
            is_target=True,
            is_obstacle=False,
        )

        world_state = WorldState(
            version="ws_repeat_test_v1",
            timestamp_ns=5555555,
            robot=self.standard_robot,
            objects=[target_obj],
            target=self.standard_target,
            environment=self.standard_env,
            task_objective=TaskObjective.PICK_AND_PLACE,
            active_constraints=[],
            last_action_outcome=LastActionOutcome(status=LastActionStatus.SUCCESS),
        )

        # First run baseline
        baseline_decision = self.engine.decide(world_state, decision_id="dec_run_0")

        # Repeat 10 times and verify strict equivalence
        for i in range(1, 11):
            repeat_decision = self.engine.decide(world_state, decision_id=f"dec_run_{i}")

            self.assertEqual(
                baseline_decision.selected_action.action_id,
                repeat_decision.selected_action.action_id,
                f"Mismatch in selected action on run {i}"
            )
            self.assertEqual(
                baseline_decision.selected_action.action_type,
                repeat_decision.selected_action.action_type,
            )
            self.assertEqual(
                baseline_decision.decision_factors,
                repeat_decision.decision_factors,
                f"Mismatch in decision factors on run {i}"
            )
            self.assertEqual(
                len(baseline_decision.candidate_evaluations),
                len(repeat_decision.candidate_evaluations),
            )
            for ev1, ev2 in zip(baseline_decision.candidate_evaluations, repeat_decision.candidate_evaluations):
                self.assertEqual(ev1.action.action_id, ev2.action.action_id)
                self.assertEqual(ev1.is_valid, ev2.is_valid)
                self.assertEqual(ev1.overall_score, ev2.overall_score)
                self.assertEqual(ev1.responsibility_score, ev2.responsibility_score)
                self.assertEqual(ev1.relevance_score, ev2.relevance_score)
                self.assertEqual(ev1.consequence_score, ev2.consequence_score)


if __name__ == "__main__":
    unittest.main()
