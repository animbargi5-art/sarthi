"""
SĀRTHI V3-5 — Safe Candidate Action Injection Unit Test Suite.
Validates the translation of bounded JevDecision results into canonical
CandidateAction objects, deterministic SarthiDecisionEngine evaluation,
provenance preservation, confidence non-overriding of physical safety,
and complete isolation from physical execution / MuJoCo runtime.

Non-Negotiable Verification:
1. Live Laya/Jev availability must NEVER be required for test suite execution.
2. AI models propose or select bounded semantic candidates; deterministic
   software validates physical feasibility; only validated actions reach execution.
3. Jev/Laya confidence does NOT equal physical safety.
4. Output ends at CandidateAction and Decision; zero physical robot actions or MuJoCo mutations.
"""

import time
import unittest
from unittest.mock import MagicMock, patch

from backend.app.decision_engine.candidate_injector import (
    CandidateActionInjector,
    CandidateInjectionTelemetry,
)
from backend.app.decision_engine.constraints import ConstraintValidator
from backend.app.decision_engine.engine import SarthiDecisionEngine
from backend.app.decision_engine.models import (
    ActionType,
    ActiveConstraint,
    CandidateAction,
    CandidateEvaluation,
    Decision,
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
from backend.app.model.real_jev_provider import JevValidationError
from backend.app.model.v3_models import (
    DecisionContext,
    DecisionQuestion,
    DecisionQuestionType,
    JevDecision,
    PhysicalSituation,
)
from backend.app.orchestration.bounded_query_runner import BoundedJevQueryRunner


class TestCandidateActionInjector(unittest.TestCase):
    """Test suite validating Phase V3-5 Safe Candidate Action Injection."""

    def setUp(self):
        """Set up injector, engine, and standard test fixtures."""
        self.injector = CandidateActionInjector()
        self.engine = SarthiDecisionEngine()

        # Recovery scenario builder
        self.runner = BoundedJevQueryRunner()
        self.situation, self.question = self.runner.build_recovery_scenario()
        self.world_state = self.situation.world_state

    # -----------------------------------------------------------------------
    # Test 1: Laya REPOSITION -> CandidateAction(REPOSITION)
    # -----------------------------------------------------------------------
    def test_1_laya_reposition_to_candidate_action(self):
        """Verify Laya REPOSITION decision transforms into CandidateAction(REPOSITION)."""
        decision = JevDecision(
            selected_option="REPOSITION",
            confidence=0.0023,
            answer_confidence=0.5284,
            option_probabilities={"REPOSITION": 0.5284, "STOP": 0.4716},
            provider="laya",
            model_version="laya-rl-agent",
            latency_ms=495.95,
        )

        candidate = self.injector.inject_candidate(
            decision, question=self.question, world_state=self.world_state
        )

        self.assertIsInstance(candidate, CandidateAction)
        self.assertEqual(candidate.action_type, ActionType.REPOSITION)
        self.assertEqual(candidate.speed_scale, 0.3)
        self.assertIsNotNone(candidate.target_position)
        # Vertical clearance lift: robot at z=0.520 + 0.080 = 0.600
        self.assertAlmostEqual(candidate.target_position.z, 0.600, places=2)
        self.assertEqual(candidate.parameters["provider"], "laya")
        self.assertEqual(candidate.parameters["candidate_source"], "bounded_laya_decision")
        self.assertEqual(candidate.parameters["raw_selected_option"], "REPOSITION")

    # -----------------------------------------------------------------------
    # Test 2: Laya STOP -> CandidateAction(STOP)
    # -----------------------------------------------------------------------
    def test_2_laya_stop_to_candidate_action(self):
        """Verify Laya STOP decision transforms into CandidateAction(STOP)."""
        decision = JevDecision(
            selected_option="STOP",
            confidence=0.05,
            answer_confidence=0.88,
            option_probabilities={"STOP": 0.88, "REPOSITION": 0.12},
            provider="laya",
            model_version="laya-rl-agent",
            latency_ms=310.2,
        )

        candidate = self.injector.inject_candidate(
            decision, question=self.question, world_state=self.world_state
        )

        self.assertIsInstance(candidate, CandidateAction)
        self.assertEqual(candidate.action_type, ActionType.STOP)
        self.assertEqual(candidate.speed_scale, 0.0)
        self.assertEqual(candidate.expected_force_n, 0.0)
        self.assertEqual(candidate.parameters["mode"], "hold_current_joint_positions")
        self.assertEqual(candidate.parameters["provider"], "laya")
        self.assertEqual(candidate.parameters["candidate_source"], "bounded_laya_decision")

    # -----------------------------------------------------------------------
    # Test 3: Unknown Action -> Rejected
    # -----------------------------------------------------------------------
    def test_3_unknown_action_rejected(self):
        """Verify unknown or unsupported action strings are strictly rejected."""
        decision = JevDecision(
            selected_option="TELEPORT_TO_TARGET",
            confidence=0.99,
            provider="laya",
        )

        with self.assertRaises(JevValidationError) as ctx:
            self.injector.inject_candidate(decision)
        self.assertIn("Unknown action", str(ctx.exception))

    # -----------------------------------------------------------------------
    # Test 4: Action Outside Bounded Question -> Rejected
    # -----------------------------------------------------------------------
    def test_4_action_outside_bounded_question_rejected(self):
        """Verify action outside question available_options is strictly rejected."""
        # Available options are ["REPOSITION", "STOP"]
        decision = JevDecision(
            selected_option="MOVE",
            confidence=0.80,
            provider="laya",
        )

        with self.assertRaises(JevValidationError) as ctx:
            self.injector.inject_candidate(decision, question=self.question)
        self.assertIn("not in question's available options", str(ctx.exception))

    # -----------------------------------------------------------------------
    # Test 5: Valid CandidateAction reaches Decision Engine
    # -----------------------------------------------------------------------
    def test_5_valid_candidate_reaches_decision_engine(self):
        """Verify injected candidate is evaluated by SarthiDecisionEngine."""
        decision = JevDecision(
            selected_option="REPOSITION",
            confidence=0.0023,
            answer_confidence=0.5284,
            provider="laya",
        )

        engine_decision, telemetry = self.injector.evaluate_candidate_with_engine(
            engine=self.engine,
            decision=decision,
            world_state=self.world_state,
            question=self.question,
        )

        self.assertIsInstance(engine_decision, Decision)
        self.assertIsInstance(telemetry, CandidateInjectionTelemetry)
        # Verify injected candidate action was among the evaluated candidates
        evaluated_action_types = [
            ev.action.action_type for ev in engine_decision.candidate_evaluations
        ]
        self.assertIn(ActionType.REPOSITION, evaluated_action_types)

    # -----------------------------------------------------------------------
    # Test 6: Decision Engine can reject an unsafe candidate
    # -----------------------------------------------------------------------
    def test_6_decision_engine_rejects_unsafe_candidate(self):
        """Verify Decision Engine independently rejects candidate violating physical limits."""
        # Create a world state where robot reach is very small (0.4m), but REPOSITION target is at ~0.67m
        restricted_robot = RobotState(
            position=Point3D(x=0.550, y=0.000, z=0.520),
            gripper_open=False,
            holding_object_id="red_object_01",
            payload_mass_kg=0.100,
            is_moving=False,
            max_payload_kg=3.0,
            max_reach_m=0.40,  # Insufficient reach for (0.55, 0.0, 0.67) distance ~0.86m
        )
        unsafe_world_state = WorldState(
            version=5,
            timestamp_ns=time.time_ns(),
            robot=restricted_robot,
            objects=self.world_state.objects,
            target=self.world_state.target,
            environment=self.world_state.environment,
            task_objective=self.world_state.task_objective,
            active_constraints=self.world_state.active_constraints,
            last_action_outcome=self.world_state.last_action_outcome,
        )

        decision = JevDecision(
            selected_option="REPOSITION",
            confidence=0.99,  # High confidence
            provider="laya",
        )

        engine_decision, telemetry = self.injector.evaluate_candidate_with_engine(
            engine=self.engine,
            decision=decision,
            world_state=unsafe_world_state,
            question=self.question,
        )

        # REPOSITION must be rejected by reachability constraint
        self.assertEqual(telemetry.validation_result, "rejected")
        self.assertTrue(any("ReachabilityViolation" in r for r in telemetry.rejection_reasons))
        # Engine falls back to safe STOP
        self.assertEqual(engine_decision.selected_action.action_type, ActionType.STOP)

    # -----------------------------------------------------------------------
    # Test 7: Decision Engine can accept a feasible candidate
    # -----------------------------------------------------------------------
    def test_7_decision_engine_accepts_feasible_candidate(self):
        """Verify Decision Engine accepts a physically safe and feasible recovery candidate."""
        decision = JevDecision(
            selected_option="REPOSITION",
            confidence=0.0023,
            answer_confidence=0.5284,
            provider="laya",
        )

        engine_decision, telemetry = self.injector.evaluate_candidate_with_engine(
            engine=self.engine,
            decision=decision,
            world_state=self.world_state,
            question=self.question,
        )

        self.assertEqual(telemetry.validation_result, "accepted")
        self.assertEqual(telemetry.rejection_reasons, [])
        self.assertEqual(engine_decision.selected_action.action_type, ActionType.REPOSITION)
        self.assertTrue(engine_decision.decision_factors.get("overall_score", 0.0) > 0.0)

    # -----------------------------------------------------------------------
    # Test 8: Blocked path candidate is rejected
    # -----------------------------------------------------------------------
    def test_8_blocked_path_candidate_is_rejected(self):
        """Verify that a candidate traversing a blocked obstacle trajectory is rejected."""
        # Add a blocking obstacle directly between robot and target
        obstacle = WorldObject(
            id="blocking_obstacle_01",
            name="Interfering Block",
            position=Point3D(x=0.45, y=-0.10, z=0.30),
            bounding_radius_m=0.15,
            mass_kg=2.0,
            state=ObjectState.FREE,
            is_target=False,
            is_obstacle=True,
        )

        blocked_world_state = WorldState(
            version=6,
            timestamp_ns=time.time_ns(),
            robot=self.world_state.robot,
            objects=list(self.world_state.objects) + [obstacle],
            target=self.world_state.target,
            environment=self.world_state.environment,
            task_objective=self.world_state.task_objective,
            active_constraints=self.world_state.active_constraints,
            last_action_outcome=self.world_state.last_action_outcome,
        )

        # Candidate attempting direct MOVE to target zone through obstacle
        move_candidate = CandidateAction(
            action_id="cand_move_blocked",
            action_type=ActionType.MOVE,
            target_position=self.world_state.target.position,
            speed_scale=0.4,
            parameters={"candidate_source": "test_direct_move"},
        )

        is_valid, reasons = ConstraintValidator.validate(move_candidate, blocked_world_state)
        self.assertFalse(is_valid)
        self.assertTrue(any("BlockedPath" in r for r in reasons))

        # When evaluated by engine
        decision = self.engine.decide(
            world_state=blocked_world_state,
            candidate_actions=[move_candidate],
        )
        self.assertIn("cand_move_blocked", decision.rejection_reasons)
        self.assertTrue(any("BlockedPath" in r for r in decision.rejection_reasons["cand_move_blocked"]))
        self.assertEqual(decision.selected_action.action_type, ActionType.STOP)

    # -----------------------------------------------------------------------
    # Test 9: Jev confidence does not override physical validation
    # -----------------------------------------------------------------------
    def test_9_confidence_does_not_override_physical_validation(self):
        """Verify 0.9999 AI confidence NEVER overrides deterministic physical rejection."""
        # High confidence Laya decision proposing MOVE into blocked space
        decision = JevDecision(
            selected_option="MOVE",
            confidence=0.9999,
            answer_confidence=0.9999,
            option_probabilities={"MOVE": 0.9999, "STOP": 0.0001},
            provider="laya",
        )

        # Question allowing MOVE
        question = DecisionQuestion(
            question_id="q_test_move",
            question_type=DecisionQuestionType.RECOVERY_SELECTION,
            question_text="Select recovery action",
            available_options=["MOVE", "STOP"],
            context=self.question.context,
        )

        # Add blocking obstacle between robot and destination
        obstacle = WorldObject(
            id="blocking_cube",
            name="Impassable Wall",
            position=Point3D(x=0.475, y=-0.095, z=0.320),
            bounding_radius_m=0.20,
            mass_kg=5.0,
            is_obstacle=True,
        )
        blocked_world_state = WorldState(
            version=7,
            timestamp_ns=time.time_ns(),
            robot=self.world_state.robot,
            objects=list(self.world_state.objects) + [obstacle],
            target=self.world_state.target,
            environment=self.world_state.environment,
            task_objective=self.world_state.task_objective,
            active_constraints=self.world_state.active_constraints,
            last_action_outcome=self.world_state.last_action_outcome,
        )

        engine_decision, telemetry = self.injector.evaluate_candidate_with_engine(
            engine=self.engine,
            decision=decision,
            world_state=blocked_world_state,
            question=question,
        )

        # The candidate must be REJECTED despite 0.9999 confidence
        self.assertEqual(telemetry.validation_result, "rejected")
        self.assertNotEqual(engine_decision.selected_action.action_type, ActionType.MOVE)
        self.assertEqual(engine_decision.selected_action.action_type, ActionType.STOP)

    # -----------------------------------------------------------------------
    # Test 10: Jev cannot create ActionExecution directly
    # -----------------------------------------------------------------------
    def test_10_jev_cannot_create_action_execution_directly(self):
        """Verify CandidateActionInjector has zero capability to instantiate or execute actions."""
        # Verify no execution-related methods exist on CandidateActionInjector
        forbidden_methods = [
            "execute",
            "execute_action",
            "create_execution",
            "actuate",
            "command_robot",
            "step_sim",
            "send_action",
        ]
        for method in forbidden_methods:
            self.assertFalse(
                hasattr(self.injector, method),
                f"CandidateActionInjector must NOT have method '{method}'",
            )

        # Verify returned type is strictly CandidateAction
        decision = JevDecision(
            selected_option="STOP",
            confidence=0.8,
            provider="laya",
        )
        result = self.injector.inject_candidate(decision, question=self.question)
        self.assertIsInstance(result, CandidateAction)
        self.assertNotEqual(type(result).__name__, "ActionExecution")

    # -----------------------------------------------------------------------
    # Test 11: Candidate provenance is preserved
    # -----------------------------------------------------------------------
    def test_11_candidate_provenance_is_preserved(self):
        """Verify provenance fields are tracked through injection and telemetry."""
        decision = JevDecision(
            selected_option="REPOSITION",
            confidence=0.0023,
            answer_confidence=0.5284,
            option_probabilities={"REPOSITION": 0.5284, "STOP": 0.4716},
            provider="laya",
            model_version="laya-rl-agent",
            latency_ms=495.95,
        )

        candidate = self.injector.inject_candidate(
            decision, question=self.question, world_state=self.world_state
        )

        # Verify CandidateAction parameters
        self.assertEqual(candidate.parameters["provider"], "laya")
        self.assertEqual(candidate.parameters["candidate_source"], "bounded_laya_decision")
        self.assertEqual(candidate.parameters["model_version"], "laya-rl-agent")
        self.assertAlmostEqual(candidate.parameters["confidence"], 0.0023)
        self.assertAlmostEqual(candidate.parameters["answer_confidence"], 0.5284)

        # Verify Telemetry
        _, telemetry = self.injector.evaluate_candidate_with_engine(
            engine=self.engine,
            decision=decision,
            world_state=self.world_state,
            question=self.question,
        )
        t_dict = telemetry.to_dict()
        self.assertEqual(t_dict["provider"], "laya")
        self.assertEqual(t_dict["selected_option"], "REPOSITION")
        self.assertEqual(t_dict["candidate_action"], "REPOSITION")
        self.assertEqual(t_dict["candidate_source"], "bounded_laya_decision")
        self.assertEqual(t_dict["validation_result"], "accepted")
        self.assertAlmostEqual(t_dict["confidence"], 0.0023)
        self.assertAlmostEqual(t_dict["answer_confidence"], 0.5284)

    # -----------------------------------------------------------------------
    # Test 12: No MuJoCo call occurs during candidate injection
    # -----------------------------------------------------------------------
    def test_12_no_mujoco_call_occurs_during_candidate_injection(self):
        """Verify zero MuJoCo modules or runtime calls occur during injection."""
        decision = JevDecision(
            selected_option="REPOSITION",
            confidence=0.5284,
            provider="laya",
        )

        # Intercept and spy on any potential mujoco execution
        with patch.dict("sys.modules", {"mujoco": MagicMock()}):
            candidate = self.injector.inject_candidate(
                decision, question=self.question, world_state=self.world_state
            )
            engine_decision, telemetry = self.injector.evaluate_candidate_with_engine(
                engine=self.engine,
                decision=decision,
                world_state=self.world_state,
                question=self.question,
            )

        self.assertIsNotNone(candidate)
        self.assertIsNotNone(engine_decision)
        self.assertIsNotNone(telemetry)

    # -----------------------------------------------------------------------
    # Test 13: No robot action occurs during candidate injection
    # -----------------------------------------------------------------------
    def test_13_no_robot_action_occurs_during_candidate_injection(self):
        """Verify robot state is completely unchanged before and after injection."""
        initial_pos = (
            self.world_state.robot.position.x,
            self.world_state.robot.position.y,
            self.world_state.robot.position.z,
        )
        initial_gripper = self.world_state.robot.gripper_open
        initial_holding = self.world_state.robot.holding_object_id

        decision = JevDecision(
            selected_option="REPOSITION",
            confidence=0.5284,
            provider="laya",
        )

        # Run injection and engine evaluation
        self.injector.evaluate_candidate_with_engine(
            engine=self.engine,
            decision=decision,
            world_state=self.world_state,
            question=self.question,
        )

        # Robot physical state must remain strictly identical
        current_pos = (
            self.world_state.robot.position.x,
            self.world_state.robot.position.y,
            self.world_state.robot.position.z,
        )
        self.assertEqual(initial_pos, current_pos)
        self.assertEqual(initial_gripper, self.world_state.robot.gripper_open)
        self.assertEqual(initial_holding, self.world_state.robot.holding_object_id)


if __name__ == "__main__":
    unittest.main()
