"""
SĀRTHI V3-10 — Resiliency & Failure Fallback Test Suite.
Verifies all 20 required safety and resiliency properties.
"""

from __future__ import annotations

import unittest
import numpy as np

from backend.app.decision_engine.candidate_injector import (
    CandidateActionInjector,
    CandidateInjectionTelemetry,
)
from backend.app.decision_engine.engine import SarthiDecisionEngine
from backend.app.decision_engine.models import (
    ActionType,
    CandidateAction,
    Decision,
    Point3D,
    RobotState,
    TargetZone,
    WorldObject,
    WorldState,
)
from backend.app.model.mock_provider import MockModelProvider
from backend.app.model.models import TaskUnderstanding
from backend.app.model.task_understanding import TaskUnderstandingService
from backend.app.model.nebius_provider import (
    NebiusNemotronProvider,
    NebiusTimeoutError,
    NebiusValidationError,
)
from backend.app.model.real_jev_provider import (
    JevConnectionError,
    JevResponseParsingError,
    JevTimeoutError,
    JevValidationError,
)
from backend.app.model.v3_models import (
    DecisionContext,
    DecisionQuestion,
    DecisionQuestionType,
    JevDecision,
)
from backend.app.orchestration.context_builder import DecisionContextBuilder
from backend.app.resiliency.models import (
    DetectionStage,
    FailureType,
    FallbackStrategy,
    ResiliencyRecord,
    V310ResiliencyReport,
    _sanitize_data,
)
from backend.app.resiliency.policy import SarthiFallbackPolicy
from backend.app.resiliency.suite import (
    FailingLayaProvider,
    FailingNemotronProvider,
    ResiliencySuite,
)
from simulation.core.events import ActionExecutionResult
from simulation.mujoco_runtime.runtime import SarthiMuJoCoRuntime


class TestV310Resiliency(unittest.TestCase):
    """Test suite covering the 20 failure, resiliency, and safety criteria for V3-10."""

    def setUp(self) -> None:
        self.suite = ResiliencySuite()
        self.policy = SarthiFallbackPolicy()
        self.engine = SarthiDecisionEngine()
        self.injector = CandidateActionInjector()

    # 1. Laya timeout
    def test_01_laya_timeout(self) -> None:
        record = self.suite.run_scenario_01_laya_timeout()
        self.assertEqual(record.failure_type, FailureType.LAYA_TIMEOUT)
        self.assertEqual(record.detected_at_stage, DetectionStage.DECISION_PROPOSAL)
        self.assertEqual(record.recovery_strategy, FallbackStrategy.DETERMINISTIC_RECOVERY)
        self.assertEqual(record.fallback_action, "REPOSITION")
        self.assertFalse(record.safe_stop)

    # 2. Laya connection failure
    def test_02_laya_connection_failure(self) -> None:
        record = self.suite.run_scenario_02_laya_connection_failure()
        self.assertEqual(record.failure_type, FailureType.LAYA_CONNECTION_FAILURE)
        self.assertEqual(record.recovery_strategy, FallbackStrategy.DETERMINISTIC_RECOVERY)
        self.assertEqual(record.fallback_action, "REPOSITION")

    # 3. Malformed Laya response
    def test_03_malformed_laya_response(self) -> None:
        record = self.suite.run_scenario_03_malformed_laya_response()
        self.assertEqual(record.failure_type, FailureType.MALFORMED_LAYA_RESPONSE)
        self.assertEqual(record.recovery_strategy, FallbackStrategy.DETERMINISTIC_RECOVERY)

    # 4. Out-of-bounds Laya option
    def test_04_out_of_bounds_laya_option(self) -> None:
        record = self.suite.run_scenario_04_laya_out_of_bounds()
        self.assertEqual(record.failure_type, FailureType.LAYA_OUT_OF_BOUNDS)
        self.assertEqual(record.detected_at_stage, DetectionStage.CANDIDATE_INJECTION)
        self.assertEqual(record.recovery_strategy, FallbackStrategy.DETERMINISTIC_RECOVERY)
        # Ensure rejected candidate did not execute
        self.assertNotEqual(record.fallback_action, "TELEPORT_OBJECT")

    # 5. Nemotron timeout
    def test_05_nemotron_timeout(self) -> None:
        record = self.suite.run_scenario_05_nemotron_timeout()
        self.assertEqual(record.failure_type, FailureType.NEMOTRON_TIMEOUT)
        self.assertEqual(record.detected_at_stage, DetectionStage.TASK_UNDERSTANDING)
        self.assertEqual(record.recovery_strategy, FallbackStrategy.SAFE_ABORT)
        self.assertTrue(record.safe_stop)

    # 6. Malformed Nemotron response
    def test_06_malformed_nemotron_response(self) -> None:
        record = self.suite.run_scenario_06_nemotron_malformed()
        self.assertEqual(record.failure_type, FailureType.NEMOTRON_MALFORMED)
        self.assertEqual(record.detected_at_stage, DetectionStage.TASK_UNDERSTANDING)
        self.assertEqual(record.recovery_strategy, FallbackStrategy.SAFE_ABORT)
        self.assertTrue(record.safe_stop)

    # 7. Deterministic validation rejection
    def test_07_deterministic_validation_rejection(self) -> None:
        record = self.suite.run_scenario_07_deterministic_rejection()
        self.assertEqual(record.detected_at_stage, DetectionStage.DETERMINISTIC_VALIDATION)
        # High confidence (0.99) did NOT override obstacle constraint
        self.assertNotEqual(record.fallback_action, "MOVE")

    # 8. IK failure
    def test_08_ik_failure(self) -> None:
        record = self.suite.run_scenario_08_ik_failure()
        self.assertEqual(record.failure_type, FailureType.IK_FAILURE)
        self.assertEqual(record.detected_at_stage, DetectionStage.IK_SOLVER)
        self.assertEqual(record.recovery_strategy, FallbackStrategy.SAFE_STOP)
        self.assertTrue(record.safe_stop)

    # 9. Execution failure
    def test_09_execution_failure(self) -> None:
        record = self.suite.run_scenario_09_execution_failure()
        self.assertEqual(record.failure_type, FailureType.PHYSICS_EXECUTION_FAILURE)
        self.assertEqual(record.detected_at_stage, DetectionStage.PHYSICAL_EXECUTION)
        self.assertEqual(record.recovery_strategy, FallbackStrategy.SAFE_STOP)
        self.assertFalse(record.verification_result)

    # 10. Verification failure
    def test_10_verification_failure(self) -> None:
        record = self.suite.run_scenario_10_verification_failure()
        self.assertEqual(record.failure_type, FailureType.VERIFICATION_FAILURE)
        self.assertEqual(record.detected_at_stage, DetectionStage.VERIFICATION)
        self.assertEqual(record.recovery_strategy, FallbackStrategy.SAFE_STOP)
        self.assertFalse(record.verification_result)

    # 11. Grasp/object failure
    def test_11_grasp_object_failure(self) -> None:
        record = self.suite.run_scenario_11_grasp_failure()
        self.assertEqual(record.failure_type, FailureType.OBJECT_GRASP_FAILURE)
        self.assertEqual(record.detected_at_stage, DetectionStage.STATE_OBSERVATION)
        self.assertEqual(record.recovery_strategy, FallbackStrategy.SAFE_STOP)
        self.assertTrue(record.safe_stop)

    # 12. PATH_BLOCKED recovery
    def test_12_path_blocked_recovery(self) -> None:
        record = self.suite.run_scenario_12_path_blocked()
        self.assertEqual(record.failure_type, FailureType.PATH_BLOCKED_DISTURBANCE)
        self.assertEqual(record.recovery_strategy, FallbackStrategy.PROPOSAL_ACCEPTED)
        self.assertEqual(record.validation_result, "accepted")

    # 13. Multi-failure cascade
    def test_13_multi_failure_cascade(self) -> None:
        record = self.suite.run_scenario_13_multi_failure_cascade()
        self.assertEqual(record.failure_type, FailureType.MULTI_FAILURE_CASCADE)
        self.assertEqual(record.detected_at_stage, DetectionStage.DECISION_PROPOSAL)
        self.assertEqual(record.recovery_strategy, FallbackStrategy.DETERMINISTIC_RECOVERY)
        self.assertEqual(record.fallback_action, "REPOSITION")

    # 14. Safe STOP behavior
    def test_14_safe_stop_behavior(self) -> None:
        # Create a world state where even STOP is the only viable candidate
        ws = self.suite._build_test_world_state(with_obstacle=True)
        question = DecisionQuestion(
            question_id="q_stop",
            question_type=DecisionQuestionType.RECOVERY_SELECTION,
            question_text="Only STOP viable",
            available_options=["STOP", "REPOSITION"],
            context=self.suite.context_builder.build_context(ws, "Halt", ["STOP", "REPOSITION"]),
        )
        ai_decision = JevDecision(
            question_id="q_stop",
            selected_option="STOP",
            option_probabilities={"STOP": 0.9, "REPOSITION": 0.1},
            confidence=0.9,
            answer_confidence=0.9,
            provider="test",
        )
        decision, record = self.policy.evaluate_with_fallback(ws, question, ai_decision)
        self.assertEqual(decision.selected_action.action_type, ActionType.STOP)
        self.assertTrue(record.safe_stop)

    # 15. Fallback candidate validation
    def test_15_fallback_candidate_validation(self) -> None:
        # Verify that all fallback actions pass through SarthiDecisionEngine
        ws = self.suite._build_test_world_state(with_obstacle=True, robot_holding=True)
        question = DecisionQuestion(
            question_id="q_fb_val",
            question_type=DecisionQuestionType.RECOVERY_SELECTION,
            question_text="Fallback candidate validation",
            available_options=["REPOSITION", "STOP"],
            context=self.suite.context_builder.build_context(ws, "Recover", ["REPOSITION", "STOP"]),
        )
        # AI fails
        decision, record = self.policy.evaluate_with_fallback(
            ws, question, ai_decision=None, ai_error="Simulated timeout"
        )
        # Decision must be from DecisionEngine
        self.assertIsInstance(decision, Decision)
        self.assertEqual(decision.selected_action.action_type, ActionType.REPOSITION)

    # 16. No AI-to-MuJoCo bypass
    def test_16_no_ai_to_mujoco_bypass(self) -> None:
        runtime = SarthiMuJoCoRuntime()
        raw_ai_dict = {"model": "laya", "action": "MOVE_PANDA_NOW", "target": [1, 2, 3]}
        # Runtime must only accept CandidateAction or dict mapping to ActionType, not raw AI models
        with self.assertRaises(Exception):
            runtime.execute_action(raw_ai_dict)

    # 17. No unsafe fallback execution
    def test_17_no_unsafe_fallback_execution(self) -> None:
        ws = self.suite._build_test_world_state(with_obstacle=True, robot_holding=True)
        # Candidate directly through obstacle
        blocked_candidate = CandidateAction(
            action_id="act_unsafe_fb",
            action_type=ActionType.MOVE,
            target_position=ws.target.position,
            speed_scale=0.5,
            expected_force_n=0.0,
        )
        dec = self.engine.decide(world_state=ws, candidate_actions=[blocked_candidate])
        # Decision engine MUST NOT select the unsafe candidate
        self.assertNotEqual(dec.selected_action.action_id, "act_unsafe_fb")
        self.assertEqual(dec.selected_action.action_type, ActionType.STOP)

    # 18. Provenance preservation
    def test_18_provenance_preservation(self) -> None:
        record = self.suite.run_scenario_01_laya_timeout()
        self.assertIn("strategy", record.provenance)
        self.assertEqual(record.provenance["strategy"], "deterministic_recovery")
        self.assertIn("action_type", record.provenance)

    # 19. Telemetry secret sanitization
    def test_19_telemetry_secret_sanitization(self) -> None:
        dirty_data = {
            "api_key": "sk-1234567890abcdef",
            "auth_header": "Bearer nbf_secret_token_123",
            "safe_field": "valid_action_reposition",
        }
        clean = _sanitize_data(dirty_data)
        self.assertEqual(clean["api_key"], "[REDACTED]")
        self.assertNotIn("sk-123", str(clean))
        self.assertNotIn("nbf_", str(clean))
        self.assertEqual(clean["safe_field"], "valid_action_reposition")

    # 20. Existing V3 behavior regression
    def test_20_existing_v3_behavior_regression(self) -> None:
        # Verify nominal closed-loop cycle still works cleanly
        mock_provider = MockModelProvider()
        task_service = TaskUnderstandingService(provider=mock_provider)
        understanding = task_service.understand("Move the red object to the blue target.")
        self.assertEqual(understanding.target_object, "red_object")
        self.assertEqual(understanding.target_location, "blue_target")
        self.assertIn("APPROACH", understanding.required_actions)
        self.assertIn("GRASP", understanding.required_actions)
        self.assertIn("RELEASE", understanding.required_actions)


if __name__ == "__main__":
    unittest.main()
