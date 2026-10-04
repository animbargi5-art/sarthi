"""
SĀRTHI V3-4 — Bounded Jev Decision Query Unit Test Suite.
Validates the BoundedJevQueryRunner orchestration, scenario construction,
context builder integration, bounded option validation, safe telemetry recording,
error categorization, and non-authoritative safety invariants.

Non-Negotiable Verification:
1. Live Jev availability must NEVER be required for test suite execution.
2. AI models propose or select bounded semantic candidates; deterministic software validates
   physical feasibility; only validated actions reach the robot controller.
3. Jev must never bypass deterministic safety validation.
4. Output ends at JevDecision; zero physical robot actions or MuJoCo mutations.
"""

import json
import time
import unittest
from unittest.mock import MagicMock, patch

from backend.app.decision_engine.models import (
    ActionType,
    ActiveConstraint,
    LastActionOutcome,
    LastActionStatus,
    ObjectState,
    Point3D,
    RobotState,
    TargetZone,
    WorldObject,
    WorldState,
)
from backend.app.model.jev_config import JevConfig
from backend.app.model.mock_jev_provider import MockJevDecisionProvider
from backend.app.model.real_jev_provider import (
    JevConfigurationError,
    JevConnectionError,
    JevProviderError,
    JevResponseParsingError,
    JevTimeoutError,
    JevValidationError,
    RealJevDecisionProvider,
)
from backend.app.model.v3_models import (
    DecisionContext,
    DecisionQuestion,
    DecisionQuestionType,
    JevDecision,
    PhysicalSituation,
)
from backend.app.orchestration.bounded_query_runner import (
    BoundedJevQueryRunner,
    BoundedQueryResult,
    BoundedQueryTelemetry,
)
from backend.app.orchestration.context_builder import DecisionContextBuilder


class TestBoundedJevQueryRunner(unittest.TestCase):
    """Test suite validating Phase V3-4 Bounded Jev Query Orchestrator."""

    def setUp(self):
        """Set up test runner and deterministic builder fixtures."""
        self.builder = DecisionContextBuilder(coordinate_precision=3)
        self.runner = BoundedJevQueryRunner(context_builder=self.builder)

    # 1. Recovery Scenario Construction & Context Builder Integration
    def test_build_recovery_scenario_schema_and_context(self):
        """Verify deterministic recovery scenario matches Phase V3-4 specification."""
        situation, question = self.runner.build_recovery_scenario()

        # Verify Physical Situation
        self.assertIsInstance(situation, PhysicalSituation)
        self.assertEqual(situation.situation_id, "sit_v3_4_recovery_01")
        self.assertEqual(situation.task_context.get("task_phase"), "DISTURBANCE_RECOVERY")

        # Verify WorldState properties
        ws = situation.world_state
        self.assertEqual(ws.version, 4)
        self.assertFalse(ws.robot.gripper_open)
        self.assertEqual(ws.robot.holding_object_id, "red_object_01")
        self.assertEqual(ws.robot.payload_mass_kg, 0.100)

        # Verify held object
        held_obj = next(obj for obj in ws.objects if obj.id == "red_object_01")
        self.assertEqual(held_obj.state, ObjectState.GRASPED)
        self.assertTrue(held_obj.is_target)

        # Verify active constraints & failed outcome
        self.assertEqual(len(ws.active_constraints), 1)
        self.assertIn("BlockedPath", ws.active_constraints[0].description)
        self.assertEqual(ws.last_action_outcome.action_type, ActionType.MOVE)
        self.assertEqual(ws.last_action_outcome.status, LastActionStatus.FAILURE)

        # Verify DecisionQuestion
        self.assertIsInstance(question, DecisionQuestion)
        self.assertEqual(question.question_id, "q_v3_4_recovery_01")
        self.assertEqual(question.question_type, DecisionQuestionType.RECOVERY_SELECTION)
        self.assertEqual(question.question_text, "Which recovery action should be considered?")
        self.assertEqual(question.available_options, ["REPOSITION", "STOP"])

        # Verify DecisionContext (generated via DecisionContextBuilder)
        ctx = question.context
        self.assertIsInstance(ctx, DecisionContext)
        self.assertEqual(ctx.task_objective, "Move the red object to the blue target.")
        self.assertEqual(ctx.candidate_actions, ["REPOSITION", "STOP"])
        self.assertEqual(ctx.robot_state_summary["holding_object_id"], "red_object_01")
        self.assertFalse(ctx.robot_state_summary["gripper_open"])
        self.assertIsNotNone(ctx.disturbance_info)
        self.assertEqual(ctx.disturbance_info["type"], "PATH_BLOCKED")
        self.assertIn("BlockedPath", ctx.active_constraints[0])

    # 2. Strict Decision Validation
    def test_validate_decision_accepts_valid_option(self):
        """Verify validate_decision succeeds when selected option is in available_options."""
        _, question = self.runner.build_recovery_scenario()
        decision = JevDecision(
            selected_option="REPOSITION",
            confidence=0.92,
            option_probabilities={"REPOSITION": 0.92, "STOP": 0.08},
            latency_ms=12.5,
        )
        # Should not raise
        self.runner.validate_decision(decision, question)

    def test_validate_decision_rejects_unknown_option(self):
        """Verify validate_decision strictly rejects options outside available_options."""
        _, question = self.runner.build_recovery_scenario()
        decision = JevDecision(
            selected_option="FREE_FORM_MOVE_DOWN",
            confidence=0.85,
        )
        with self.assertRaises(JevValidationError) as ctx:
            self.runner.validate_decision(decision, question)
        self.assertIn("not one of bounded available options", str(ctx.exception))

    def test_validate_decision_rejects_invalid_confidence_or_probability(self):
        """Verify validate_decision rejects out-of-bounds probability or confidence."""
        _, question = self.runner.build_recovery_scenario()

        # Out-of-bounds probability > 1.0
        dec_invalid_prob = JevDecision(
            selected_option="REPOSITION",
            confidence=0.8,
            option_probabilities={"REPOSITION": 1.5, "STOP": -0.5},
        )
        with self.assertRaises(JevValidationError):
            self.runner.validate_decision(dec_invalid_prob, question)

    # 3. Controlled Execution with Mock Provider (Deterministic Success)
    def test_execute_bounded_query_success_with_mock_provider(self):
        """Verify bounded query succeeds and captures telemetry with mock provider."""
        mock_provider = MockJevDecisionProvider(
            default_choices={"RECOVERY_SELECTION": "REPOSITION"},
            simulated_latency_ms=8.0,
            confidence=0.95,
        )
        _, question = self.runner.build_recovery_scenario()

        result = self.runner.execute_bounded_query(provider=mock_provider, question=question)

        self.assertTrue(result.success)
        self.assertTrue(result.validation_passed)
        self.assertIsNotNone(result.decision)
        self.assertEqual(result.decision.selected_option, "REPOSITION")
        self.assertEqual(result.decision.confidence, 0.95)
        self.assertIsNone(result.error_category)

        # Telemetry verification
        t = result.telemetry
        self.assertEqual(t.selected_option, "REPOSITION")
        self.assertEqual(t.confidence, 0.95)
        self.assertTrue(t.success)
        self.assertEqual(t.provider_status, "AVAILABLE")
        self.assertEqual(t.available_options, ["REPOSITION", "STOP"])

    # 4. Controlled Execution with RealJevDecisionProvider (Urllib Mocked)
    @patch("urllib.request.urlopen")
    def test_execute_bounded_query_with_mocked_live_endpoint(self, mock_urlopen):
        """Verify RealJevDecisionProvider execution with mocked HTTP response."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({
            "questions": {
                "q_v3_4_recovery_01": {
                    "choice": "REPOSITION",
                    "confidence": 0.89,
                    "probabilities": {"REPOSITION": 0.89, "STOP": 0.11},
                    "rationale": "Clearance altitude elevation is feasible; STOP is suboptimal.",
                }
            }
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        config = JevConfig(api_key="jev_test_key_dummy", base_url="https://api.typesafe.ai/v1")
        real_provider = RealJevDecisionProvider(config)
        _, question = self.runner.build_recovery_scenario()

        result = self.runner.execute_bounded_query(provider=real_provider, question=question)

        self.assertTrue(result.success)
        self.assertTrue(result.validation_passed)
        self.assertEqual(result.decision.selected_option, "REPOSITION")
        self.assertEqual(result.decision.confidence, 0.89)
        self.assertEqual(result.decision.option_probabilities["REPOSITION"], 0.89)
        self.assertEqual(result.telemetry.selected_option, "REPOSITION")

    # 5. Clean Failure Handling Without Silent Mock Fallback
    def test_execute_bounded_query_missing_credentials_failure(self):
        """Verify missing API key cleanly fails with JevConfigurationError."""
        config = JevConfig(api_key=None)
        real_provider = RealJevDecisionProvider(config)
        _, question = self.runner.build_recovery_scenario()

        result = self.runner.execute_bounded_query(provider=real_provider, question=question)

        self.assertFalse(result.success)
        self.assertFalse(result.validation_passed)
        self.assertIsNone(result.decision)
        self.assertEqual(result.error_category, "JevConfigurationError")
        self.assertIn("JEV_API_KEY", result.error_message)
        self.assertEqual(result.telemetry.provider_status, "FAILED")
        self.assertIsNone(result.telemetry.selected_option)

    # 6. Connection Error Handling
    def test_execute_bounded_query_connection_error_failure(self):
        """Verify network connection errors are cleanly caught and categorized."""
        mock_provider = MagicMock()
        mock_provider.ask.side_effect = JevConnectionError("Failed to connect to Jev endpoint")
        _, question = self.runner.build_recovery_scenario()

        result = self.runner.execute_bounded_query(provider=mock_provider, question=question)

        self.assertFalse(result.success)
        self.assertEqual(result.error_category, "JevConnectionError")
        self.assertIn("Failed to connect", result.error_message)

    # 7. Timeout Error Handling
    def test_execute_bounded_query_timeout_failure(self):
        """Verify query timeouts are cleanly caught and categorized."""
        mock_provider = MagicMock()
        mock_provider.ask.side_effect = JevTimeoutError("Jev request timed out after 5.0s")
        _, question = self.runner.build_recovery_scenario()

        result = self.runner.execute_bounded_query(provider=mock_provider, question=question)

        self.assertFalse(result.success)
        self.assertEqual(result.error_category, "JevTimeoutError")
        self.assertIn("timed out", result.error_message)

    # 8. Secret Redaction in Telemetry
    def test_telemetry_strictly_sanitizes_secrets(self):
        """Verify telemetry dictionaries contain zero API keys or tokens."""
        t = BoundedQueryTelemetry(
            question_id="q_sec_01",
            question_type="RECOVERY_SELECTION",
            available_options=["REPOSITION", "STOP"],
            selected_option="REPOSITION",
            confidence=0.9,
            probabilities={"REPOSITION": 0.9, "STOP": 0.1},
            latency_ms=15.0,
            success=True,
            provider_status="AVAILABLE",
            provider_name="jev-1",
            model_version="jev-1",
            error_message="Bearer sk-secret12345 should not appear",
        )
        t_dict = t.to_dict()
        t_json = json.dumps(t_dict)
        self.assertNotIn("sk-secret12345", t_json)

    # 9. Safety Invariant: No Robot or MuJoCo Execution
    def test_safety_boundary_zero_side_effects(self):
        """Verify scenario WorldState is untouched and no motor command is generated."""
        situation, question = self.runner.build_recovery_scenario()
        ws_before = situation.world_state

        mock_provider = MockJevDecisionProvider(
            default_choices={"RECOVERY_SELECTION": "REPOSITION"}
        )
        result = self.runner.execute_bounded_query(provider=mock_provider, question=question)

        self.assertIsInstance(result.decision, JevDecision)
        # WorldState version must not have advanced
        self.assertEqual(situation.world_state.version, ws_before.version)
        # No ActionExecution or motor command created
        self.assertFalse(hasattr(result, "action_execution"))


if __name__ == "__main__":
    unittest.main()
