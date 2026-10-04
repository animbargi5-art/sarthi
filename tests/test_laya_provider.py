"""
SĀRTHI V3-4B — Local Laya Decision Provider Unit Test Suite.
Validates LayaDecisionProvider against exact observed Laya 0.3.25 /v1/systemone schemas,
bounded option validation, confidence and answer_confidence separation,
error categorization, secret sanitization, and offline execution independence.

Non-Negotiable Verification:
1. All tests must run 100% offline (mocked urllib calls). Laya server NOT required.
2. AI models propose or select bounded semantic candidates; deterministic software validates
   physical feasibility; only validated actions reach the robot controller.
3. answer_confidence is preserved separately and NOT converted into confidence.
4. Output ends strictly at JevDecision.
"""

import json
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from backend.app.decision_engine.models import (
    ActionType,
    ActiveConstraint,
    LastActionOutcome,
    LastActionStatus,
    Point3D,
    RobotState,
    TargetZone,
    WorldObject,
    WorldState,
)
from backend.app.model.laya_config import LayaConfig
from backend.app.model.laya_provider import LayaDecisionProvider
from backend.app.model.mock_jev_provider import MockJevDecisionProvider
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


def _create_mock_http_response(body_dict: dict, status: int = 200):
    """Helper to mock urllib.request.urlopen context manager."""
    mock_resp = MagicMock()
    mock_resp.status = status
    json_bytes = json.dumps(body_dict).encode("utf-8")
    mock_resp.read.return_value = json_bytes
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False
    return mock_resp


class TestLayaDecisionProvider(unittest.TestCase):
    """Test suite for local LayaDecisionProvider adapter."""

    def setUp(self):
        """Set up canonical question and provider fixtures."""
        self.config = LayaConfig(
            base_url="http://127.0.0.1:8000",
            model="typed-decisions",
            timeout=5.0,
        )
        self.provider = LayaDecisionProvider(self.config)

        # Canonical DecisionContext
        self.context = DecisionContext(
            context_id="ctx_recovery_test_01",
            task_objective="Move the red object to the blue target.",
            robot_state_summary={"holding_object_id": "red_object_01", "gripper_open": False},
            active_constraints=["BlockedPath: obstacle blocking_barrier_01 intersects transit corridor"],
            previous_action_outcome="MOVE:FAILURE",
            candidate_actions=["REPOSITION", "STOP"],
            disturbance_info={"type": "PATH_BLOCKED", "obstacle_id": "blocking_barrier_01"},
        )

        # Canonical DecisionQuestion
        self.question = DecisionQuestion(
            question_id="recovery",
            question_type=DecisionQuestionType.RECOVERY_SELECTION,
            question_text="Which recovery action should be considered?",
            available_options=["REPOSITION", "STOP"],
            context=self.context,
        )

        # The EXACT observed response from Laya 0.3.25
        self.observed_laya_response = {
            "model": "laya-rl-agent",
            "answers": {
                "recovery": {
                    "type": "choice",
                    "choice": "REPOSITION",
                    "probabilities": {
                        "REPOSITION": 0.6565,
                        "STOP": 0.3435,
                    },
                    "confidence": 0.0719,
                    "answer_confidence": 0.6565,
                    "action": {
                        "act_probability": 1.0,
                    },
                }
            },
            "usage": {
                "input_tokens": 79,
                "output_tokens": 0,
                "state_tokens": 35,
                "state_tokens_dropped": 0,
                "truncated": False,
            },
            "routing": {
                "model": "typed-decisions",
                "repo": "convaiinnovations/laya/typed-decisions",
                "reason": "explicit model='typed-decisions'",
            },
        }

    # 1. Successful parsing of the exact observed response
    @patch("urllib.request.urlopen")
    def test_01_successful_parsing_exact_observed_response(self, mock_urlopen):
        """Verify successful parsing of the exact observed Laya 0.3.25 response payload."""
        mock_urlopen.return_value = _create_mock_http_response(self.observed_laya_response)

        decision = self.provider.ask(self.question)
        self.assertIsInstance(decision, JevDecision)
        self.assertEqual(decision.provider, "laya")
        self.assertEqual(decision.model_version, "laya-rl-agent")
        self.assertEqual(decision.request_id, "recovery")

    # 2. REPOSITION is returned correctly
    @patch("urllib.request.urlopen")
    def test_02_reposition_returned_correctly(self, mock_urlopen):
        """Verify that 'REPOSITION' is accurately extracted as selected_option."""
        mock_urlopen.return_value = _create_mock_http_response(self.observed_laya_response)

        decision = self.provider.ask(self.question)
        self.assertEqual(decision.selected_option, "REPOSITION")

    # 3. Probabilities are preserved exactly
    @patch("urllib.request.urlopen")
    def test_03_probabilities_preserved_exactly(self, mock_urlopen):
        """Verify probabilities {REPOSITION: 0.6565, STOP: 0.3435} are preserved exactly."""
        mock_urlopen.return_value = _create_mock_http_response(self.observed_laya_response)

        decision = self.provider.ask(self.question)
        self.assertEqual(decision.option_probabilities, {"REPOSITION": 0.6565, "STOP": 0.3435})

    # 4. confidence is preserved correctly
    @patch("urllib.request.urlopen")
    def test_04_confidence_preserved_correctly(self, mock_urlopen):
        """Verify that confidence (0.0719) is accurately assigned to confidence."""
        mock_urlopen.return_value = _create_mock_http_response(self.observed_laya_response)

        decision = self.provider.ask(self.question)
        self.assertEqual(decision.confidence, 0.0719)

    # 5. answer_confidence is not incorrectly mapped into confidence
    @patch("urllib.request.urlopen")
    def test_05_answer_confidence_separated_from_confidence(self, mock_urlopen):
        """Verify answer_confidence (0.6565) is NOT silently mapped into confidence (0.0719)."""
        mock_urlopen.return_value = _create_mock_http_response(self.observed_laya_response)

        decision = self.provider.ask(self.question)
        self.assertEqual(decision.confidence, 0.0719)
        self.assertEqual(decision.answer_confidence, 0.6565)
        self.assertNotEqual(decision.confidence, decision.answer_confidence)

    # 6. Unknown choice is rejected
    @patch("urllib.request.urlopen")
    def test_06_unknown_choice_rejected(self, mock_urlopen):
        """Verify that an action outside available_options raises JevValidationError."""
        bad_response = {
            "answers": {
                "recovery": {
                    "type": "choice",
                    "choice": "UNKNOWN_ACTION_PRIMITIVE",
                    "confidence": 0.9,
                }
            }
        }
        mock_urlopen.return_value = _create_mock_http_response(bad_response)

        with self.assertRaises(JevValidationError) as ctx:
            self.provider.ask(self.question)
        self.assertIn("not in question's available options", str(ctx.exception))

    # 7. Missing answer is rejected
    @patch("urllib.request.urlopen")
    def test_07_missing_answer_rejected(self, mock_urlopen):
        """Verify that missing question answer raises JevResponseParsingError."""
        missing_response = {"answers": {}}
        mock_urlopen.return_value = _create_mock_http_response(missing_response)

        with self.assertRaises(JevResponseParsingError) as ctx:
            self.provider.ask(self.question)
        self.assertIn("missing question result", str(ctx.exception).lower())

    # 8. Invalid probability is rejected
    @patch("urllib.request.urlopen")
    def test_08_invalid_probability_rejected(self, mock_urlopen):
        """Verify probability out of [0.0, 1.0] raises JevValidationError."""
        invalid_prob_response = {
            "answers": {
                "recovery": {
                    "type": "choice",
                    "choice": "REPOSITION",
                    "probabilities": {"REPOSITION": 1.5, "STOP": -0.5},
                }
            }
        }
        mock_urlopen.return_value = _create_mock_http_response(invalid_prob_response)

        with self.assertRaises(JevValidationError) as ctx:
            self.provider.ask(self.question)
        self.assertIn("outside [0.0, 1.0]", str(ctx.exception))

    # 9. Invalid confidence is rejected
    @patch("urllib.request.urlopen")
    def test_09_invalid_confidence_rejected(self, mock_urlopen):
        """Verify confidence out of [0.0, 1.0] raises JevValidationError."""
        invalid_conf_response = {
            "answers": {
                "recovery": {
                    "type": "choice",
                    "choice": "REPOSITION",
                    "confidence": 2.5,
                }
            }
        }
        mock_urlopen.return_value = _create_mock_http_response(invalid_conf_response)

        with self.assertRaises(JevValidationError) as ctx:
            self.provider.ask(self.question)
        self.assertIn("outside [0.0, 1.0]", str(ctx.exception))

    # 10. Timeout handling
    @patch("urllib.request.urlopen")
    def test_10_timeout_handling(self, mock_urlopen):
        """Verify query exceeding timeout raises JevTimeoutError."""
        mock_urlopen.side_effect = TimeoutError("Request timed out")

        with self.assertRaises(JevTimeoutError) as ctx:
            self.provider.ask(self.question)
        self.assertIn("timed out", str(ctx.exception).lower())

    # 11. Connection failure handling
    @patch("urllib.request.urlopen")
    def test_11_connection_failure_handling(self, mock_urlopen):
        """Verify network connection refusal raises JevConnectionError."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        with self.assertRaises(JevConnectionError) as ctx:
            self.provider.ask(self.question)
        self.assertIn("failed to connect", str(ctx.exception).lower())

    # 12. Secret-safe error messages
    def test_12_secret_safe_errors(self):
        """Verify _sanitize strips Bearer tokens, secrets, and credentials."""
        raw_msg = "Error with Bearer sk-live12345 and API_KEY=supersecretpass"
        sanitized = self.provider._sanitize(raw_msg)
        self.assertNotIn("sk-live12345", sanitized)
        self.assertNotIn("supersecretpass", sanitized)

    # 13. Bounded option enforcement
    def test_13_bounded_option_enforcement(self):
        """Verify payload criteria strictly reflects question available options."""
        payload = self.provider._build_bounded_payload(self.question)
        q_dict = payload["questions"]["recovery"]
        criteria = q_dict["criteria"]
        self.assertEqual(set(criteria.keys()), {"REPOSITION", "STOP"})
        self.assertNotIn("FREE_FORM_COMMAND", criteria)

    # 14. Mock provider remains functional
    def test_14_mock_provider_remains_functional(self):
        """Verify existing MockJevDecisionProvider remains intact and functional."""
        mock = MockJevDecisionProvider(
            default_choices={"RECOVERY_SELECTION": "STOP"},
            confidence=0.88,
        )
        decision = mock.ask(self.question)
        self.assertIsInstance(decision, JevDecision)
        self.assertEqual(decision.selected_option, "STOP")
        self.assertEqual(decision.confidence, 0.88)


if __name__ == "__main__":
    unittest.main()
