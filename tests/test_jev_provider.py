"""
SĀRTHI V3 — Real Jev Decision Provider Unit Test Suite.
Validates the RealJevDecisionProvider adapter, bounded question request construction,
response parsing (canonical & flat), bounded option validation, error handling,
timeout enforcement, secret sanitization, and non-authoritative boundary guarantees.

Non-Negotiable Architecture Invariant:
Jev proposes bounded candidate decisions; SĀRTHI Decision Engine remains the sole physical authority.
Jev must never directly command robot motors or bypass safety validation.
"""

import asyncio
import io
import json
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from backend.app.decision_engine.models import (
    ActionType,
    ActiveConstraint,
    CandidateAction,
    Decision,
    LastActionOutcome,
    LastActionStatus,
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


def _create_mock_http_response(body_dict: dict, status: int = 200):
    """Helper to mock urllib.request.urlopen context manager."""
    mock_resp = MagicMock()
    mock_resp.status = status
    json_bytes = json.dumps(body_dict).encode("utf-8")
    mock_resp.read.return_value = json_bytes
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False
    return mock_resp


class TestJevProvider(unittest.TestCase):
    """Test suite validating RealJevDecisionProvider adapter."""

    def setUp(self):
        """Set up canonical state and bounded question fixtures."""
        self.config = JevConfig(
            api_key="jev_test_key_abc123",
            base_url="https://api.typesafe.ai/v1",
            model="jev-1",
            timeout=2.0,
        )
        self.provider = RealJevDecisionProvider(self.config)

        # Build canonical situation & context
        ws = WorldState(
            version=4,
            timestamp_ns=2000,
            robot=RobotState(
                position=Point3D(x=0.55, y=0.0, z=0.52),
                gripper_open=False,
                holding_object_id="red_object_01",
                payload_mass_kg=0.1,
            ),
            objects=[
                WorldObject(
                    id="red_object_01",
                    name="red_object",
                    position=Point3D(x=0.55, y=0.0, z=0.52),
                    is_target=True,
                )
            ],
            target=TargetZone(
                id="blue_target",
                position=Point3D(x=0.40, y=-0.19, z=0.12),
                tolerance_radius_m=0.06,
            ),
            active_constraints=[
                ActiveConstraint(
                    constraint_id="c_blocked_path",
                    description="PATH_BLOCKED: blocking_barrier_01 intersects transit corridor",
                    required_clearance_m=0.082,
                )
            ],
            last_action_outcome=LastActionOutcome(
                action_type=ActionType.MOVE,
                status=LastActionStatus.FAILURE,
                error_message="BlockedPath: Clearance 0.066m < required 0.082m",
            ),
        )

        situation = PhysicalSituation(
            situation_id="sit_recovery_01",
            world_state=ws,
            active_constraints=ws.active_constraints,
            last_action_outcome=ws.last_action_outcome,
            task_context={"task_phase": "DISTURBANCE_RECOVERY"},
        )

        self.context = DecisionContext.from_situation(
            context_id="ctx_recovery_01",
            situation=situation,
            candidate_actions=["REPOSITION", "REPLAN", "REGRASP", "STOP"],
            task_objective="PICK_AND_PLACE",
            disturbance_info={"anomaly": "PATH_BLOCKED", "obstacle_id": "blocking_barrier_01"},
        )

        self.question = DecisionQuestion(
            question_id="q_recovery_01",
            question_type=DecisionQuestionType.RECOVERY_SELECTION,
            question_text="Which recovery strategy should be considered?",
            available_options=["REPOSITION", "REPLAN", "REGRASP", "STOP"],
            context=self.context,
        )

    # 1. Provider Availability & Configuration Check
    def test_provider_unavailable_when_api_key_missing(self):
        """Verify provider correctly reports is_available=False when API key is missing."""
        empty_provider = RealJevDecisionProvider(JevConfig(api_key=None))
        self.assertFalse(empty_provider.is_available())

        with self.assertRaises(JevConfigurationError) as ctx:
            empty_provider.ask(self.question)
        self.assertIn("JEV_API_KEY", str(ctx.exception))

    # 2. Successful Response Parsing — Canonical TypeSafe Format
    @patch("urllib.request.urlopen")
    def test_successful_canonical_response_parsing(self, mock_urlopen):
        """Verify parsing of canonical TypeSafe AI response structure."""
        canonical_response = {
            "questions": {
                "q_recovery_01": {
                    "choice": "REPOSITION",
                    "confidence": 0.94,
                    "probabilities": {
                        "REPOSITION": 0.94,
                        "REPLAN": 0.03,
                        "REGRASP": 0.02,
                        "STOP": 0.01,
                    },
                    "rationale": "Obstacle intersects path; vertical repositioning provides 0.35m clearance.",
                }
            }
        }
        mock_urlopen.return_value = _create_mock_http_response(canonical_response)

        decision = self.provider.ask(self.question)

        self.assertIsInstance(decision, JevDecision)
        self.assertEqual(decision.selected_option, "REPOSITION")
        self.assertEqual(decision.confidence, 0.94)
        self.assertEqual(decision.option_probabilities["REPOSITION"], 0.94)
        self.assertEqual(decision.provider, "jev")
        self.assertEqual(decision.model_version, "jev-1")
        self.assertGreater(decision.latency_ms, 0.0)
        self.assertEqual(decision.request_id, "q_recovery_01")
        self.assertIn("vertical repositioning", decision.rationale)

    # 3. Successful Response Parsing — Flattened Gateway Format
    @patch("urllib.request.urlopen")
    def test_successful_flat_response_parsing(self, mock_urlopen):
        """Verify parsing of flat gateway/proxy response structure."""
        flat_response = {
            "choice": "REPOSITION",
            "confidence": 0.89,
            "probabilities": {"REPOSITION": 0.89, "REPLAN": 0.11},
        }
        mock_urlopen.return_value = _create_mock_http_response(flat_response)

        decision = self.provider.ask(self.question)
        self.assertIsInstance(decision, JevDecision)
        self.assertEqual(decision.selected_option, "REPOSITION")
        self.assertEqual(decision.confidence, 0.89)

    # 4. Bounded Option Validation Rejection
    @patch("urllib.request.urlopen")
    def test_bounded_option_validation_rejects_hallucinated_choice(self, mock_urlopen):
        """Verify provider rejects any choice that is not in the question's available options."""
        invalid_choice_response = {
            "questions": {
                "q_recovery_01": {
                    "choice": "UNSAFE_WARP_ACTION",  # Not in ["REPOSITION", "REPLAN", "REGRASP", "STOP"]
                    "confidence": 0.99,
                }
            }
        }
        mock_urlopen.return_value = _create_mock_http_response(invalid_choice_response)

        with self.assertRaises(JevValidationError) as ctx:
            self.provider.ask(self.question)
        self.assertIn("UNSAFE_WARP_ACTION", str(ctx.exception))
        self.assertIn("available options", str(ctx.exception))

    # 5. Malformed Response Rejection — Non-JSON
    @patch("urllib.request.urlopen")
    def test_malformed_response_non_json_rejected(self, mock_urlopen):
        """Verify non-JSON response raises JevResponseParsingError."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"<html>502 Bad Gateway</html>"
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = False
        mock_urlopen.return_value = mock_resp

        with self.assertRaises(JevResponseParsingError):
            self.provider.ask(self.question)

    # 6. Malformed Response Rejection — Missing Required Keys
    @patch("urllib.request.urlopen")
    def test_malformed_response_missing_choice_rejected(self, mock_urlopen):
        """Verify response missing choice/selected_option raises JevResponseParsingError."""
        bad_response = {
            "questions": {
                "q_recovery_01": {
                    "confidence": 0.85
                    # Missing 'choice'
                }
            }
        }
        mock_urlopen.return_value = _create_mock_http_response(bad_response)

        with self.assertRaises(JevResponseParsingError):
            self.provider.ask(self.question)

    # 7. Invalid Confidence & Probability Values Out of Bounds
    @patch("urllib.request.urlopen")
    def test_invalid_confidence_out_of_bounds_rejected(self, mock_urlopen):
        """Verify confidence > 1.0 is rejected by validation."""
        out_of_bounds_conf = {
            "questions": {
                "q_recovery_01": {
                    "choice": "REPOSITION",
                    "confidence": 1.25,  # Invalid
                }
            }
        }
        mock_urlopen.return_value = _create_mock_http_response(out_of_bounds_conf)

        with self.assertRaises(JevValidationError) as ctx:
            self.provider.ask(self.question)
        self.assertIn("Confidence score", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_invalid_probability_out_of_bounds_rejected(self, mock_urlopen):
        """Verify negative probability is rejected by validation."""
        out_of_bounds_prob = {
            "questions": {
                "q_recovery_01": {
                    "choice": "REPOSITION",
                    "probabilities": {"REPOSITION": -0.1, "STOP": 0.5},
                }
            }
        }
        mock_urlopen.return_value = _create_mock_http_response(out_of_bounds_prob)

        with self.assertRaises(JevValidationError) as ctx:
            self.provider.ask(self.question)
        self.assertIn("Probability", str(ctx.exception))

    # 8. Timeout Handling
    @patch("urllib.request.urlopen")
    def test_timeout_handling(self, mock_urlopen):
        """Verify timeout raises structured JevTimeoutError."""
        mock_urlopen.side_effect = TimeoutError("Connection timed out")

        with self.assertRaises(JevTimeoutError) as ctx:
            self.provider.ask(self.question)
        self.assertIn("timed out", str(ctx.exception).lower())

    # 9. Connection Failure Handling
    @patch("urllib.request.urlopen")
    def test_connection_failure_handling(self, mock_urlopen):
        """Verify HTTP 503 / network error raises JevConnectionError."""
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.typesafe.ai/v1/systemone",
            code=503,
            msg="Service Unavailable",
            hdrs={},
            fp=io.BytesIO(b"Service overloaded"),
        )

        with self.assertRaises(JevConnectionError) as ctx:
            self.provider.ask(self.question)
        self.assertIn("503", str(ctx.exception))

    # 10. Secret-Safe Error Serialization
    @patch("urllib.request.urlopen")
    def test_secret_safe_error_serialization(self, mock_urlopen):
        """Verify API keys and Bearer tokens are NEVER leaked in exception messages."""
        secret_key = "jev_live_top_secret_token_999888"
        secret_config = JevConfig(api_key=secret_key, timeout=1.0)
        provider_with_secret = RealJevDecisionProvider(secret_config)

        # Trigger HTTP error containing raw auth header
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.typesafe.ai/v1/systemone",
            code=401,
            msg="Unauthorized",
            hdrs={"Authorization": f"Bearer {secret_key}"},
            fp=io.BytesIO(f"Invalid key: {secret_key}".encode("utf-8")),
        )

        with self.assertRaises(JevConnectionError) as ctx:
            provider_with_secret.ask(self.question)

        error_message = str(ctx.exception)
        self.assertNotIn(secret_key, error_message)
        self.assertIn("[REDACTED_JEV_KEY]", error_message)

    # 11. Asynchronous Query Execution
    @patch("urllib.request.urlopen")
    def test_ask_async_execution(self, mock_urlopen):
        """Verify ask_async successfully evaluates bounded questions."""
        response = {
            "questions": {
                "q_recovery_01": {
                    "choice": "REPOSITION",
                    "confidence": 0.91,
                }
            }
        }
        mock_urlopen.return_value = _create_mock_http_response(response)

        async def run_test():
            return await self.provider.ask_async(self.question)

        decision = asyncio.run(run_test())
        self.assertIsInstance(decision, JevDecision)
        self.assertEqual(decision.selected_option, "REPOSITION")

    # 12. Mock Jev Provider Fallback Equivalence
    def test_mock_jev_provider_fallback_equivalence(self):
        """Verify MockJevDecisionProvider satisfies JevDecisionProvider interface seamlessly."""
        mock_prov = MockJevDecisionProvider(default_choices={"RECOVERY_SELECTION": "REPOSITION"})
        self.assertTrue(mock_prov.is_available())

        decision = mock_prov.ask(self.question)
        self.assertIsInstance(decision, JevDecision)
        self.assertEqual(decision.selected_option, "REPOSITION")
        self.assertEqual(decision.provider, "mock_jev")
        self.assertIn("REPOSITION", decision.option_probabilities)

    # 13. Architectural Invariant: Jev Output is Non-Authoritative
    def test_non_authoritative_enforcement(self):
        """
        Verify that JevDecision is strictly non-authoritative:
        - It is NOT an instance of Decision or CandidateAction
        - It does NOT contain actuation methods or trajectory waypoints
        - SĀRTHI Decision Engine remains the sole physical action authority
        """
        jd = JevDecision(
            selected_option="REPOSITION",
            confidence=0.95,
            rationale="Proposed recovery",
        )

        # Cannot be treated as physical Decision
        self.assertNotIsInstance(jd, Decision)
        self.assertNotIsInstance(jd, CandidateAction)
        self.assertFalse(hasattr(jd, "selected_action"))
        self.assertFalse(hasattr(jd, "candidate_evaluations"))
        self.assertFalse(hasattr(jd, "is_authorized"))
        self.assertFalse(hasattr(jd, "execute"))

        # Verify downstream path: JevDecision must be mapped to CandidateAction,
        # then evaluated by Decision Engine to become an authorized Decision.
        candidate = CandidateAction(
            action_id="act_reposition_candidate",
            action_type=ActionType.REPOSITION,
            parameters={"source": "JevDecision", "selected_option": jd.selected_option},
        )
        self.assertIsInstance(candidate, CandidateAction)
        self.assertEqual(candidate.action_type, ActionType.REPOSITION)


if __name__ == "__main__":
    unittest.main()
