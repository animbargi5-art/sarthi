"""
SĀRTHI V3 — Interface & Schema Formalization Unit Test Suite.
Validates Pydantic v2 data models, compact DecisionContext assembly, bounded DecisionQuestion
validation, non-authoritative JevDecision boundaries, JevDecisionProvider abstractions,
deterministic mock provider responses, latency instrumentation, physics validation schemas,
and secret-safe telemetry serialization.

Non-Negotiable Verification:
- AI models propose or select bounded semantic candidates; deterministic software validates
  physical feasibility; only validated actions reach the robot controller.
- Jev must never bypass deterministic safety validation.
- JevDecision is strictly non-authoritative and does not authorize physical execution.
"""

import asyncio
import json
import time
import unittest
from pydantic import ValidationError

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
from backend.app.model.v3_models import (
    DecisionContext,
    DecisionLatencyRecord,
    DecisionQuestion,
    DecisionQuestionType,
    HumanInstruction,
    JevDecision,
    PhysicalSituation,
    PhysicsValidationResult,
    RecoveryEvent,
    TelemetryEvent,
)
from backend.app.model.jev_provider import JevDecisionProvider
from backend.app.model.mock_jev_provider import MockJevDecisionProvider


class TestV3Interfaces(unittest.TestCase):
    """Test suite covering SĀRTHI V3 interface freezing and schema formalization."""

    def setUp(self):
        """Set up canonical state fixtures for V3 tests."""
        self.point_robot = Point3D(x=0.55, y=0.0, z=0.52)
        self.robot = RobotState(
            position=self.point_robot,
            gripper_open=True,
            holding_object_id=None,
            payload_mass_kg=0.0,
        )
        self.target_zone = TargetZone(
            id="blue_target",
            position=Point3D(x=0.40, y=-0.19, z=0.12),
            tolerance_radius_m=0.06,
        )
        self.object_red = WorldObject(
            id="red_object_01",
            name="red_object",
            position=Point3D(x=0.45, y=0.10, z=0.12),
            bounding_radius_m=0.02,
            mass_kg=0.1,
            is_target=True,
        )
        self.world_state = WorldState(
            version=1,
            timestamp_ns=1000,
            robot=self.robot,
            objects=[self.object_red],
            target=self.target_zone,
            active_constraints=[
                ActiveConstraint(
                    constraint_id="c_clearance",
                    description="Keep clearance >= 0.082m",
                    required_clearance_m=0.082,
                )
            ],
            last_action_outcome=LastActionOutcome(
                action_type=ActionType.APPROACH,
                status=LastActionStatus.SUCCESS,
            ),
        )
        self.situation = PhysicalSituation(
            situation_id="sit_001",
            world_state=self.world_state,
            active_constraints=self.world_state.active_constraints,
            last_action_outcome=self.world_state.last_action_outcome,
            task_context={"task_phase": "POST_APPROACH"},
            timestamp_ns=1000,
        )

    # 1. HumanInstruction validation
    def test_human_instruction_validation(self):
        """Verify HumanInstruction enforces valid strings, immutability, and metadata."""
        inst = HumanInstruction(
            instruction="Move the red object to the blue target.",
            task_id="task_move_01",
            timestamp_ns=123456789,
            metadata={"operator": "lead_engineer", "priority": "high"},
        )
        self.assertEqual(inst.instruction, "Move the red object to the blue target.")
        self.assertEqual(inst.task_id, "task_move_01")
        self.assertEqual(inst.timestamp_ns, 123456789)
        self.assertEqual(inst.metadata["operator"], "lead_engineer")

        # Frozen immutability check
        with self.assertRaises(ValidationError):
            inst.instruction = "Change command."  # Cannot mutate frozen model

        # Empty instruction rejected
        with self.assertRaises(ValidationError):
            HumanInstruction(instruction="", task_id="t1")

        # Empty task_id rejected
        with self.assertRaises(ValidationError):
            HumanInstruction(instruction="Valid command", task_id="")

    # 2. DecisionContext serialization
    def test_decision_context_serialization(self):
        """Verify DecisionContext is compact, serializable, and built cleanly from PhysicalSituation."""
        ctx = DecisionContext.from_situation(
            context_id="ctx_001",
            situation=self.situation,
            candidate_actions=["APPROACH", "GRASP", "STOP"],
            task_objective="PICK_AND_PLACE",
        )

        self.assertEqual(ctx.context_id, "ctx_001")
        self.assertEqual(ctx.task_objective, "PICK_AND_PLACE")
        self.assertIn("position", ctx.robot_state_summary)
        self.assertTrue(ctx.robot_state_summary["gripper_open"])
        self.assertIsNotNone(ctx.target_object_summary)
        self.assertEqual(ctx.target_object_summary["id"], "red_object_01")
        self.assertEqual(ctx.destination_summary["id"], "blue_target")
        self.assertEqual(ctx.candidate_actions, ["APPROACH", "GRASP", "STOP"])

        # Serialization to JSON string and back
        json_str = ctx.model_dump_json()
        self.assertIsInstance(json_str, str)
        parsed_dict = json.loads(json_str)
        self.assertEqual(parsed_dict["context_id"], "ctx_001")
        self.assertIn("target_object_summary", parsed_dict)

        # Confirm compact footprint (no raw pointers or giant arrays)
        self.assertLess(len(json_str), 1500)

    # 3. DecisionQuestion allowed options
    def test_decision_question_allowed_options(self):
        """Verify DecisionQuestion requires at least 2 distinct discrete options and valid types."""
        ctx = DecisionContext.from_situation(
            context_id="ctx_002",
            situation=self.situation,
            candidate_actions=["REPOSITION", "STOP"],
        )

        # Valid question
        q = DecisionQuestion(
            question_id="q_001",
            question_type=DecisionQuestionType.RECOVERY_SELECTION,
            question_text="Which recovery strategy should be considered?",
            available_options=["REPOSITION", "REPLAN", "REGRASP", "STOP"],
            context=ctx,
        )
        self.assertEqual(q.question_id, "q_001")
        self.assertEqual(q.question_type, DecisionQuestionType.RECOVERY_SELECTION)
        self.assertEqual(len(q.available_options), 4)

        # Error on single option
        with self.assertRaises(ValidationError):
            DecisionQuestion(
                question_id="q_bad_1",
                question_type=DecisionQuestionType.ACTION_SELECTION,
                question_text="Only one choice?",
                available_options=["APPROACH"],
                context=ctx,
            )

        # Error on empty options
        with self.assertRaises(ValidationError):
            DecisionQuestion(
                question_id="q_bad_2",
                question_type=DecisionQuestionType.ACTION_SELECTION,
                question_text="No choices?",
                available_options=[],
                context=ctx,
            )

        # Error on duplicate options
        with self.assertRaises(ValidationError):
            DecisionQuestion(
                question_id="q_bad_3",
                question_type=DecisionQuestionType.ACTION_SELECTION,
                question_text="Duplicate choices?",
                available_options=["APPROACH", "APPROACH"],
                context=ctx,
            )

    # 4. JevDecision schema validation
    def test_jev_decision_schema_validation(self):
        """Verify JevDecision validates options, probability distributions, confidence, and latency."""
        jd = JevDecision(
            selected_option="REPOSITION",
            option_probabilities={"REPOSITION": 0.92, "REPLAN": 0.05, "STOP": 0.03},
            confidence=0.92,
            rationale="Obstacle blocks nominal corridor; vertical reposition provides safe clearance.",
            provider="jev",
            model_version="jev-fast-v1",
            latency_ms=8.5,
            timestamp_ns=1700000000,
            request_id="req_123",
        )
        self.assertEqual(jd.selected_option, "REPOSITION")
        self.assertEqual(jd.confidence, 0.92)
        self.assertEqual(jd.latency_ms, 8.5)
        self.assertEqual(jd.option_probabilities["REPOSITION"], 0.92)

        # Confidence out of bounds (> 1.0)
        with self.assertRaises(ValidationError):
            JevDecision(selected_option="REPOSITION", confidence=1.5)

        # Confidence out of bounds (< 0.0)
        with self.assertRaises(ValidationError):
            JevDecision(selected_option="REPOSITION", confidence=-0.1)

        # Latency out of bounds (< 0.0)
        with self.assertRaises(ValidationError):
            JevDecision(selected_option="REPOSITION", latency_ms=-5.0)

    # 5. JevDecision cannot directly execute an action
    def test_jev_decision_cannot_directly_execute_action(self):
        """
        Verify structural and architectural separation:
        JevDecision is NOT an authoritative Decision and cannot execute or authorize physical actuation.
        """
        jd = JevDecision(
            selected_option="MOVE",
            confidence=0.99,
            rationale="Direct line of sight",
        )

        # Architectural type invariants
        self.assertNotIsInstance(jd, Decision)
        self.assertNotIsInstance(jd, CandidateAction)
        self.assertFalse(hasattr(jd, "selected_action"))
        self.assertFalse(hasattr(jd, "candidate_evaluations"))
        self.assertFalse(hasattr(jd, "execute"))
        self.assertFalse(hasattr(jd, "authorize"))

        # Verify that only a Decision Engine Decision object contains authoritative authorization
        authoritative_decision = Decision(
            decision_id="dec_001",
            world_state_version=1,
            selected_action=CandidateAction(
                action_id="act_001",
                action_type=ActionType.APPROACH,
            ),
            candidate_evaluations=[],
        )
        self.assertIsInstance(authoritative_decision, Decision)
        self.assertNotEqual(type(jd), type(authoritative_decision))

    # 6. Provider interface behavior
    def test_provider_interface_behavior(self):
        """Verify JevDecisionProvider is an abstract base class that cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            JevDecisionProvider()  # Cannot instantiate abstract class

        # Concrete subclass must implement ask, ask_async, is_available
        class IncompleteProvider(JevDecisionProvider):
            def ask(self, question: DecisionQuestion) -> JevDecision:
                pass

        with self.assertRaises(TypeError):
            IncompleteProvider()  # Missing ask_async and is_available

    # 7. Mock provider deterministic response
    def test_mock_provider_deterministic_response(self):
        """Verify MockJevDecisionProvider produces deterministic, valid JevDecision objects."""
        ctx = DecisionContext.from_situation(
            context_id="ctx_mock",
            situation=self.situation,
            candidate_actions=["REPOSITION", "REPLAN", "STOP"],
        )
        q = DecisionQuestion(
            question_id="q_mock_1",
            question_type=DecisionQuestionType.RECOVERY_SELECTION,
            question_text="Which recovery strategy should be considered?",
            available_options=["REPOSITION", "REPLAN", "REGRASP", "STOP"],
            context=ctx,
        )

        provider = MockJevDecisionProvider()
        self.assertTrue(provider.is_available())

        # Synchronous query
        decision = provider.ask(q)
        self.assertIsInstance(decision, JevDecision)
        self.assertEqual(decision.selected_option, "REPOSITION")
        self.assertGreaterEqual(decision.confidence, 0.9)
        self.assertEqual(decision.provider, "mock_jev")
        self.assertIn("REPOSITION", decision.option_probabilities)
        self.assertEqual(len(provider.query_history), 1)

        # Asynchronous query
        async def run_async():
            return await provider.ask_async(q)

        async_decision = asyncio.run(run_async())
        self.assertIsInstance(async_decision, JevDecision)
        self.assertEqual(async_decision.selected_option, "REPOSITION")
        self.assertEqual(len(provider.query_history), 2)

    # 8. DecisionLatencyRecord
    def test_decision_latency_record(self):
        """Verify DecisionLatencyRecord validates timing markers and allows partial/optional values."""
        record = DecisionLatencyRecord(
            cycle_id="cycle_001",
            tau_nemotron=280.5,
            tau_context=1.2,
            tau_jev=6.8,
            tau_validation=0.4,
            tau_ik=2.1,
            tau_sim=150.0,
            tau_verify=0.8,
            total_cycle_ms=441.8,
            timestamp_ns=time.time_ns(),
        )
        self.assertEqual(record.cycle_id, "cycle_001")
        self.assertEqual(record.tau_nemotron, 280.5)
        self.assertEqual(record.tau_jev, 6.8)
        self.assertEqual(record.total_cycle_ms, 441.8)

        # Partial values allowed without fabrication
        partial_record = DecisionLatencyRecord(cycle_id="cycle_partial")
        self.assertIsNone(partial_record.tau_nemotron)
        self.assertIsNone(partial_record.tau_jev)
        self.assertEqual(partial_record.cycle_id, "cycle_partial")

        # Negative latencies rejected
        with self.assertRaises(ValidationError):
            DecisionLatencyRecord(cycle_id="cycle_bad", tau_jev=-1.0)

    # 9. PhysicsValidationResult
    def test_physics_validation_result(self):
        """Verify PhysicsValidationResult captures test dimensions, metrics, and thresholds."""
        res = PhysicsValidationResult(
            validation_id="PV-01",
            test_name="Timestep Sensitivity",
            category="NUMERICAL_STABILITY",
            passed=True,
            measured_metrics={"max_trajectory_deviation_m": 0.003},
            configured_thresholds={"max_allowed_deviation_m": 0.010},
            violations=[],
            seed_metadata={"seed": 42, "timesteps_tested": [0.001, 0.002, 0.005]},
            timestamp_ns=time.time_ns(),
        )
        self.assertEqual(res.validation_id, "PV-01")
        self.assertTrue(res.passed)
        self.assertEqual(res.measured_metrics["max_trajectory_deviation_m"], 0.003)
        self.assertEqual(len(res.violations), 0)

        # Failed result with violations
        res_fail = PhysicsValidationResult(
            validation_id="PV-07",
            test_name="Collision/Clearance Validation",
            category="SAFETY_BOUNDS",
            passed=False,
            measured_metrics={"min_clearance_m": 0.045},
            configured_thresholds={"required_clearance_m": 0.082},
            violations=["Path approached obstacle at 0.045m, below required 0.082m buffer"],
        )
        self.assertFalse(res_fail.passed)
        self.assertEqual(len(res_fail.violations), 1)

    # 10. RecoveryEvent
    def test_recovery_event(self):
        """Verify RecoveryEvent records disturbance and recovery audit details."""
        rec = RecoveryEvent(
            event_id="rec_001",
            disturbance_type="PATH_BLOCKED",
            rejected_action="act_move_to_target_zone",
            selected_recovery="REPOSITION",
            reason="Obstacle detected in corridor (clearance 0.066m < required 0.082m)",
            before_state_version=3,
            after_state_version=4,
            verification_result="REPOSITION_VERIFIED",
            recovery_count=1,
            timestamp_ns=time.time_ns(),
        )
        self.assertEqual(rec.event_id, "rec_001")
        self.assertEqual(rec.disturbance_type, "PATH_BLOCKED")
        self.assertEqual(rec.selected_recovery, "REPOSITION")
        self.assertEqual(rec.recovery_count, 1)

        # Recovery count < 1 rejected
        with self.assertRaises(ValidationError):
            RecoveryEvent(
                event_id="rec_bad",
                disturbance_type="PATH_BLOCKED",
                rejected_action="MOVE",
                selected_recovery="REPOSITION",
                reason="test",
                recovery_count=0,
            )

    # 11. TelemetryEvent
    def test_telemetry_event(self):
        """Verify TelemetryEvent records structured telemetry data."""
        te = TelemetryEvent(
            event_id="tel_001",
            event_type="DECISION",
            timestamp_ns=time.time_ns(),
            task_id="task_001",
            stage="RECOVERY_SELECTION",
            success=True,
            metadata={"candidate": "REPOSITION", "clearance_altitude_m": 0.351},
            latency_ms=4.2,
        )
        self.assertEqual(te.event_id, "tel_001")
        self.assertEqual(te.stage, "RECOVERY_SELECTION")
        self.assertTrue(te.success)
        self.assertEqual(te.latency_ms, 4.2)
        self.assertEqual(te.metadata["candidate"], "REPOSITION")

    # 12. Secret-safe serialization
    def test_secret_safe_serialization(self):
        """Verify TelemetryEvent automatically sanitizes API keys and authorization headers."""
        sensitive_metadata = {
            "normal_metric": 42.0,
            "nebius_api_key": "nbf_live_super_secret_token_12345",
            "auth_header": "Bearer sk-proj-9876543210secret",
            "api_token": "some-raw-token",
            "nested_info": "safe_information",
        }

        te = TelemetryEvent(
            event_id="tel_leak_test",
            event_type="AUDIT",
            timestamp_ns=time.time_ns(),
            stage="SECURITY_CHECK",
            metadata=sensitive_metadata,
        )

        # All sensitive fields must be redacted
        self.assertEqual(te.metadata["normal_metric"], 42.0)
        self.assertEqual(te.metadata["nebius_api_key"], "[REDACTED]")
        self.assertEqual(te.metadata["auth_header"], "[REDACTED]")
        self.assertEqual(te.metadata["api_token"], "[REDACTED]")
        self.assertEqual(te.metadata["nested_info"], "safe_information")

        # Serialized JSON must contain zero sensitive token strings
        te_json = te.model_dump_json()
        self.assertNotIn("nbf_live_", te_json)
        self.assertNotIn("sk-proj-", te_json)
        self.assertNotIn("Bearer ", te_json)


if __name__ == "__main__":
    unittest.main()
