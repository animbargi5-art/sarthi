"""
SĀRTHI V3-6 — End-to-End Autonomous Orchestrator Unit Test Suite.
Validates the complete cognitive-to-physical pipeline:
  Human Instruction
  -> TaskUnderstandingService (via ModelProvider)
  -> DecisionContextBuilder
  -> Bounded DecisionQuestion
  -> Fast Laya Decision Proposal (via JevDecisionProvider)
  -> CandidateActionInjector
  -> SarthiDecisionEngine Deterministic Physical Validation
  -> MuJoCo Closed-Loop Execution & Placement Verification
  -> Dynamic Obstacle Disturbance Recovery
  -> Secret-Free Audit Telemetry

Non-Negotiable Invariants:
1. Live network / external services must NEVER be required for unit test execution.
2. AI models propose or select bounded semantic candidates; deterministic software
   validates physical feasibility; only validated actions reach execution.
3. High Laya confidence NEVER overrides deterministic physical validation.
4. No direct AI -> MuJoCo or motor command path exists.
5. All secrets are scrubbed from telemetry and error traces.
"""

import json
import math
import os
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
from backend.app.model.mock_jev_provider import MockJevDecisionProvider
from backend.app.model.mock_provider import MockModelProvider
from backend.app.model.models import TaskUnderstanding
from backend.app.model.task_understanding import TaskUnderstandingService
from backend.app.model.nebius_provider import (
    NebiusNemotronProvider,
    NebiusParsingError,
    NebiusTimeoutError,
)
from backend.app.model.real_jev_provider import (
    JevConnectionError,
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
from backend.app.orchestration.loop import TaskRunStatus
from backend.app.orchestration.v3_orchestrator import (
    V3ExecutionTelemetry,
    V3OrchestrationService,
    V3StepTelemetry,
)
from simulation.adapters.base import SimulationAdapter
from simulation.mujoco_runtime import require_mujoco
from simulation.mujoco_runtime.runtime import SarthiMuJoCoRuntime


class MockLayaProvider:
    """Deterministic mock Laya provider for offline testing."""
    provider_name: str = "laya"
    model: str = "laya-rl-agent"

    def __init__(
        self,
        fixed_option: str = "REPOSITION",
        fixed_confidence: float = 0.88,
        provider_name: str = "laya",
        model_name: str = "laya-rl-agent",
    ) -> None:
        self.fixed_option = fixed_option
        self.fixed_confidence = fixed_confidence
        self.provider_name = provider_name
        self.model = model_name

    def decide(self, question: DecisionQuestion) -> JevDecision:
        opts = question.available_options
        sel = self.fixed_option
        n_rest = max(1, len(opts) - 1)
        probs = {
            opt: (self.fixed_confidence if opt == sel else (1.0 - self.fixed_confidence) / n_rest)
            for opt in opts
        }
        return JevDecision(
            selected_option=sel,
            confidence=self.fixed_confidence,
            answer_confidence=self.fixed_confidence,
            option_probabilities=probs,
            provider=self.provider_name,
            model_version=self.model,
            latency_ms=12.5,
        )

    def ask(self, question: DecisionQuestion) -> JevDecision:
        return self.decide(question)


class TestV3Orchestrator(unittest.TestCase):
    """Phase V3-6 End-to-End Orchestration Test Suite."""

    @classmethod
    def setUpClass(cls):
        require_mujoco()

    def setUp(self):
        """Initializes MuJoCo runtime, providers, and V3 orchestration service."""
        self.runtime = SarthiMuJoCoRuntime(
            config={"steps_per_action": 400, "convergence_tolerance": 0.03}
        )
        self.engine = SarthiDecisionEngine()
        self.context_builder = DecisionContextBuilder(coordinate_precision=3)
        self.candidate_injector = CandidateActionInjector()
        self.mock_model_provider = MockModelProvider()
        self.mock_laya_provider = MockLayaProvider(
            provider_name="laya",
            model_name="laya-rl-agent",
            fixed_option="REPOSITION",
            fixed_confidence=0.88,
        )

        self.service = V3OrchestrationService(
            adapter=self.runtime,
            engine=self.engine,
            context_builder=self.context_builder,
            candidate_injector=self.candidate_injector,
            model_provider=self.mock_model_provider,
            laya_provider=self.mock_laya_provider,
            max_steps=10,
        )
        self.instruction = "Move the red object to the blue target."

    # -----------------------------------------------------------------------
    # 1. Human instruction reaches Nemotron / task understanding
    # -----------------------------------------------------------------------
    def test_01_human_instruction_reaches_task_understanding(self):
        """1. Verify human natural language instruction is received by TaskUnderstandingService."""
        with patch.object(self.service.task_service, "understand", wraps=self.service.task_service.understand) as spy_understand:
            understanding = self.service.task_service.understand(self.instruction)
            spy_understand.assert_called_once_with(self.instruction)
            self.assertIsInstance(understanding, TaskUnderstanding)
            self.assertEqual(understanding.target_object, "red_object")
            self.assertEqual(understanding.target_location, "blue_target")

    # -----------------------------------------------------------------------
    # 2. Physical WorldState reaches DecisionContextBuilder
    # -----------------------------------------------------------------------
    def test_02_physical_world_state_reaches_decision_context_builder(self):
        """2. Verify canonical WorldState from MuJoCo is ingested by DecisionContextBuilder."""
        ws = self.runtime.get_world_state()
        context = self.service.context_builder.build_context(
            world_state=ws,
            task_objective="Move the red object to the blue target.",
            candidate_actions=["APPROACH", "STOP"],
        )

        self.assertIsInstance(context, DecisionContext)
        self.assertEqual(context.robot_state_summary["gripper_open"], ws.robot.gripper_open)
        self.assertAlmostEqual(context.robot_state_summary["position"][0], ws.robot.position.x, places=2)
        self.assertAlmostEqual(context.robot_state_summary["position"][1], ws.robot.position.y, places=2)

    # -----------------------------------------------------------------------
    # 3. DecisionQuestion contains bounded available options
    # -----------------------------------------------------------------------
    def test_03_decision_question_contains_bounded_available_options(self):
        """3. Verify constructed DecisionQuestion contains strictly bounded domain options."""
        ws = self.runtime.get_world_state()
        question = self.service.build_bounded_question(ws, step_number=1)

        self.assertIsInstance(question, DecisionQuestion)
        self.assertIsInstance(question.available_options, list)
        self.assertTrue(len(question.available_options) >= 2)
        # All options must be valid SĀRTHI ActionType values
        for opt in question.available_options:
            self.assertIn(opt, [a.value for a in ActionType])

    # -----------------------------------------------------------------------
    # 4. Laya receives only the bounded decision context
    # -----------------------------------------------------------------------
    def test_04_laya_receives_only_bounded_decision_context(self):
        """4. Verify Laya provider receives only bounded DecisionQuestion with minimized context."""
        with patch.object(self.mock_laya_provider, "decide", wraps=self.mock_laya_provider.decide) as spy_laya:
            ws = self.runtime.get_world_state()
            self.service.execute_v3_step(ws, step_number=1)

            spy_laya.assert_called_once()
            called_arg = spy_laya.call_args[0][0]
            self.assertIsInstance(called_arg, DecisionQuestion)
            self.assertIsInstance(called_arg.context, DecisionContext)
            # Context must have no direct simulation handles or motor controllers
            self.assertFalse(hasattr(called_arg.context, "mjModel"))
            self.assertFalse(hasattr(called_arg.context, "runtime"))

    # -----------------------------------------------------------------------
    # 5. Laya returns JevDecision
    # -----------------------------------------------------------------------
    def test_05_laya_returns_jev_decision(self):
        """5. Verify Laya query produces a validated JevDecision instance."""
        ws = self.runtime.get_world_state()
        question = self.service.build_bounded_question(ws, step_number=1)
        decision = self.mock_laya_provider.decide(question)

        self.assertIsInstance(decision, JevDecision)
        self.assertEqual(decision.provider, "laya")
        self.assertIsNotNone(decision.selected_option)
        self.assertIsNotNone(decision.confidence)

    # -----------------------------------------------------------------------
    # 6. CandidateActionInjector converts the decision
    # -----------------------------------------------------------------------
    def test_06_candidate_action_injector_converts_decision(self):
        """6. Verify CandidateActionInjector translates JevDecision to canonical CandidateAction."""
        decision = JevDecision(
            selected_option="REPOSITION",
            confidence=0.85,
            answer_confidence=0.85,
            provider="laya",
        )
        ws = self.runtime.get_world_state()
        candidate = self.service.candidate_injector.inject_candidate(
            decision=decision,
            world_state=ws,
        )

        self.assertIsInstance(candidate, CandidateAction)
        self.assertEqual(candidate.action_type, ActionType.REPOSITION)
        self.assertEqual(candidate.parameters["provider"], "laya")
        self.assertEqual(candidate.parameters["candidate_source"], "bounded_laya_decision")

    # -----------------------------------------------------------------------
    # 7. Candidate reaches SarthiDecisionEngine
    # -----------------------------------------------------------------------
    def test_07_candidate_reaches_sarthi_decision_engine(self):
        """7. Verify injected CandidateAction is submitted to SarthiDecisionEngine."""
        decision = JevDecision(
            selected_option="REPOSITION",
            confidence=0.85,
            provider="laya",
        )
        ws = self.runtime.get_world_state()
        engine_dec, tel = self.service.candidate_injector.evaluate_candidate_with_engine(
            engine=self.engine,
            decision=decision,
            world_state=ws,
        )

        self.assertIsInstance(engine_dec, Decision)
        evaluated_actions = [ev.action.action_type for ev in engine_dec.candidate_evaluations]
        self.assertIn(ActionType.REPOSITION, evaluated_actions)

    # -----------------------------------------------------------------------
    # 8. Deterministic validation occurs before physical execution
    # -----------------------------------------------------------------------
    def test_08_deterministic_validation_occurs_before_physical_execution(self):
        """8. Verify ConstraintValidator runs and validates action before runtime dispatch."""
        with patch.object(ConstraintValidator, "validate", wraps=ConstraintValidator.validate) as spy_val:
            ws = self.runtime.get_world_state()
            self.service.execute_v3_step(ws, step_number=1)
            self.assertTrue(spy_val.called)

    # -----------------------------------------------------------------------
    # 9. Accepted candidate can proceed to existing execution layer
    # -----------------------------------------------------------------------
    def test_09_accepted_candidate_proceeds_to_execution(self):
        """9. Verify a physically feasible candidate is executed by MuJoCo runtime."""
        # Step 1 with APPROACH proposal
        laya_approach = MockLayaProvider(
            provider_name="laya", fixed_option="APPROACH", fixed_confidence=0.95
        )
        self.service.laya_provider = laya_approach

        ws_before = self.runtime.get_world_state()
        status, step_tel, ws_after = self.service.execute_v3_step(ws_before, step_number=1)

        self.assertEqual(step_tel.deterministic_validation_result, "accepted")
        self.assertTrue(step_tel.execution_success)
        self.assertGreater(ws_after.version, ws_before.version)

    # -----------------------------------------------------------------------
    # 10. Unsafe candidate is rejected before MuJoCo execution
    # -----------------------------------------------------------------------
    def test_10_unsafe_candidate_rejected_before_execution(self):
        """10. Verify unsafe candidate (e.g. out of reach) is rejected before MuJoCo execution."""
        # Craft JevDecision proposing an action that violates physical reach
        laya_unreachable = MockLayaProvider(
            provider_name="laya", fixed_option="MOVE", fixed_confidence=0.99
        )
        self.service.laya_provider = laya_unreachable

        # Create situation where direct MOVE target is unreachable
        ws = self.runtime.get_world_state()
        bad_question = DecisionQuestion(
            question_id="q_unsafe",
            question_type=DecisionQuestionType.RECOVERY_SELECTION,
            question_text="Unsafe question",
            available_options=["MOVE", "STOP"],
            context=self.service.context_builder.build_context(
                ws, "Move to target", ["MOVE", "STOP"]
            ),
        )

        with patch.object(self.runtime, "execute_action", wraps=self.runtime.execute_action) as spy_exec:
            # Inject a move candidate to unreachable distance (x=10.0)
            with patch.object(self.candidate_injector, "inject_candidate", return_value=CandidateAction(
                action_id="act_bad_move",
                action_type=ActionType.MOVE,
                target_position=Point3D(x=10.0, y=0.0, z=0.2),
            )):
                status, step_tel, _ = self.service.execute_v3_step(
                    ws, step_number=1, custom_question=bad_question
                )

            # The MOVE action was NOT executed; engine fell back to STOP
            called_action = spy_exec.call_args[0][0]
            self.assertEqual(called_action.action_type, ActionType.STOP)
            self.assertEqual(step_tel.deterministic_validation_result, "rejected")

    # -----------------------------------------------------------------------
    # 11. High Laya confidence cannot bypass physical validation
    # -----------------------------------------------------------------------
    def test_11_high_laya_confidence_cannot_bypass_physical_validation(self):
        """11. Verify 0.9999 AI confidence NEVER overrides deterministic physical rejection."""
        # Approach and grasp first so robot is holding object at table level
        self.runtime.execute_action(
            CandidateAction(
                action_type=ActionType.APPROACH,
                action_id="act_app",
                target_position=Point3D(x=0.25, y=0.15, z=0.20),
            )
        )
        self.runtime.execute_action(
            CandidateAction(
                action_type=ActionType.GRASP,
                action_id="act_gr",
                target_position=Point3D(x=0.25, y=0.15, z=0.20),
                target_object_id="red_object_01",
            )
        )

        # Inject obstacle between robot and destination
        self.runtime.inject_disturbance("PATH_BLOCKED")
        ws = self.runtime.get_world_state()

        # Laya proposes MOVE through blocked path with 0.9999 confidence
        laya_overconfident = MockLayaProvider(
            provider_name="laya", fixed_option="MOVE", fixed_confidence=0.9999
        )
        self.service.laya_provider = laya_overconfident

        question = DecisionQuestion(
            question_id="q_overconfident",
            question_type=DecisionQuestionType.RECOVERY_SELECTION,
            question_text="Consider action",
            available_options=["MOVE", "STOP"],
            context=self.service.context_builder.build_context(ws, "Move", ["MOVE", "STOP"]),
        )

        status, step_tel, _ = self.service.execute_v3_step(
            ws, step_number=1, custom_question=question
        )

        # Candidate MUST be rejected despite 0.9999 confidence
        self.assertEqual(step_tel.deterministic_validation_result, "rejected")
        self.assertNotEqual(step_tel.final_decision_action, "MOVE")
        self.assertEqual(step_tel.final_decision_action, "STOP")

    # -----------------------------------------------------------------------
    # 12. Laya timeout / failure uses deterministic fallback path
    # -----------------------------------------------------------------------
    def test_12_laya_timeout_failure_uses_deterministic_fallback(self):
        """12. Verify orchestrator falls back to deterministic decision engine if Laya times out."""
        # Mock Laya provider that raises JevTimeoutError
        failing_laya = MagicMock()
        failing_laya.decide.side_effect = JevTimeoutError("Laya HTTP request timed out after 5.0s")
        failing_laya.provider_name = "laya"
        failing_laya.model = "laya-rl-agent"
        self.service.laya_provider = failing_laya

        ws = self.runtime.get_world_state()
        status, step_tel, ws_after = self.service.execute_v3_step(ws, step_number=1)

        # Fallback was engaged; pipeline did NOT crash; engine selected deterministic action
        self.assertEqual(step_tel.deterministic_validation_result, "fallback")
        self.assertTrue(step_tel.execution_success)
        self.assertEqual(step_tel.final_decision_action, "APPROACH")

    # -----------------------------------------------------------------------
    # 13. Malformed Laya response cannot reach execution
    # -----------------------------------------------------------------------
    def test_13_malformed_laya_response_cannot_reach_execution(self):
        """13. Verify invalid/unrecognized Laya responses are caught before execution."""
        bad_laya = MockLayaProvider(
            provider_name="laya", fixed_option="ILLEGAL_TELEPORT_COMMAND", fixed_confidence=0.9
        )
        self.service.laya_provider = bad_laya

        ws = self.runtime.get_world_state()
        # Candidate injector will raise JevValidationError, triggering fallback
        status, step_tel, _ = self.service.execute_v3_step(ws, step_number=1)

        self.assertEqual(step_tel.deterministic_validation_result, "rejected")
        self.assertNotEqual(step_tel.final_decision_action, "ILLEGAL_TELEPORT_COMMAND")

    # -----------------------------------------------------------------------
    # 14. Nemotron timeout / malformed output cannot reach physical execution
    # -----------------------------------------------------------------------
    def test_14_nemotron_timeout_malformed_output_aborts_without_execution(self):
        """14. Verify Nemotron failure aborts task run with zero physical robot actions."""
        from backend.app.model.provider import ModelProvider

        class FailingModel(ModelProvider):
            def understand_task(self, instruction, world_context=None):
                raise NebiusTimeoutError("Nemotron call timed out")

        self.service.task_service = TaskUnderstandingService(FailingModel())

        with patch.object(self.runtime, "execute_action") as spy_exec:
            telemetry = self.service.run_instruction("Move the red object to the blue target.")
            self.assertEqual(telemetry.final_task_status, "FAILED")
            self.assertFalse(telemetry.completed)
            # ZERO actions executed
            self.assertEqual(len(telemetry.steps), 0)
            spy_exec.assert_not_called()

    # -----------------------------------------------------------------------
    # 15. No direct Laya -> MuJoCo path exists
    # -----------------------------------------------------------------------
    def test_15_no_direct_laya_to_mujoco_path_exists(self):
        """15. Verify Laya provider has zero methods or references to simulation / motors."""
        forbidden_attrs = ["adapter", "runtime", "execute_action", "step", "command_motors", "mjData"]
        for attr in forbidden_attrs:
            self.assertFalse(
                hasattr(self.mock_laya_provider, attr),
                f"Laya provider violates boundary: has '{attr}' attribute!",
            )

    # -----------------------------------------------------------------------
    # 16. No direct Nemotron -> MuJoCo path exists
    # -----------------------------------------------------------------------
    def test_16_no_direct_nemotron_to_mujoco_path_exists(self):
        """16. Verify Nemotron provider has zero methods or references to simulation / motors."""
        forbidden_attrs = ["adapter", "runtime", "execute_action", "step", "command_motors", "mjData"]
        for attr in forbidden_attrs:
            self.assertFalse(
                hasattr(self.mock_model_provider, attr),
                f"Model provider violates boundary: has '{attr}' attribute!",
            )

    # -----------------------------------------------------------------------
    # 17. Existing recovery behavior remains intact
    # -----------------------------------------------------------------------
    def test_17_existing_recovery_behavior_remains_intact(self):
        """17. Verify full pick-and-place with PATH_BLOCKED disturbance recovers via REPOSITION."""
        def disturbance_cb(step_num: int, ws: WorldState):
            if step_num == 2:
                # Disturbance injected post-grasp
                self.runtime.inject_disturbance("PATH_BLOCKED")

        # Configure dynamic Laya mock matching tabletop phases
        class DynamicLayaProvider:
            provider_name = "laya"
            model = "laya-rl-agent"
            def decide(self, question: DecisionQuestion) -> JevDecision:
                opts = question.available_options
                if "REPOSITION" in opts and "STOP" in opts and len(opts) == 2:
                    return JevDecision(selected_option="REPOSITION", confidence=0.88, provider="laya")
                if "APPROACH" in opts:
                    return JevDecision(selected_option="APPROACH", confidence=0.92, provider="laya")
                if "GRASP" in opts:
                    return JevDecision(selected_option="GRASP", confidence=0.95, provider="laya")
                if "MOVE" in opts:
                    return JevDecision(selected_option="MOVE", confidence=0.90, provider="laya")
                if "RELEASE" in opts:
                    return JevDecision(selected_option="RELEASE", confidence=0.94, provider="laya")
                return JevDecision(selected_option=opts[0], confidence=0.50, provider="laya")

        self.service.laya_provider = DynamicLayaProvider()

        telemetry = self.service.run_instruction(
            self.instruction,
            step_callback=disturbance_cb,
        )

        self.assertTrue(telemetry.completed)
        self.assertEqual(telemetry.final_task_status, "COMPLETED")
        self.assertTrue(any(ev["event"] == "DISTURBANCE_RECOVERY_REPOSITION" for ev in telemetry.recovery_events))

        # Verify ground truth physical placement inside tolerance
        sr = self.runtime.state_reader
        final_obj_pos = sr.get_red_object_position()
        tgt_pos = sr.get_blue_target_position()
        tol = sr.get_blue_target_tolerance()
        h_dist = math.sqrt((final_obj_pos.x - tgt_pos.x)**2 + (final_obj_pos.y - tgt_pos.y)**2)
        self.assertLessEqual(h_dist, tol)

    # -----------------------------------------------------------------------
    # 18. Provenance is preserved from Laya -> CandidateAction -> Decision -> Telemetry
    # -----------------------------------------------------------------------
    def test_18_provenance_preserved_end_to_end(self):
        """18. Verify provenance metadata flows from Laya to CandidateAction and Telemetry."""
        laya_prov = MockLayaProvider(
            provider_name="laya",
            model_name="laya-rl-agent",
            fixed_option="APPROACH",
            fixed_confidence=0.77,
        )
        self.service.laya_provider = laya_prov

        ws = self.runtime.get_world_state()
        _, step_tel, _ = self.service.execute_v3_step(ws, step_number=1)

        self.assertEqual(step_tel.laya_provider, "laya")
        self.assertEqual(step_tel.laya_model, "laya-rl-agent")
        self.assertEqual(step_tel.selected_option, "APPROACH")
        self.assertAlmostEqual(step_tel.laya_confidence, 0.77)
        self.assertEqual(step_tel.candidate_action, "APPROACH")
        self.assertEqual(step_tel.deterministic_validation_result, "accepted")

    # -----------------------------------------------------------------------
    # 19. No secrets appear in telemetry
    # -----------------------------------------------------------------------
    def test_19_no_secrets_appear_in_telemetry(self):
        """19. Verify generated telemetry contains zero secret tokens or credentials."""
        telemetry = self.service.run_instruction("Move the red object to the blue target.")
        tel_dict = telemetry.to_dict()
        tel_json = json.dumps(tel_dict)

        forbidden_patterns = ["NEBIUS_API_KEY", "JEV_API_KEY", "sk-", "nbf_", "Bearer "]
        for pat in forbidden_patterns:
            self.assertNotIn(
                pat.lower(),
                tel_json.lower(),
                f"Secret pattern '{pat}' leaked in telemetry JSON!",
            )

    # -----------------------------------------------------------------------
    # 20. Existing tests remain passing (verified via full suite)
    # -----------------------------------------------------------------------
    def test_20_service_exposes_compliant_interface(self):
        """20. Verify V3OrchestrationService adheres to the unified orchestration interface."""
        self.assertTrue(callable(getattr(self.service, "run_instruction", None)))
        self.assertTrue(callable(getattr(self.service, "execute_v3_step", None)))
        self.assertTrue(callable(getattr(self.service, "build_bounded_question", None)))
        self.assertTrue(callable(getattr(self.service, "is_task_completed", None)))


if __name__ == "__main__":
    unittest.main()
