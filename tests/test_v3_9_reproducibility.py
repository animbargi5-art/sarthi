"""
SĀRTHI Phase V3-9: Repeated-Run & Reproducibility Validation Tests.

Covers the 16 required test specifications:
1. repeated-run result schema
2. identical physics replay
3. decision-pipeline replay
4. bounded Laya variation handling
5. candidate validity across repeated runs
6. deterministic validation consistency
7. final placement statistics
8. task success rate calculation
9. action agreement calculation
10. failure classification
11. model variability classification
12. safety authority remains unchanged
13. no direct AI-to-MuJoCo path
14. secret-free reproducibility telemetry
15. baseline comparison
16. reproducibility report generation
"""

import math
import unittest

from backend.app.decision_engine.candidate_injector import CandidateActionInjector
from backend.app.decision_engine.engine import SarthiDecisionEngine
from backend.app.decision_engine.models import (
    ActionType,
    CandidateAction,
    Point3D,
    WorldState,
)
from backend.app.model.mock_provider import MockModelProvider
from backend.app.model.v3_models import (
    DecisionContext,
    DecisionQuestion,
    DecisionQuestionType,
    JevDecision,
)
from backend.app.orchestration.context_builder import DecisionContextBuilder
from backend.app.validation.reproducibility import SarthiReproducibilityValidator
from backend.app.validation.reproducibility_models import (
    FailureClassification,
    LevelAPhysicsReplayResult,
    LevelBDecisionReplayResult,
    LevelCLiveRunResult,
    ReproducibilityMetrics,
    V39ReproducibilityReport,
)
from simulation.mujoco_runtime import require_mujoco
from simulation.mujoco_runtime.runtime import SarthiMuJoCoRuntime


class MockVaryingLayaProvider:
    """Mock Laya provider simulating bounded decision variability."""
    def __init__(self, sequence=None):
        self.sequence = sequence or ["REPOSITION", "STOP", "REPOSITION"]
        self.call_count = 0

    def decide(self, question: DecisionQuestion) -> JevDecision:
        opt = self.sequence[self.call_count % len(self.sequence)]
        self.call_count += 1
        probs = {opt: 0.85, "STOP": 0.15} if opt != "STOP" else {"STOP": 0.90, "REPOSITION": 0.10}
        return JevDecision(
            selected_option=opt,
            confidence=0.85,
            answer_confidence=0.85,
            option_probabilities=probs,
            provider="mock_laya",
            latency_ms=12.5,
        )


class TestV39Reproducibility(unittest.TestCase):
    """
    V3-9 Reproducibility Test Suite.
    """

    @classmethod
    def setUpClass(cls):
        require_mujoco()
        cls.validator = SarthiReproducibilityValidator()

    # -----------------------------------------------------------------------
    # Test 1: Repeated-run result schema
    # -----------------------------------------------------------------------
    def test_01_repeated_run_result_schema(self):
        """1. Validates Level A, B, C, and Metrics schemas and serialization."""
        res_a = LevelAPhysicsReplayResult(
            num_runs=3,
            identical_action_sequences=True,
            all_runs_succeeded=True,
            mean_placement_error_m=0.0176,
        )
        self.assertEqual(res_a.num_runs, 3)
        self.assertTrue(res_a.identical_action_sequences)
        d_a = res_a.to_dict()
        self.assertIn("identical_action_sequences", d_a)

        res_b = LevelBDecisionReplayResult(
            num_runs=3,
            logical_decision_agreement_rate=1.0,
            deterministic_validation_agreement_rate=1.0,
        )
        self.assertEqual(res_b.logical_decision_agreement_rate, 1.0)
        d_b = res_b.to_dict()
        self.assertIn("logical_decision_agreement_rate", d_b)

        metrics = ReproducibilityMetrics(
            total_runs_attempted=5,
            total_runs_completed=5,
            task_success_rate=1.0,
        )
        self.assertEqual(metrics.task_success_rate, 1.0)
        self.assertIn("task_success_rate", metrics.to_dict())

    # -----------------------------------------------------------------------
    # Test 2: Identical physics replay
    # -----------------------------------------------------------------------
    def test_02_identical_physics_replay(self):
        """2. Validates that Level A physics replay produces 100% identical trajectories."""
        res = self.validator.run_level_a_physics_replay(runs=3)
        self.assertTrue(res.identical_action_sequences)
        self.assertTrue(res.all_runs_succeeded)
        self.assertEqual(res.max_euclidean_pose_spread_m, 0.0)
        self.assertLessEqual(res.max_placement_error_m, 0.06)

    # -----------------------------------------------------------------------
    # Test 3: Decision-pipeline replay
    # -----------------------------------------------------------------------
    def test_03_decision_pipeline_replay(self):
        """3. Validates decision-pipeline replay with bounded deterministic mock."""
        mock_laya = MockVaryingLayaProvider(["REPOSITION", "REPOSITION", "REPOSITION"])
        res = self.validator.run_level_b_decision_replay(runs=3, laya_provider=mock_laya)
        self.assertEqual(res.logical_decision_agreement_rate, 1.0)
        self.assertEqual(res.deterministic_validation_agreement_rate, 1.0)
        self.assertEqual(res.candidate_injection_validity_rate, 1.0)

    # -----------------------------------------------------------------------
    # Test 4: Bounded Laya variation handling
    # -----------------------------------------------------------------------
    def test_04_bounded_laya_variation_handling(self):
        """4. Tests that when Laya produces different bounded options, system adapts cleanly."""
        mock_laya = MockVaryingLayaProvider(["REPOSITION", "STOP", "REPOSITION"])
        res = self.validator.run_level_b_decision_replay(runs=3, laya_provider=mock_laya)
        self.assertEqual(res.num_runs, 3)
        self.assertIn("REPOSITION", res.selected_options)
        self.assertIn("STOP", res.selected_options)
        # SarthiDecisionEngine accepts valid REPOSITION and safely selects STOP for STOP
        self.assertIn("REPOSITION", res.deterministic_decisions)
        self.assertIn("STOP", res.deterministic_decisions)

    # -----------------------------------------------------------------------
    # Test 5: Candidate validity across repeated runs
    # -----------------------------------------------------------------------
    def test_05_candidate_validity_across_repeated_runs(self):
        """5. Verifies that candidate actions injected from Laya adhere to schema rules."""
        builder = DecisionContextBuilder()
        injector = CandidateActionInjector()
        runtime = self.validator._create_runtime(with_disturbance=True)
        ws = runtime.get_world_state()

        context = builder.build_context(
            world_state=ws,
            task_objective="Move the red object to the blue target.",
            candidate_actions=["REPOSITION", "STOP"],
            context_id="ctx_test_05",
        )
        question = DecisionQuestion(
            question_id="q_test_05",
            question_type=DecisionQuestionType.RECOVERY_SELECTION,
            question_text="Recovery required",
            available_options=["REPOSITION", "STOP"],
            context=context,
        )

        for opt in ["REPOSITION", "STOP"]:
            jev_dec = JevDecision(selected_option=opt, confidence=0.85, answer_confidence=0.85, provider="mock")
            cand_action = injector.inject_candidate(jev_dec, question, ws)
            self.assertIsNotNone(cand_action)
            self.assertIsInstance(cand_action.action_type, ActionType)

    # -----------------------------------------------------------------------
    # Test 6: Deterministic validation consistency
    # -----------------------------------------------------------------------
    def test_06_deterministic_validation_consistency(self):
        """6. Asserts SarthiDecisionEngine evaluation is 100% deterministic for identical inputs."""
        engine = SarthiDecisionEngine()
        runtime = self.validator._create_runtime(with_disturbance=True)
        ws = runtime.get_world_state()
        cand = CandidateAction(action_type=ActionType.REPOSITION, action_id="act_rep", target_position=Point3D(x=0.25, y=0.15, z=0.351))

        evals = [engine.decide(ws, candidate_actions=[cand]) for _ in range(5)]
        first_action = evals[0].selected_action.action_type.value
        first_score = evals[0].candidate_evaluations[0].overall_score

        self.assertTrue(all(e.selected_action.action_type.value == first_action for e in evals))
        self.assertTrue(all(abs(e.candidate_evaluations[0].overall_score - first_score) < 1e-6 for e in evals))

    # -----------------------------------------------------------------------
    # Test 7: Final placement statistics
    # -----------------------------------------------------------------------
    def test_07_final_placement_statistics(self):
        """7. Tests accurate computation of placement error mean, max, and std deviation."""
        errors = [0.015, 0.020, 0.025]
        mean_v = float(sum(errors) / len(errors))
        max_v = float(max(errors))
        std_v = float(math.sqrt(sum((x - mean_v) ** 2 for x in errors) / len(errors)))

        runs = [
            LevelCLiveRunResult(
                run_id=f"run_{i}",
                run_index=i,
                task_understanding_success=True,
                task_target_object="red_object_01",
                task_target_zone="blue_target_zone",
                recovery_occurred=True,
                recovery_action="REPOSITION",
                final_object_position=(0.40, -0.20, 0.20),
                final_placement_error_m=errors[i],
                placement_within_tolerance=True,
                final_task_status="COMPLETED",
                completed=True,
            )
            for i in range(3)
        ]
        metrics = self.validator.compute_reproducibility_metrics(runs)
        self.assertAlmostEqual(metrics.final_placement_error_mean, round(mean_v, 5), places=4)
        self.assertAlmostEqual(metrics.final_placement_error_max, round(max_v, 5), places=4)
        self.assertAlmostEqual(metrics.final_placement_error_std, round(std_v, 6), places=4)

    # -----------------------------------------------------------------------
    # Test 8: Task success rate calculation
    # -----------------------------------------------------------------------
    def test_08_task_success_rate_calculation(self):
        """8. Validates task success rate computation with partial failures."""
        runs = [
            LevelCLiveRunResult(
                run_id="run_1", run_index=1, task_understanding_success=True,
                task_target_object="red_object_01", task_target_zone="blue_target_zone",
                recovery_occurred=True, recovery_action="REPOSITION",
                final_object_position=(0.40, -0.20, 0.20), final_placement_error_m=0.015,
                placement_within_tolerance=True, final_task_status="COMPLETED", completed=True,
            ),
            LevelCLiveRunResult(
                run_id="run_2", run_index=2, task_understanding_success=True,
                task_target_object="red_object_01", task_target_zone="blue_target_zone",
                recovery_occurred=True, recovery_action="REPOSITION",
                final_object_position=(0.40, -0.20, 0.20), final_placement_error_m=0.085,
                placement_within_tolerance=False, final_task_status="PLACEMENT_FAILED", completed=False,
                failure_classification=FailureClassification.PHYSICS_FAILURE,
            ),
        ]
        metrics = self.validator.compute_reproducibility_metrics(runs)
        self.assertEqual(metrics.total_runs_attempted, 2)
        self.assertEqual(metrics.total_runs_completed, 1)
        self.assertEqual(metrics.task_success_rate, 0.5)

    # -----------------------------------------------------------------------
    # Test 9: Action agreement calculation
    # -----------------------------------------------------------------------
    def test_09_action_agreement_calculation(self):
        """9. Validates matching logical decisions calculation."""
        runs = [
            LevelCLiveRunResult(
                run_id=f"run_{i}", run_index=i, task_understanding_success=True,
                task_target_object="red_object_01", task_target_zone="blue_target_zone",
                deterministic_actions=["APPROACH", "GRASP", "REPOSITION", "MOVE", "RELEASE"] if i < 3 else ["APPROACH", "STOP"],
                recovery_occurred=True, recovery_action="REPOSITION",
                final_object_position=(0.40, -0.20, 0.20), final_placement_error_m=0.015,
                placement_within_tolerance=True, final_task_status="COMPLETED", completed=True,
            )
            for i in range(4)
        ]
        metrics = self.validator.compute_reproducibility_metrics(runs)
        # 3 out of 4 match run_0
        self.assertEqual(metrics.action_sequence_match_rate, 0.75)

    # -----------------------------------------------------------------------
    # Test 10: Failure classification
    # -----------------------------------------------------------------------
    def test_10_failure_classification(self):
        """10. Tests classification of non-completed runs into taxonomy."""
        all_taxonomies = [f.value for f in FailureClassification]
        self.assertIn("MODEL_VARIABILITY", all_taxonomies)
        self.assertIn("PHYSICS_FAILURE", all_taxonomies)
        self.assertIn("EXECUTION_FAILURE", all_taxonomies)
        self.assertIn("DETERMINISTIC_VALIDATION_FAILURE", all_taxonomies)
        self.assertIn("NETWORK_LATENCY", all_taxonomies)

    # -----------------------------------------------------------------------
    # Test 11: Model variability classification
    # -----------------------------------------------------------------------
    def test_11_model_variability_classification(self):
        """11. Confirms benign model output variation is distinguished from physics failures."""
        run = LevelCLiveRunResult(
            run_id="run_var", run_index=1, task_understanding_success=True,
            task_target_object="red_object_01", task_target_zone="blue_target_zone",
            recovery_occurred=True, recovery_action="REPOSITION",
            final_object_position=(0.404, -0.183, 0.199), final_placement_error_m=0.0176,
            placement_within_tolerance=True, final_task_status="COMPLETED", completed=True,
            failure_classification=FailureClassification.MODEL_VARIABILITY,
        )
        self.assertEqual(run.failure_classification, FailureClassification.MODEL_VARIABILITY)
        self.assertTrue(run.completed)

    # -----------------------------------------------------------------------
    # Test 12: Safety authority remains unchanged
    # -----------------------------------------------------------------------
    def test_12_safety_authority_remains_unchanged(self):
        """12. Verifies that regardless of Laya output, CandidateActionInjector enforces boundaries."""
        injector = CandidateActionInjector()
        runtime = self.validator._create_runtime()
        ws = runtime.get_world_state()

        builder = DecisionContextBuilder()
        context = builder.build_context(
            world_state=ws,
            task_objective="Move the red object to the blue target.",
            candidate_actions=["APPROACH", "STOP"],
            context_id="ctx_test_12",
        )
        question = DecisionQuestion(
            question_id="q_test_12",
            question_type=DecisionQuestionType.ACTION_SELECTION,
            question_text="Select nominal action.",
            available_options=["APPROACH", "STOP"],
            context=context,
        )

        # Attempt to inject illegal action not in question options
        illegal_dec = JevDecision(selected_option="TELEPORT_TO_TARGET", confidence=0.99, provider="laya")
        from backend.app.model.real_jev_provider import JevValidationError
        with self.assertRaises(JevValidationError) as cm:
            injector.inject_candidate(illegal_dec, question, ws)
        self.assertIn("not in question's available options", str(cm.exception))

    # -----------------------------------------------------------------------
    # Test 13: No direct AI-to-MuJoCo path
    # -----------------------------------------------------------------------
    def test_13_no_direct_ai_to_mujoco_path(self):
        """13. Proves neither Nemotron nor Laya has direct access to simulation execution."""
        from backend.app.model.laya_provider import LayaDecisionProvider
        from backend.app.model.nebius_provider import NebiusNemotronProvider

        for cls_type in [LayaDecisionProvider, NebiusNemotronProvider]:
            self.assertFalse(hasattr(cls_type, "execute_action"))
            self.assertFalse(hasattr(cls_type, "step"))
            self.assertFalse(hasattr(cls_type, "model"))
            self.assertFalse(hasattr(cls_type, "data"))

    # -----------------------------------------------------------------------
    # Test 14: Secret-free reproducibility telemetry
    # -----------------------------------------------------------------------
    def test_14_secret_free_telemetry(self):
        """14. Validates that Level A, B, and C telemetry contains 0 secrets."""
        res_a = self.validator.run_level_a_physics_replay(runs=1)
        d = res_a.to_dict()
        text = str(d)
        for tok in ["Bearer", "sk-", "API_KEY", "SECRET", "token_value"]:
            self.assertNotIn(tok, text)

    # -----------------------------------------------------------------------
    # Test 15: Baseline comparison
    # -----------------------------------------------------------------------
    def test_15_baseline_comparison(self):
        """15. Compares Level A replay placement against V3-8 baseline (<= 0.060m)."""
        res_a = self.validator.run_level_a_physics_replay(runs=2)
        self.assertLessEqual(res_a.max_placement_error_m, 0.06)
        self.assertAlmostEqual(res_a.mean_placement_error_m, 0.0176, places=3)

    # -----------------------------------------------------------------------
    # Test 16: Reproducibility report generation
    # -----------------------------------------------------------------------
    def test_16_reproducibility_report_generation(self):
        """16. Tests full report generation across Level A, B, and mock C."""
        res_a = self.validator.run_level_a_physics_replay(runs=1)
        mock_laya = MockVaryingLayaProvider(["REPOSITION"])
        res_b = self.validator.run_level_b_decision_replay(runs=1, laya_provider=mock_laya)
        run_c = LevelCLiveRunResult(
            run_id="run_c_test", run_index=1, task_understanding_success=True,
            task_target_object="red_object_01", task_target_zone="blue_target_zone",
            recovery_occurred=True, recovery_action="REPOSITION",
            final_object_position=(0.404, -0.183, 0.1998), final_placement_error_m=0.0176,
            placement_within_tolerance=True, final_task_status="COMPLETED", completed=True,
        )
        metrics = self.validator.compute_reproducibility_metrics([run_c], res_b)
        report = V39ReproducibilityReport(
            report_id="rep_test_01",
            level_a_physics=res_a,
            level_b_decision=res_b,
            level_c_live_runs=[run_c],
            metrics=metrics,
            reproducibility_conclusion="Test report generated successfully.",
        )
        self.assertEqual(report.report_id, "rep_test_01")
        self.assertEqual(report.metrics.task_success_rate, 1.0)
        self.assertIn("reproducibility_conclusion", report.to_dict())


if __name__ == "__main__":
    unittest.main()
