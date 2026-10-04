"""
SĀRTHI V3-7 — Latency Instrumentation & Benchmarking Unit Test Suite.

Proves:
1. All latency markers are recorded.
2. Values are non-negative.
3. tau_total is measured independently.
4. Missing optional stages do not corrupt telemetry.
5. Secret sanitization remains intact.
6. Existing V3 telemetry remains compatible.
7. Benchmark statistics are mathematically correct.
8. Instrumentation does not change execution ordering.
9. Instrumentation cannot execute robot actions.
10. Existing safety authority remains unchanged.
"""

from __future__ import annotations

import json
import math
import time
import unittest
from typing import Any, Dict, List, Optional

import numpy as np

from backend.app.decision_engine.candidate_injector import CandidateActionInjector
from backend.app.decision_engine.engine import SarthiDecisionEngine
from backend.app.decision_engine.models import (
    ActionType,
    CandidateAction,
    Point3D,
    RobotState,
    WorldState,
)
from backend.app.model.mock_provider import MockModelProvider
from backend.app.model.models import TaskUnderstanding
from backend.app.model.v3_models import (
    DecisionContext,
    DecisionQuestion,
    DecisionQuestionType,
    JevDecision,
)
from backend.app.orchestration.context_builder import DecisionContextBuilder
from backend.app.orchestration.latency_tracker import (
    BenchmarkSuiteResult,
    LATENCY_MARKER_NAMES,
    LatencyTracker,
    V3LatencyBreakdown,
    aggregate_latency_samples,
    compute_metric_statistics,
)
from backend.app.orchestration.loop import TaskRunStatus
from backend.app.orchestration.v3_orchestrator import (
    V3ExecutionTelemetry,
    V3OrchestrationService,
    V3StepTelemetry,
)
from simulation.mujoco_runtime.runtime import SarthiMuJoCoRuntime


class MockLayaProvider:
    """Deterministic fast mock Laya decision provider for testing."""
    provider_name = "laya"
    model = "laya-rl-agent"

    def __init__(self, fixed_option: str = "APPROACH", fixed_confidence: float = 0.85):
        self.fixed_option = fixed_option
        self.fixed_confidence = fixed_confidence

    def decide(self, question: DecisionQuestion) -> JevDecision:
        opts = question.available_options
        chosen = self.fixed_option if self.fixed_option in opts else opts[0]
        probs = {opt: (self.fixed_confidence if opt == chosen else (1.0 - self.fixed_confidence) / max(1, len(opts) - 1)) for opt in opts}
        return JevDecision(
            selected_option=chosen,
            option_probabilities=probs,
            confidence=self.fixed_confidence,
            answer_confidence=self.fixed_confidence,
            provider="laya",
            model_version=self.model,
            latency_ms=12.5,
        )


class TestV37LatencyInstrumentation(unittest.TestCase):
    """Focused test suite validating V3-7 latency tracking and statistics."""

    def setUp(self) -> None:
        self.runtime = SarthiMuJoCoRuntime()
        self.engine = SarthiDecisionEngine()
        self.candidate_injector = CandidateActionInjector()
        self.context_builder = DecisionContextBuilder()
        self.mock_model_provider = MockModelProvider()
        self.mock_laya_provider = MockLayaProvider()

        self.service = V3OrchestrationService(
            adapter=self.runtime,
            engine=self.engine,
            context_builder=self.context_builder,
            candidate_injector=self.candidate_injector,
            model_provider=self.mock_model_provider,
            laya_provider=self.mock_laya_provider,
            max_steps=5,
        )

    # -----------------------------------------------------------------------
    # 1. All latency markers are recorded
    # -----------------------------------------------------------------------
    def test_01_all_latency_markers_recorded(self):
        """1. Verify all 8 required latency markers are populated in telemetry."""
        telemetry = self.service.run_instruction("Move the red object to the blue target.")
        self.assertIn("latency", telemetry.to_dict())
        lat = telemetry.latency

        expected_markers = [
            "tau_nemotron_ms",
            "tau_context_ms",
            "tau_jev_ms",
            "tau_validation_ms",
            "tau_ik_ms",
            "tau_sim_ms",
            "tau_verify_ms",
            "tau_total_ms",
        ]
        for marker in expected_markers:
            self.assertIn(marker, lat, f"Missing latency marker: {marker}")
            self.assertIsInstance(lat[marker], (int, float))

    # -----------------------------------------------------------------------
    # 2. Values are non-negative
    # -----------------------------------------------------------------------
    def test_02_values_are_non_negative(self):
        """2. Verify that all recorded latency values are strictly non-negative."""
        telemetry = self.service.run_instruction("Move the red object to the blue target.")
        lat = telemetry.latency
        for marker, val in lat.items():
            self.assertGreaterEqual(val, 0.0, f"Marker {marker} has negative value: {val}")

        # Check per-step latency markers
        for step in telemetry.steps:
            self.assertIsNotNone(step.latency)
            for m, val in step.latency.items():
                self.assertGreaterEqual(val, 0.0, f"Step marker {m} has negative value: {val}")

    # -----------------------------------------------------------------------
    # 3. tau_total is measured independently
    # -----------------------------------------------------------------------
    def test_03_tau_total_measured_independently(self):
        """3. Verify tau_total is measured via an outer monotonic transaction timer."""
        tracker = LatencyTracker()
        tracker.start_total()
        tracker.record("tau_nemotron_ms", 10.0)
        tracker.record("tau_context_ms", 5.0)
        time.sleep(0.02)  # 20ms outer elapsed
        total = tracker.stop_total()
        self.assertGreaterEqual(total, 18.0)  # Measured independently, not just 10+5=15
        breakdown = tracker.get_breakdown()
        self.assertEqual(breakdown.tau_nemotron_ms, 10.0)
        self.assertEqual(breakdown.tau_context_ms, 5.0)
        self.assertGreaterEqual(breakdown.tau_total_ms, 18.0)

    # -----------------------------------------------------------------------
    # 4. Missing optional stages do not corrupt telemetry
    # -----------------------------------------------------------------------
    def test_04_missing_optional_stages_do_not_corrupt_telemetry(self):
        """4. Verify that early termination (e.g. cognition error) gracefully sets remaining markers to 0.0."""
        class FailingModelProvider(MockModelProvider):
            def understand_task(self, instruction: str, world_context: Optional[Dict[str, Any]] = None):
                raise RuntimeError("Cognition offline")

        service_fail = V3OrchestrationService(
            adapter=self.runtime,
            model_provider=FailingModelProvider(),
            laya_provider=self.mock_laya_provider,
        )
        telemetry = service_fail.run_instruction("Move the red object to the blue target.")
        self.assertEqual(telemetry.final_task_status, "FAILED")
        lat = telemetry.latency
        self.assertGreaterEqual(lat["tau_nemotron_ms"], 0.0)
        self.assertGreaterEqual(lat["tau_total_ms"], 0.0)
        self.assertEqual(lat["tau_context_ms"], 0.0)
        self.assertEqual(lat["tau_sim_ms"], 0.0)

    # -----------------------------------------------------------------------
    # 5. Secret sanitization remains intact
    # -----------------------------------------------------------------------
    def test_05_secret_sanitization_remains_intact(self):
        """5. Verify that latency dictionaries contain zero leaked secrets or tokens."""
        telemetry = self.service.run_instruction("Move the red object to the blue target.")
        tel_json = json.dumps(telemetry.to_dict())

        for secret in ["sk-", "Bearer", "nbf_", "NEBIUS_API_KEY", "JEV_API_KEY", "token", "password"]:
            self.assertNotIn(
                secret.lower(),
                tel_json.lower(),
                f"Secret pattern '{secret}' leaked in serialized telemetry!",
            )

    # -----------------------------------------------------------------------
    # 6. Existing V3 telemetry remains compatible
    # -----------------------------------------------------------------------
    def test_06_existing_v3_telemetry_remains_compatible(self):
        """6. Verify latency_markers and existing telemetry keys are preserved."""
        telemetry = self.service.run_instruction("Move the red object to the blue target.")
        d = telemetry.to_dict()

        # Check existing fields preserved
        self.assertIn("run_id", d)
        self.assertIn("human_instruction", d)
        self.assertIn("task_understanding", d)
        self.assertIn("latency_markers", d)
        self.assertIn("cognition_latency_ms", d["latency_markers"])
        self.assertIn("total_run_duration_ms", d["latency_markers"])
        self.assertIn("latency", d)

    # -----------------------------------------------------------------------
    # 7. Benchmark statistics are mathematically correct
    # -----------------------------------------------------------------------
    def test_07_benchmark_statistics_mathematically_correct(self):
        """7. Verify compute_metric_statistics accurately computes count, min, max, mean, median, p95."""
        samples = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = compute_metric_statistics(samples)

        self.assertEqual(stats["count"], 5.0)
        self.assertEqual(stats["min"], 10.0)
        self.assertEqual(stats["max"], 50.0)
        self.assertAlmostEqual(stats["mean"], 30.0)
        self.assertAlmostEqual(stats["median"], 30.0)
        # 95th percentile of [10, 20, 30, 40, 50]
        expected_p95 = float(np.percentile([10.0, 20.0, 30.0, 40.0, 50.0], 95))
        self.assertAlmostEqual(stats["p95"], expected_p95)

        # Empty sample handling
        empty_stats = compute_metric_statistics([])
        self.assertEqual(empty_stats["count"], 0.0)
        self.assertEqual(empty_stats["mean"], 0.0)

    # -----------------------------------------------------------------------
    # 8. Instrumentation does not change execution ordering
    # -----------------------------------------------------------------------
    def test_08_instrumentation_does_not_change_execution_ordering(self):
        """8. Verify action primitives execute in exact valid sequence with timing active."""
        class SequenceLayaProvider:
            provider_name = "laya"
            model = "laya-rl-agent"
            def decide(self, question: DecisionQuestion) -> JevDecision:
                opts = question.available_options
                for opt in ["APPROACH", "GRASP", "MOVE", "RELEASE", "STOP"]:
                    if opt in opts:
                        return JevDecision(selected_option=opt, confidence=0.9, provider="laya")
                return JevDecision(selected_option=opts[0], confidence=0.5, provider="laya")

        self.service.laya_provider = SequenceLayaProvider()
        telemetry = self.service.run_instruction("Move the red object to the blue target.")
        executed_actions = [s.final_decision_action for s in telemetry.steps]

        # First two actions must be APPROACH then GRASP
        self.assertGreaterEqual(len(executed_actions), 2)
        self.assertEqual(executed_actions[0], "APPROACH")
        self.assertEqual(executed_actions[1], "GRASP")

    # -----------------------------------------------------------------------
    # 9. Instrumentation cannot execute robot actions
    # -----------------------------------------------------------------------
    def test_09_instrumentation_cannot_execute_robot_actions(self):
        """9. Verify LatencyTracker has zero methods or handles to control simulation or robot."""
        tracker = LatencyTracker()
        forbidden_attrs = ["adapter", "runtime", "step", "execute_action", "command_motors", "mjData"]
        for attr in forbidden_attrs:
            self.assertFalse(hasattr(tracker, attr), f"LatencyTracker has forbidden attribute: {attr}")

    # -----------------------------------------------------------------------
    # 10. Existing safety authority remains unchanged
    # -----------------------------------------------------------------------
    def test_10_existing_safety_authority_remains_unchanged(self):
        """10. Verify that unsafe candidate actions are still rejected and timing is recorded."""
        class UnsafeCandidateLaya:
            provider_name = "laya"
            model = "laya-rl-agent"
            def decide(self, question: DecisionQuestion) -> JevDecision:
                # Propose GRASP before APPROACH (unsafe)
                return JevDecision(selected_option="GRASP", confidence=0.99, provider="laya")

        self.service.laya_provider = UnsafeCandidateLaya()
        ws = self.runtime.get_world_state()
        _, step_tel, _ = self.service.execute_v3_step(ws, step_number=1)

        # Candidate rejected by deterministic validation
        self.assertEqual(step_tel.deterministic_validation_result, "rejected")
        # Timing still recorded properly
        self.assertGreaterEqual(step_tel.latency["tau_validation_ms"], 0.0)
        self.assertGreaterEqual(step_tel.latency["tau_total_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
