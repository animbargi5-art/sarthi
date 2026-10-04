#!/usr/bin/env python3
"""
SĀRTHI V3-7 — Decision Latency Benchmarking Runner.

Executes three controlled benchmark suites:
  A. Offline Deterministic Benchmark (5 cycles, mock providers, MuJoCo tabletop scenario)
  B. Local Laya Benchmark (20 queries to local Laya server http://127.0.0.1:8000)
  C. Full Live V3 Benchmark (5 cycles with real Nemotron + real local Laya + MuJoCo recovery)

Strict Invariants:
  - Sub-millisecond monotonic timing (time.perf_counter).
  - Calculates count, min, max, mean, median, and p95 for all markers.
  - Secret-free serialization in records/v3_7_latency_benchmark.json.
  - Measurement only: zero changes to physics, controllers, or decision logic.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
import os
import pathlib
import sys
import time
from typing import Any, Dict, List

# Ensure project root in sys.path
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Ensure UTF-8 on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Load environment configuration safely
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT_ROOT, "configs", ".env"))
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
except ImportError:
    pass

from backend.app.decision_engine.candidate_injector import CandidateActionInjector
from backend.app.decision_engine.engine import SarthiDecisionEngine
from backend.app.decision_engine.models import WorldState
from backend.app.model.laya_config import LayaConfig
from backend.app.model.laya_provider import LayaDecisionProvider
from backend.app.model.mock_provider import MockModelProvider
from backend.app.model.nebius_provider import NebiusNemotronProvider
from backend.app.model.v3_models import (
    DecisionContext,
    DecisionQuestion,
    DecisionQuestionType,
    JevDecision,
)
from backend.app.orchestration.context_builder import DecisionContextBuilder
from backend.app.orchestration.latency_tracker import (
    LATENCY_MARKER_NAMES,
    V3LatencyBreakdown,
    aggregate_latency_samples,
    compute_metric_statistics,
)
from backend.app.orchestration.v3_orchestrator import (
    V3ExecutionTelemetry,
    V3OrchestrationService,
)
from simulation.mujoco_runtime.runtime import SarthiMuJoCoRuntime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sarthi_v3_7_benchmark")


class DeterministicMockLaya:
    """Deterministic fast Laya mock for offline overhead benchmarking."""
    provider_name = "laya"
    model = "laya-rl-agent"

    def decide(self, question: DecisionQuestion) -> JevDecision:
        opts = question.available_options
        if "REPOSITION" in opts and "STOP" in opts and len(opts) == 2:
            return JevDecision(selected_option="REPOSITION", confidence=0.88, provider="laya")
        for preferred in ["APPROACH", "GRASP", "MOVE", "RELEASE"]:
            if preferred in opts:
                return JevDecision(selected_option=preferred, confidence=0.92, provider="laya")
        return JevDecision(selected_option=opts[0], confidence=0.50, provider="laya")


# ---------------------------------------------------------------------------
# Benchmark A: Offline Deterministic Benchmark
# ---------------------------------------------------------------------------
def run_benchmark_a_offline(num_runs: int = 5) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f" BENCHMARK A: OFFLINE DETERMINISTIC BENCHMARK ({num_runs} runs)")
    print("=" * 80)

    raw_runs: List[Dict[str, float]] = []

    for i in range(1, num_runs + 1):
        runtime = SarthiMuJoCoRuntime()
        service = V3OrchestrationService(
            adapter=runtime,
            engine=SarthiDecisionEngine(),
            candidate_injector=CandidateActionInjector(),
            context_builder=DecisionContextBuilder(),
            model_provider=MockModelProvider(),
            laya_provider=DeterministicMockLaya(),
            max_steps=10,
        )

        def disturbance_cb(step_num: int, ws: WorldState):
            if step_num == 2:
                runtime.inject_disturbance("PATH_BLOCKED")

        telemetry = service.run_instruction(
            "Move the red object to the blue target.",
            step_callback=disturbance_cb,
        )
        lat = telemetry.latency
        raw_runs.append(lat)
        print(f"  Run {i}/{num_runs}: tau_total={lat.get('tau_total_ms', 0):.2f}ms | tau_sim={lat.get('tau_sim_ms', 0):.2f}ms | tau_ik={lat.get('tau_ik_ms', 0):.2f}ms | tau_val={lat.get('tau_validation_ms', 0):.2f}ms")

    stats = aggregate_latency_samples(raw_runs)
    return {
        "benchmark_name": "Benchmark_A_Offline_Deterministic",
        "description": "Deterministic overhead measurement with mock providers and Franka Panda tabletop scenario",
        "num_runs": num_runs,
        "statistics": stats,
        "raw_runs": raw_runs,
    }


# ---------------------------------------------------------------------------
# Benchmark B: Local Laya Benchmark
# ---------------------------------------------------------------------------
def run_benchmark_b_laya(num_queries: int = 20) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f" BENCHMARK B: LOCAL LAYA BENCHMARK ({num_queries} queries)")
    print("=" * 80)

    laya_provider = LayaDecisionProvider(LayaConfig.from_env())
    if not laya_provider.is_available():
        raise RuntimeError("Local Laya server is not available at configured URL.")

    ctx = DecisionContext(
        context_id="ctx_bench_b",
        task_objective="Move the red object to the blue target.",
        robot_state_summary={"holding_object_id": None},
    )

    query_templates = [
        (["APPROACH", "STOP"], "Which action should be executed to navigate the gripper toward the object?"),
        (["GRASP", "STOP"], "Which action should be executed to grasp the object?"),
        (["REPOSITION", "STOP"], "Which recovery action should be considered to clear the obstacle?"),
        (["MOVE", "STOP"], "Which action should be executed to transport the payload to the target zone?"),
        (["RELEASE", "STOP"], "Which action should be executed to release the object in the zone?"),
    ]

    raw_queries: List[Dict[str, float]] = []
    tau_jev_values: List[float] = []

    for i in range(1, num_queries + 1):
        opts, prompt = query_templates[(i - 1) % len(query_templates)]
        q = DecisionQuestion(
            question_id=f"q_bench_b_{i}",
            question_type=DecisionQuestionType.RECOVERY_SELECTION,
            question_text=prompt,
            available_options=opts,
            context=ctx,
        )

        t0 = time.perf_counter()
        decision = laya_provider.ask(q)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        tau_jev_values.append(elapsed_ms)
        sample = {
            "query_index": i,
            "question_type": opts[0],
            "selected_option": decision.selected_option,
            "tau_jev_ms": round(elapsed_ms, 3),
            "reported_latency_ms": decision.latency_ms,
        }
        raw_queries.append(sample)
        if i <= 5 or i % 5 == 0:
            print(f"  Query {i:2d}/{num_queries}: {opts[0]:10s} -> {decision.selected_option:10s} | tau_jev={elapsed_ms:.2f}ms")

    stats = compute_metric_statistics(tau_jev_values)
    return {
        "benchmark_name": "Benchmark_B_Local_Laya",
        "description": "Direct latency measurement of local Laya server across 20 bounded decisions",
        "num_runs": num_queries,
        "statistics": {"tau_jev_ms": stats},
        "raw_runs": raw_queries,
    }


# ---------------------------------------------------------------------------
# Benchmark C: Full Live V3 Benchmark
# ---------------------------------------------------------------------------
def run_benchmark_c_live(num_runs: int = 5) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f" BENCHMARK C: FULL LIVE V3 BENCHMARK ({num_runs} runs)")
    print("=" * 80)

    nebius_provider = NebiusNemotronProvider()
    laya_config = LayaConfig.from_env(timeout=10.0)
    laya_provider = LayaDecisionProvider(laya_config)

    raw_runs: List[Dict[str, float]] = []

    for i in range(1, num_runs + 1):
        runtime = SarthiMuJoCoRuntime()
        service = V3OrchestrationService(
            adapter=runtime,
            engine=SarthiDecisionEngine(),
            candidate_injector=CandidateActionInjector(),
            context_builder=DecisionContextBuilder(),
            model_provider=nebius_provider,
            laya_provider=laya_provider,
            max_steps=10,
        )

        def disturbance_cb(step_num: int, ws: WorldState):
            if step_num == 2:
                runtime.inject_disturbance("PATH_BLOCKED")

        print(f"\n  --- Live Cycle {i}/{num_runs} starting... ---")
        telemetry = service.run_instruction(
            "Move the red object to the blue target.",
            step_callback=disturbance_cb,
        )

        # Physical placement verification check
        sr = runtime.state_reader
        final_obj = sr.get_red_object_position()
        tgt = sr.get_blue_target_position()
        tol = sr.get_blue_target_tolerance()
        h_dist = math.sqrt((final_obj.x - tgt.x)**2 + (final_obj.y - tgt.y)**2)
        placement_ok = h_dist <= tol

        lat = telemetry.latency
        raw_runs.append(lat)
        print(f"  Live Cycle {i}/{num_runs} Completed: Placement error={h_dist:.4f}m <= {tol:.4f}m ({'PASS' if placement_ok else 'FAIL'})")
        print(f"    tau_total: {lat.get('tau_total_ms', 0):.2f}ms | tau_nemotron: {lat.get('tau_nemotron_ms', 0):.2f}ms | tau_jev: {lat.get('tau_jev_ms', 0):.2f}ms | tau_sim: {lat.get('tau_sim_ms', 0):.2f}ms")

    stats = aggregate_latency_samples(raw_runs)
    return {
        "benchmark_name": "Benchmark_C_Full_Live_V3",
        "description": "Full cognitive-to-physical closed-loop benchmark with real Nemotron, real local Laya, and Franka Panda MuJoCo recovery",
        "num_runs": num_runs,
        "statistics": stats,
        "raw_runs": raw_runs,
    }


def main():
    print("=" * 80)
    print(" SĀRTHI V3-7: DECISION LATENCY INSTRUMENTATION & BENCHMARKING")
    print("=" * 80)

    # 1. Run Benchmark A: Offline
    bench_a = run_benchmark_a_offline(num_runs=5)

    # 2. Run Benchmark B: Local Laya
    bench_b = run_benchmark_b_laya(num_queries=20)

    # 3. Run Benchmark C: Full Live V3
    bench_c = run_benchmark_c_live(num_runs=5)

    # 4. Assemble Comprehensive Benchmark Record
    full_benchmark_record = {
        "project": "SĀRTHI",
        "phase": "V3-7",
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "hardware_environment": {
            "os": "Windows 11",
            "cpu": "Intel / AMD x86_64",
            "python_version": sys.version.split()[0],
            "simulator": "MuJoCo 3.14.0 (CPU / RK4)",
            "local_laya_url": "http://127.0.0.1:8000",
            "local_laya_model": "typed-decisions",
            "nemotron_endpoint": "https://api.tokenfactory.nebius.com/v1",
            "nemotron_model": "nvidia/Nemotron-3-Ultra-550b-a55b",
        },
        "benchmarks": {
            "benchmark_a_offline": bench_a,
            "benchmark_b_local_laya": bench_b,
            "benchmark_c_full_live_v3": bench_c,
        },
    }

    # 5. Save to records/v3_7_latency_benchmark.json
    output_path = _PROJECT_ROOT / "records" / "v3_7_latency_benchmark.json"
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_benchmark_record, f, indent=2)

    print("\n" + "=" * 80)
    print(f" Benchmark results written to: {output_path}")
    print(f" File size: {os.path.getsize(output_path)} bytes")

    # Verify zero secrets in benchmark file
    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()
    for token in ["NEBIUS_API_KEY", "JEV_API_KEY", "Bearer", "sk-", "nbf_"]:
        if token.lower() in content.lower():
            raise RuntimeError(f"CRITICAL: Secret token '{token}' detected in benchmark record!")
    print(" Secret Sanitization Check: PASS (Zero secrets detected)")
    print("=" * 80)


if __name__ == "__main__":
    main()
