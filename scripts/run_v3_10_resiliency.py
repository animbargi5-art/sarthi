#!/usr/bin/env python3
"""
SĀRTHI V3-10 — Failure & Resiliency Fallback Suite Runner.
Executes all 13 controlled failure scenarios and live validation benchmarks A, B, C, D.
Persists comprehensive secret-free audit telemetry to records/v3_10_resiliency.json.

Strict Invariants:
1. SarthiDecisionEngine remains the sole physical authority.
2. AI failures cannot cause physical execution.
3. Invalid candidates cannot execute.
4. Unsafe candidates are deterministically rejected.
5. All secrets, keys, and tokens are 100% sanitized.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
import time

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

from backend.app.resiliency.suite import ResiliencySuite

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sarthi_v3_10_runner")


def run_resiliency_validation(
    telemetry_output_path: str = "records/v3_10_resiliency.json",
) -> None:
    print("=" * 80)
    print(" SĀRTHI V3-10: FAILURE & RESILIENCY FALLBACK SUITE RUNNER")
    print("=" * 80)

    suite = ResiliencySuite()

    # 1. Run all 13 Controlled Failure Scenarios
    print("\n[PHASE 1] Executing 13 Controlled Failure Scenarios...")
    report = suite.run_all_scenarios(include_live=False)

    for i, s in enumerate(report.scenarios, 1):
        print(f"  [{i:02d}] {s.failure_type.value:35s} -> Strategy: {s.recovery_strategy.value:22s} | Stage: {s.detected_at_stage.value:22s} | Status: {s.final_status}")

    # 2. Run Live Validations A, B, C, D
    print("\n[PHASE 2] Executing Live Validations (MuJoCo + Real AI Models)...")

    print("  -> Live Validation A: Nominal V3 run (Real Nemotron + Real Laya + MuJoCo)...")
    live_a = suite.run_live_normal()
    print(f"     Status: {live_a['final_task_status']} | Placement Error: {live_a['placement_error_m']:.4f}m | Within Tol: {live_a['within_tolerance']}")

    print("  -> Live Validation B: PATH_BLOCKED recovery (Real Nemotron + Real Laya + MuJoCo + Obstacle)...")
    live_b = suite.run_live_path_blocked()
    print(f"     Status: {live_b['final_task_status']} | Placement Error: {live_b['placement_error_m']:.4f}m | Recovery: {live_b['recovery_occurred']}")

    print("  -> Live Validation C: Controlled Laya Failure Live (Real Nemotron + Failing Laya + Fallback)...")
    live_c = suite.run_live_laya_fallback()
    print(f"     Status: {live_c['final_task_status']} | Fallback Steps: {live_c['fallback_engaged_steps_count']} | Placement Error: {live_c['placement_error_m']:.4f}m")

    print("  -> Live Validation D: Live Deterministic Rejection (Unsafe Move Candidate Injection)...")
    live_d = suite.run_live_deterministic_rejection()
    print(f"     Rejected Unsafe Candidate: {live_d['unsafe_candidate_rejected']} | Selected Action: {live_d['selected_action']} | Safe Halt/Recovery: {live_d['safe_halt_or_recovery']}")

    # Merge live summary into report
    report.live_validation_summary = {
        "live_a_normal": live_a,
        "live_b_path_blocked": live_b,
        "live_c_laya_fallback": live_c,
        "live_d_deterministic_rejection": live_d,
    }

    # 3. Sanitize and Save Telemetry Record
    output_abs_path = os.path.join(_PROJECT_ROOT, telemetry_output_path)
    os.makedirs(os.path.dirname(output_abs_path), exist_ok=True)

    sanitized_report = report.to_sanitized_dict()

    with open(output_abs_path, "w", encoding="utf-8") as f:
        json.dump(sanitized_report, f, indent=2)

    print(f"\n[OUTPUT] Saved secret-free resiliency telemetry to: {output_abs_path}")

    # 4. Summary Output
    print("\n" + "=" * 80)
    print(" SĀRTHI V3-10 RESILIENCY & FALLBACK AUDIT SUMMARY")
    print("=" * 80)
    print(f"Total Scenarios Evaluated: {report.total_scenarios_evaluated}")
    print(f"Total Passed:              {report.total_passed}")
    print(f"Total Failed:              {report.total_failed}")
    print(f"Unsafe Executions:         {report.unsafe_executions_count} (MUST BE 0)")
    print(f"AI-to-Robot Bypasses:      {report.ai_to_robot_bypasses_count} (MUST BE 0)")
    print(f"Secret Sanitization:       {report.secret_sanitization_verified}")
    print(f"Conclusion:                {report.resiliency_conclusion}")
    print("=" * 80)


if __name__ == "__main__":
    run_resiliency_validation()
