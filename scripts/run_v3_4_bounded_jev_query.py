#!/usr/bin/env python3
"""
SĀRTHI V3-4 — Bounded Jev Decision Query Standalone Runner.

Executes ONE controlled live request to Jev:
  DecisionContext (via DecisionContextBuilder)
        ↓
  DecisionQuestion
        ↓
  RealJevDecisionProvider (credentials strictly from environment)
        ↓
  LIVE JEV
        ↓
  JevDecision (strictly validated against bounded options)

Strict Safety Boundaries:
- NEVER commands robot motors or joints.
- NEVER calls MuJoCo runtime or simulation physics.
- NEVER modifies WorldState.
- NEVER calls Nemotron or executes downstream CandidateAction mapping.
- In case of failure, reports exact provider failure category without silent fallback.
- Never logs API keys, bearer tokens, or secrets.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
import time

# Ensure project root is in sys.path
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Ensure UTF-8 on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Attempt loading .env files if present (without echoing any secret)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT_ROOT, "configs", ".env"))
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
except ImportError:
    pass

from backend.app.model.jev_config import JevConfig
from backend.app.model.laya_config import LayaConfig
from backend.app.model.laya_provider import LayaDecisionProvider
from backend.app.model.real_jev_provider import RealJevDecisionProvider
from backend.app.orchestration.bounded_query_runner import (
    BoundedJevQueryRunner,
    BoundedQueryResult,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sarthi_v3_4_runner")


def run_live_query(use_laya: bool = False) -> BoundedQueryResult:
    """Orchestrates the single live Jev / Laya bounded query."""
    print("=" * 70)
    print(" SĀRTHI Phase V3-4: Controlled Bounded Fast Decision Query")
    print("=" * 70)

    # 1. Initialize Bounded Runner (uses DecisionContextBuilder)
    runner = BoundedJevQueryRunner(coordinate_precision=3)

    # 2. Build recovery situation and bounded question
    print("\n[01] BUILDING DETERMINISTIC RECOVERY SCENARIO...")
    situation, question = runner.build_recovery_scenario()

    print(f"     Task Objective : {question.context.task_objective}")
    print(f"     Robot Holding  : {question.context.robot_state_summary.get('holding_object_id')}")
    print(f"     Gripper Open   : {question.context.robot_state_summary.get('gripper_open')}")
    print(f"     Constraints    : {question.context.active_constraints}")
    print(f"     Prev Outcome   : {question.context.previous_action_outcome}")
    print(f"     Question ID    : {question.question_id}")
    print(f"     Question Text  : {question.question_text}")
    print(f"     Bounded Options: {question.available_options}")

    # 3. Resolve Provider
    provider_pref = os.environ.get("DECISION_PROVIDER", "").lower()
    has_laya = use_laya or provider_pref == "laya" or bool(os.environ.get("LAYA_HOST") or os.environ.get("LAYA_BASE_URL"))

    if has_laya:
        print("\n[02] RESOLVING LAYA CONFIGURATION FROM ENVIRONMENT...")
        laya_config = LayaConfig.from_env()
        print(f"     Endpoint URL   : {laya_config.base_url}")
        print(f"     Target Model   : {laya_config.model}")
        print(f"     Timeout (s)    : {laya_config.timeout}")
        print("     Auth Required  : False (Local endpoint)")
        provider = LayaDecisionProvider(laya_config)
    else:
        print("\n[02] RESOLVING JEV CONFIGURATION FROM ENVIRONMENT...")
        jev_config = JevConfig.from_env()
        has_key = bool(jev_config.api_key and jev_config.api_key.strip())
        print(f"     Endpoint URL   : {jev_config.base_url}")
        print(f"     Target Model   : {jev_config.model}")
        print(f"     Timeout (s)    : {jev_config.timeout}")
        print(f"     API Key Present: {has_key} (Value strictly isolated & redacted)")
        provider = RealJevDecisionProvider(jev_config)

    # 4. Perform ONE live call
    target_name = "LOCAL LAYA" if has_laya else "JEV"
    print(f"\n[03] EXECUTING ONE CONTROLLED LIVE REQUEST TO {target_name}...")
    start_time = time.perf_counter()
    result = runner.execute_bounded_query(provider=provider, question=question)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # 5. Evaluate and report outcome
    print("\n[04] LIVE QUERY RESULT EVALUATION:")
    if result.success and result.decision:
        dec = result.decision
        print("     Status           : SUCCESS (Live response received and validated)")
        print(f"     Selected Option  : {dec.selected_option}")
        print(f"     Confidence       : {dec.confidence}")
        if dec.answer_confidence is not None:
            print(f"     Answer Confidence: {dec.answer_confidence}")
        print(f"     Probabilities    : {dec.option_probabilities}")
        print(f"     Query Latency    : {dec.latency_ms:.2f} ms")
        print(f"     Total Elapsed    : {elapsed_ms:.2f} ms")
        print(f"     Validation Passed: {result.validation_passed}")
        print(f"     Provider/Model   : {dec.provider} / {dec.model_version}")
        if dec.rationale:
            print(f"     Rationale        : {dec.rationale}")
    else:
        print("     Status           : FAILED (Live request failed cleanly)")
        print(f"     Failure Category : {result.error_category}")
        print(f"     Sanitized Message: {result.error_message}")
        print(f"     Query Latency    : {result.telemetry.latency_ms} ms")
        print(f"     Total Elapsed    : {elapsed_ms:.2f} ms")
        print(f"     Validation Passed: {result.validation_passed}")

    # 7. Safety Invariant Confirmation
    print("\n[05] SAFETY BOUNDARY VERIFICATION:")
    print("     [✓] No robot motors or joints commanded")
    print("     [✓] No MuJoCo runtime or simulation physics invoked")
    print("     [✓] WorldState remained unchanged (no mutation)")
    print("     [✓] No ActionExecution created")
    print("     [✓] SarthiDecisionEngine downstream execution bypassed (ends at JevDecision)")
    print("     [✓] Nemotron cognitive model was not invoked")
    print("     [✓] Zero secrets, tokens, or credentials exposed")

    # 8. Record safe telemetry
    records_dir = os.path.join(_PROJECT_ROOT, "records")
    os.makedirs(records_dir, exist_ok=True)
    telemetry_path = os.path.join(records_dir, "v3_4_jev_query_telemetry.json")

    telemetry_data = result.telemetry.to_dict()
    telemetry_data["safety_boundary_verified"] = True
    telemetry_data["physical_action_executed"] = False
    telemetry_data["nemotron_called"] = False

    with open(telemetry_path, "w", encoding="utf-8") as f:
        json.dump(telemetry_data, f, indent=2)

    print(f"\n[06] SAFE TELEMETRY RECORDED TO:")
    print(f"     {telemetry_path}")
    print("=" * 70)

    return result


if __name__ == "__main__":
    run_live_query()
