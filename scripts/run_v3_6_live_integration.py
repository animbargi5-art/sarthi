#!/usr/bin/env python3
"""
SĀRTHI V3-6 — Human Command + Physical Situation End-to-End Live Runner.

Executes ONE controlled live V3-6 integration run using:
  - Real NVIDIA Nemotron (Nebius Token Factory)
  - Real Local Laya Server (http://127.0.0.1:8000)
  - Canonical CandidateActionInjector
  - Authoritative SarthiDecisionEngine
  - Existing MuJoCo Tabletop Pick-and-Place (tabletop_pick_and_place_mvp) with Franka Panda
  - PATH_BLOCKED disturbance injection and deterministic REPOSITION recovery

Architectural Invariants Strictly Enforced:
1. Nemotron performs cognitive task understanding only; zero motor/execution access.
2. Laya produces fast bounded candidate proposals; zero motor/execution access.
3. CandidateActionInjector safely translates Laya decisions without authority.
4. SarthiDecisionEngine remains the sole deterministic physical authority.
5. All AI proposals pass through physical constraint validation.
6. Secrets and API keys are strictly excluded from telemetry, logs, and output.
"""

from __future__ import annotations

import json
import logging
import math
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

# Load environment configuration (safely, without echoing any secret)
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
from backend.app.model.nebius_provider import NebiusNemotronProvider
from backend.app.orchestration.context_builder import DecisionContextBuilder
from backend.app.orchestration.v3_orchestrator import (
    V3ExecutionTelemetry,
    V3OrchestrationService,
)
from simulation.mujoco_runtime.runtime import SarthiMuJoCoRuntime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sarthi_v3_6_live_runner")


def run_live_v3_6_integration(
    telemetry_output_path: str = "records/v3_6_e2e_telemetry.json",
) -> V3ExecutionTelemetry:
    """Runs the controlled live V3-6 end-to-end integration."""
    print("=" * 80)
    print(" SĀRTHI V3-6: HUMAN COMMAND + PHYSICAL SITUATION END-TO-END INTEGRATION")
    print("=" * 80)

    # 1. Initialize Real Live Providers
    print("\n[INIT] Initializing Live AI Providers & Physical MuJoCo Runtime...")

    # Real NVIDIA Nemotron via Nebius Token Factory
    nebius_provider = NebiusNemotronProvider()
    print(f"  - Model Provider: Real NebiusNemotronProvider ({nebius_provider.model_name})")

    # Real Local Laya Server
    laya_config = LayaConfig.from_env()
    laya_provider = LayaDecisionProvider(laya_config)
    print(f"  - Fast Decision Provider: Real Local LayaDecisionProvider ({laya_config.base_url}, model={laya_config.model})")

    # MuJoCo Runtime with Franka Panda tabletop scenario
    runtime = SarthiMuJoCoRuntime()
    print("  - Physical Simulation: SarthiMuJoCoRuntime (tabletop_pick_and_place_mvp.xml)")

    # SĀRTHI Decision Engine (Deterministic physical authority)
    decision_engine = SarthiDecisionEngine()
    print("  - Physical Authority: SarthiDecisionEngine with ConstraintValidator")

    # Candidate Action Injector
    candidate_injector = CandidateActionInjector()
    print("  - Action Translation: CandidateActionInjector (Bounded JevDecision -> CandidateAction)")

    # V3 Orchestration Service
    service = V3OrchestrationService(
        model_provider=nebius_provider,
        laya_provider=laya_provider,
        sim_adapter=runtime,
        decision_engine=decision_engine,
        candidate_injector=candidate_injector,
        context_builder=DecisionContextBuilder(),
        max_steps=10,
    )

    # 2. Inspect Initial Physical Environment
    initial_ws = runtime.get_world_state()
    print(f"\n[SCENARIO] Tabletop Scene Initialized:")
    print(f"  - Robot EE: ({initial_ws.robot.position.x:.3f}, {initial_ws.robot.position.y:.3f}, {initial_ws.robot.position.z:.3f})")
    target_obj = next((o for o in initial_ws.objects if o.is_target), None)
    if target_obj:
        print(f"  - Red Object: '{target_obj.id}' at ({target_obj.position.x:.3f}, {target_obj.position.y:.3f}, {target_obj.position.z:.3f})")
    print(f"  - Blue Target Zone: '{initial_ws.target.id}' at ({initial_ws.target.position.x:.3f}, {initial_ws.target.position.y:.3f}, {initial_ws.target.position.z:.3f}), tol={initial_ws.target.tolerance_radius_m}m")

    # 3. Define Disturbance Callback (PATH_BLOCKED injected post-grasp)
    def on_step_callback(step_num: int, ws: WorldState):
        if step_num == 2:
            print("\n>>> [DISTURBANCE INJECTION] Injecting dynamic PATH_BLOCKED obstacle into MuJoCo world...")
            runtime.inject_disturbance("PATH_BLOCKED")
            print(">>> [DISTURBANCE INJECTED] Barrier placed between grasp position and target zone.")

    # 4. Human Instruction
    human_instruction = "Move the red object to the blue target."
    print(f"\n[HUMAN INSTRUCTION] \"{human_instruction}\"")

    # 5. Execute End-to-End Instruction
    abs_telemetry_path = str(_PROJECT_ROOT / telemetry_output_path)
    os.makedirs(os.path.dirname(abs_telemetry_path), exist_ok=True)

    print("\n[ORCHESTRATION] Executing closed-loop V3 pipeline...")
    telemetry = service.run_instruction(
        instruction=human_instruction,
        step_callback=on_step_callback,
        output_telemetry_path=abs_telemetry_path,
    )

    # 6. Print Step-by-Step Audit
    print("\n" + "=" * 80)
    print(" SĀRTHI V3-6 EXECUTION AUDIT SUMMARY")
    print("=" * 80)

    print(f"\nTask Understanding (NVIDIA Nemotron):")
    print(f"  - Task ID: {telemetry.task_understanding.get('task_id')}")
    print(f"  - Objective: {telemetry.task_understanding.get('objective')}")
    print(f"  - Target Object: {telemetry.task_understanding.get('target_object')}")
    print(f"  - Target Location: {telemetry.task_understanding.get('target_location')}")
    print(f"  - Required Actions: {telemetry.task_understanding.get('required_actions')}")
    print(f"  - Nemotron Confidence: {telemetry.task_understanding.get('confidence')}")

    print(f"\nClosed-Loop Execution Steps ({len(telemetry.steps)} total):")
    for step in telemetry.steps:
        print(f"\n  --- Step {step.step_number} ---")
        print(f"  Question: [{step.question_id}] {step.question_text}")
        print(f"  Available Options: {step.available_options}")
        print(f"  Laya Fast Decision: {step.selected_option} (Probabilities: {step.laya_probabilities}, Latency: {step.laya_latency_ms:.2f}ms)")
        print(f"  CandidateAction Injected: {step.candidate_action}")
        print(f"  Deterministic Validation: {step.deterministic_validation_result.upper()} (Action: {step.final_decision_action}, Score: {step.final_decision_score:.3f})")
        if step.rejection_reasons:
            print(f"  Rejection Reasons: {step.rejection_reasons}")
        print(f"  MuJoCo Execution Result: success={step.execution_success}, verified={step.verification_success}")

    print(f"\nRecovery Events: {telemetry.recovery_events}")
    print(f"Final Task Status: {telemetry.final_task_status}")
    print(f"Completed: {telemetry.completed}")

    # 7. Physical Outcome Verification
    sr = runtime.state_reader
    final_obj_pos = sr.get_red_object_position()
    tgt_pos = sr.get_blue_target_position()
    tol = sr.get_blue_target_tolerance()
    horiz_dist = math.sqrt((final_obj_pos.x - tgt_pos.x)**2 + (final_obj_pos.y - tgt_pos.y)**2)
    print(f"\n[PHYSICAL VERIFICATION]")
    print(f"  - Red Object Position: ({final_obj_pos.x:.4f}, {final_obj_pos.y:.4f}, {final_obj_pos.z:.4f})")
    print(f"  - Target Zone Center:  ({tgt_pos.x:.4f}, {tgt_pos.y:.4f}, {tgt_pos.z:.4f})")
    print(f"  - Placement Error:     {horiz_dist:.4f}m (Tolerance: <= {tol:.4f}m)")
    placement_ok = horiz_dist <= tol
    print(f"  - Tolerance Check:     {'PASS' if placement_ok else 'FAIL'}")

    print(f"\n[TELEMETRY] Written to: {abs_telemetry_path}")
    print(f"  - Telemetry File Size: {os.path.getsize(abs_telemetry_path)} bytes")

    # 8. Verify No Secrets in Telemetry
    with open(abs_telemetry_path, "r", encoding="utf-8") as f:
        content = f.read()
    for token in ["NEBIUS_API_KEY", "JEV_API_KEY", "Bearer", "sk-", "nbf_"]:
        if token.lower() in content.lower():
            raise RuntimeError(f"CRITICAL: Secret token '{token}' detected in telemetry!")
    print("  - Secret Sanitization Check: PASS (Zero secrets detected)")

    print("\n" + "=" * 80)
    print(f" SĀRTHI V3-6 LIVE INTEGRATION: {'SUCCESS' if telemetry.completed and placement_ok else 'FAILED'}")
    print("=" * 80)

    return telemetry


if __name__ == "__main__":
    run_live_v3_6_integration()
