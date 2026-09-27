#!/usr/bin/env python3
"""
SĀRTHI MuJoCo End-to-End Closed-Loop Integration + Disturbance Recovery Runner.

Executes the full SĀRTHI orchestration pipeline:
  Natural Language
  -> TaskUnderstandingService (via live Nebius Token Factory Nemotron)
  -> Canonical WorldState (observed from live MuJoCo mjData)
  -> SarthiDecisionEngine (deterministic candidate action evaluation)
  -> SarthiMuJoCoRuntime (SimulationAdapter contract & physical Panda physics)
  -> Dynamic disturbance injection (PATH_BLOCKED post-grasp)
  -> Autonomous recovery replanning (Decision Engine selects REPOSITION)
  -> Transit & placement (MOVE -> RELEASE)
  -> Ground-truth physical verification (measured object coords vs target tolerance)

Strict Invariants:
- Zero cognitive or replanning logic inside simulation/articulation/runtime.
- SarthiDecisionEngine is the sole physical action authority.
- No hardcoded recovery shortcuts ("if blocked: action = REPOSITION").
- Zero secret or API-key leakage in telemetry, events, or stdout.
- Physical success is strictly evaluated from MuJoCo physics measurements.
"""

import argparse
import math
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

# Load environment credentials securely without echoing
load_dotenv(os.path.join(_PROJECT_ROOT, "configs", ".env"))

from backend.app.decision_engine.engine import SarthiDecisionEngine
from backend.app.decision_engine.models import (
    ActionType,
    CandidateAction,
    CandidateEvaluation,
    Decision,
    ObjectState,
    Point3D,
    WorldState,
)
from backend.app.model.config import NebiusConfig
from backend.app.model.nebius_provider import (
    NebiusNemotronProvider,
    NebiusProviderError,
    _sanitize_error_message,
)
from backend.app.model.task_understanding import TaskUnderstandingService
from backend.app.orchestration.execution_result import TaskExecutionResult
from backend.app.orchestration.loop import TaskRunStatus
from backend.app.orchestration.task_runner import SarthiTaskRunner
from simulation.mujoco_runtime import is_mujoco_available, require_mujoco
from simulation.mujoco_runtime.runtime import SarthiMuJoCoRuntime
from simulation.mujoco_runtime.state_reader import SarthiMuJoCoStateReader


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SĀRTHI MuJoCo End-to-End Autonomous Closed-Loop Runner"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Move the red object to the blue target.",
        help="Natural language task instruction",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run simulation in headless mode (default: True)",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        default=False,
        help="Launch interactive MuJoCo passive viewer window (if supported)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum closed-loop execution steps",
    )
    parser.add_argument(
        "--steps-per-action",
        type=int,
        default=400,
        help="Simulation ticks per action execution",
    )
    return parser.parse_args()


def print_banner(text: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n  {text}\n{line}")


def print_section(title: str) -> None:
    print(f"\n--- {title} ---")


def main() -> None:
    args = parse_arguments()

    print_banner("SĀRTHI MUJOCO END-TO-END CLOSED-LOOP ORCHESTRATION")
    print(f"Instruction:        \"{args.instruction}\"")
    print(f"Mode:               {'Interactive Viewer' if args.viewer else 'Headless'}")
    print(f"Max Steps:          {args.max_steps}")
    print(f"Steps Per Action:   {args.steps_per_action}")

    # 1. Runtime & Physics Engine Verification
    if not is_mujoco_available():
        print("ERROR: MuJoCo runtime is not installed or available.", file=sys.stderr)
        sys.exit(1)

    # 2. Live Nebius Token Factory Provider Initialization
    print_section("Phase 1: Cognitive Task Understanding via Live Nebius Nemotron")
    nebius_cfg = NebiusConfig.from_env()
    print(f"Configured Model:   {nebius_cfg.model}")
    print(f"Inference Endpoint: {nebius_cfg.base_url}")

    try:
        provider = NebiusNemotronProvider()
        print("Connecting to live Nebius Token Factory endpoint...")
        service = TaskUnderstandingService(provider)
        start_cognition = time.time()
        tu = service.understand(args.instruction)
        cognition_time = time.time() - start_cognition
        print(f"Cognitive Parsing Succeeded in {cognition_time:.2f}s:")
        print(f"  Task ID:           {tu.task_id}")
        print(f"  Objective:         {tu.objective}")
        print(f"  Target Entity:     {tu.target_object}")
        print(f"  Destination Zone:  {tu.target_location}")
        print(f"  Intended Actions:  {tu.required_actions}")
        print(f"  Confidence:        {tu.confidence:.2f}")
        print(f"  Reasoning:         {tu.reasoning_summary}")
    except NebiusProviderError as err:
        sanitized = _sanitize_error_message(str(err))
        print(f"\nERROR: Live Nebius Nemotron Provider Failed: {sanitized}", file=sys.stderr)
        print("Note: Silent fallback to MockModelProvider is strictly prohibited.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        sanitized = _sanitize_error_message(str(exc))
        print(f"\nERROR: Unexpected failure contacting Nebius API: {sanitized}", file=sys.stderr)
        sys.exit(1)

    # 3. MuJoCo Physical Simulation Initialization
    print_section("Phase 2: MuJoCo Physical Simulation Setup")
    runtime = SarthiMuJoCoRuntime(
        config={
            "steps_per_action": args.steps_per_action,
            "convergence_tolerance": 0.03,
        }
    )
    sr: SarthiMuJoCoStateReader = runtime.state_reader
    initial_ws = runtime.get_world_state()

    print(f"Scene Model:        Franka Emika Panda (7-DOF + Parallel Gripper)")
    print(f"Initial Robot Pos:  x={initial_ws.robot.position.x:.3f}, y={initial_ws.robot.position.y:.3f}, z={initial_ws.robot.position.z:.3f}")
    target_obj = next((o for o in initial_ws.objects if o.is_target), None)
    print(f"Target Object Pos:  x={target_obj.position.x:.3f}, y={target_obj.position.y:.3f}, z={target_obj.position.z:.3f} (mass: {target_obj.mass_kg}kg)")
    print(f"Destination Target: x={initial_ws.target.position.x:.3f}, y={initial_ws.target.position.y:.3f}, z={initial_ws.target.position.z:.3f} (tol: {initial_ws.target.tolerance_radius_m:.3f}m)")

    # 4. Orchestration Stack Assembly
    engine = SarthiDecisionEngine()
    runner = SarthiTaskRunner(adapter=runtime, engine=engine, max_steps=args.max_steps)

    # Telemetry collectors
    step_telemetry: List[Dict[str, Any]] = []
    causal_trace: Dict[str, Any] = {
        "disturbance_injected": False,
        "disturbed_step": None,
        "pre_disturbance_ws_version": None,
        "post_disturbance_ws_version": None,
        "rejected_blocked_candidates": [],
        "recovery_action_selected": None,
        "recovery_executed_success": False,
    }

    # 5. Disturbance Callback Hook
    def on_step_executed(step_num: int, ws_after_action: WorldState) -> None:
        last_action_res = runtime.execution_history[-1] if runtime.execution_history else None
        record = {
            "step": step_num,
            "action_type": last_action_res.action_type if last_action_res else "UNKNOWN",
            "action_id": last_action_res.action_id if last_action_res else "UNKNOWN",
            "success": last_action_res.success if last_action_res else False,
            "failure_reason": last_action_res.failure_reason if last_action_res else None,
            "robot_pos": ws_after_action.robot.position,
            "gripper_open": ws_after_action.robot.gripper_open,
            "holding_object": ws_after_action.robot.holding_object_id,
            "object_pos": sr.get_red_object_position(),
            "object_state": sr.determine_object_state(),
            "active_constraints": len(ws_after_action.active_constraints),
            "world_state_version": ws_after_action.version,
        }
        step_telemetry.append(record)

        print(f"\n>>> [STEP {step_num}] Executed: {record['action_type']} | Success: {record['success']} | WS Ver: {record['world_state_version']}")
        print(f"    Robot EE:     ({record['robot_pos'].x:.3f}, {record['robot_pos'].y:.3f}, {record['robot_pos'].z:.3f}) | Holding: {record['holding_object']}")
        print(f"    Object Pos:   ({record['object_pos'].x:.3f}, {record['object_pos'].y:.3f}, {record['object_pos'].z:.3f}) | State: {record['object_state']}")

        # Disturbance injection condition: immediately after object is grasped (Step 2)
        if step_num == 2 and not causal_trace["disturbance_injected"]:
            print("\n" + "!" * 78)
            print("!!! CAUSAL DISTURBANCE TRIGGER: Injecting 'PATH_BLOCKED' into MuJoCo physics !!!")
            print("!" * 78)

            causal_trace["disturbance_injected"] = True
            causal_trace["disturbed_step"] = step_num
            causal_trace["pre_disturbance_ws_version"] = ws_after_action.version

            # Inject physical obstacle into MuJoCo scene
            injected = runtime.inject_disturbance("PATH_BLOCKED")
            disturbed_ws = runtime.get_world_state()
            causal_trace["post_disturbance_ws_version"] = disturbed_ws.version

            obs = next((o for o in disturbed_ws.objects if o.is_obstacle), None)
            print(f"Disturbance Activated: {injected}")
            print(f"  Physical Obstacle Coords: ({obs.position.x:.3f}, {obs.position.y:.3f}, {obs.position.z:.3f})")
            print(f"  New WorldState Version:   {disturbed_ws.version}")
            print(f"  Active Constraints:       {[c.constraint_id for c in disturbed_ws.active_constraints]}")

            # Inspect what Decision Engine will evaluate on next tick
            peek_decision = engine.decide(disturbed_ws)
            for cand in peek_decision.candidate_evaluations:
                if not cand.is_valid:
                    reasons = "; ".join(cand.rejection_reasons)
                    causal_trace["rejected_blocked_candidates"].append({
                        "action_id": cand.action.action_id,
                        "action_type": cand.action.action_type.value,
                        "rejection_reasons": cand.rejection_reasons,
                    })
                    print(f"  Constraint Rejection: {cand.action.action_id} ({cand.action.action_type.value}) REJECTED: {reasons}")

            causal_trace["recovery_action_selected"] = peek_decision.selected_action.action_type.value
            print(f"  Autonomous Recovery Selection: {causal_trace['recovery_action_selected']} (Score: {peek_decision.decision_factors.get('composite_score', 0.0):.3f})")

    # 6. Execute Closed-Loop Orchestration
    print_section("Phase 3: Autonomous Closed-Loop Execution")
    start_exec = time.time()
    task_result: TaskExecutionResult = runner.run_instruction(
        args.instruction,
        model_provider=provider,
        step_callback=on_step_executed,
    )
    total_exec_time = time.time() - start_exec

    # 7. Check if recovery action was actually dispatched and executed
    repo_results = [r for r in runtime.execution_history if r.action_type == "REPOSITION"]
    if repo_results and repo_results[0].success:
        causal_trace["recovery_executed_success"] = True

    # 8. Physical Ground-Truth Measurements
    print_section("Phase 4: Physical Ground-Truth Verification (MuJoCo Measurements)")
    final_obj_pos = sr.get_red_object_position()
    tgt_pos = sr.get_blue_target_position()
    tol = sr.get_blue_target_tolerance()

    horizontal_dist = math.sqrt(
        (final_obj_pos.x - tgt_pos.x) ** 2 + (final_obj_pos.y - tgt_pos.y) ** 2
    )
    vertical_offset = abs(final_obj_pos.z - (tgt_pos.z + 0.03))
    is_at_tabletop = vertical_offset <= 0.035
    within_tolerance = horizontal_dist <= tol
    is_released = not sr.is_object_grasped() and sr.is_gripper_open()
    final_obj_state = sr.determine_object_state()

    print(f"Final Red Object Pos:   x={final_obj_pos.x:.4f}, y={final_obj_pos.y:.4f}, z={final_obj_pos.z:.4f}")
    print(f"Target Zone Center:     x={tgt_pos.x:.4f}, y={tgt_pos.y:.4f}, z={tgt_pos.z:.4f}")
    print(f"Placement Error (H):    {horizontal_dist:.4f} m (Acceptable: <= {tol:.4f} m) -> {'PASS' if within_tolerance else 'FAIL'}")
    print(f"Tabletop Elevation:     {vertical_offset:.4f} m (Acceptable: <= 0.035 m) -> {'PASS' if is_at_tabletop else 'FAIL'}")
    print(f"Gripper Open/Released:  {is_released} -> {'PASS' if is_released else 'FAIL'}")
    print(f"Canonical Object State: {final_obj_state} -> {'PASS' if final_obj_state == ObjectState.PLACED else 'FAIL'}")

    # 9. Verify 11 Physical Success Criteria
    criteria = {
        "1. Robot physically approached object": any(r["action_type"] == "APPROACH" and r["success"] for r in step_telemetry),
        "2. Gripper physically grasped object": any(r["action_type"] == "GRASP" and r["success"] for r in step_telemetry),
        "3. Disturbance occurred after grasp": causal_trace["disturbance_injected"],
        "4. Decision Engine observed changed WorldState": causal_trace["post_disturbance_ws_version"] is not None and causal_trace["post_disturbance_ws_version"] > causal_trace["pre_disturbance_ws_version"],
        "5. Nominal blocked path rejected": len(causal_trace["rejected_blocked_candidates"]) > 0,
        "6. Recovery action selected by Decision Engine": causal_trace["recovery_action_selected"] == "REPOSITION",
        "7. Robot physically executed recovery": causal_trace["recovery_executed_success"],
        "8. Robot reached target zone": any(r["action_type"] == "MOVE" and r["success"] for r in step_telemetry[2:]),
        "9. Object physically within target tolerance": within_tolerance,
        "10. Gripper physically released object": is_released,
        "11. Final verification succeeded": final_obj_state == ObjectState.PLACED and task_result.completed,
    }

    print_section("Phase 5: Physical Success Criteria Evaluation")
    all_passed = True
    for desc, passed in criteria.items():
        status_str = "[OK]  " if passed else "[FAIL]"
        print(f"  {status_str} {desc}")
        if not passed:
            all_passed = False

    # 10. Structured Telemetry Summary
    print_section("Phase 6: Structured End-to-End Telemetry Summary")
    print(f"Initial Instruction:     \"{args.instruction}\"")
    print(f"Nemotron Model Used:     {nebius_cfg.model}")
    print(f"Nemotron Response Conf:  {tu.confidence:.2f}")
    print(f"Executed Actions:        {task_result.executed_actions}")
    print(f"Adaptive Recovery Count: {task_result.recovery_count}")
    print(f"Total Execution Steps:   {len(step_telemetry)}")
    print(f"Total Physics Time:      {runtime.simulation_time:.2f} s")
    print(f"Wall Clock Time:         {total_exec_time:.2f} s")
    print(f"Task Terminal Status:    {task_result.run_status}")
    print(f"Causal Narrative:        PATH_BLOCKED changed physical WorldState "
          f"(v{causal_trace['pre_disturbance_ws_version']} -> v{causal_trace['post_disturbance_ws_version']}) "
          f"-> Decision Engine rejected {[c['action_id'] for c in causal_trace['rejected_blocked_candidates']]} "
          f"-> Selected {causal_trace['recovery_action_selected']} "
          f"-> Panda executed recovery -> Completed to {final_obj_state}.")

    # 11. Process Exit Code
    if all_passed and task_result.completed:
        print_banner("E2E VALIDATION RESULT: SUCCESS (All 11 Physical Criteria Satisfied)")
        sys.exit(0)
    else:
        print_banner("E2E VALIDATION RESULT: FAILED (Physical Success Criteria Not Satisfied)")
        sys.exit(1)


if __name__ == "__main__":
    main()
