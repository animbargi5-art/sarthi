#!/usr/bin/env python3
"""
SĀRTHI MuJoCo End-to-End Closed-Loop Integration + Disturbance Recovery Runner.

Executes the full SĀRTHI physical AI orchestration pipeline:
  Natural Language
  -> TaskUnderstandingService (via live Nebius Token Factory Nemotron or MockModelProvider)
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
import datetime
import json
import math
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
from dotenv import load_dotenv

# Load environment credentials securely without echoing
load_dotenv(os.path.join(_PROJECT_ROOT, "configs", ".env"))

import mujoco

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
from backend.app.model.mock_provider import MockModelProvider
from backend.app.model.models import TaskUnderstanding
from backend.app.model.nebius_provider import (
    NebiusNemotronProvider,
    NebiusProviderError,
    _sanitize_error_message,
)
from backend.app.model.provider import ModelProvider
from backend.app.model.task_understanding import TaskUnderstandingService
from backend.app.orchestration.execution_result import TaskExecutionResult
from backend.app.orchestration.loop import TaskRunStatus
from backend.app.orchestration.task_runner import SarthiTaskRunner
from simulation.mujoco_runtime import is_mujoco_available, require_mujoco
from simulation.mujoco_runtime.runtime import SarthiMuJoCoRuntime
from simulation.mujoco_runtime.state_reader import SarthiMuJoCoStateReader


class EventTraceRecorder:
    """
    Records and formats sequential, human-readable execution events
    derived strictly from live runtime and decision-engine evidence.
    """

    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def record(self, event_name: str, description: str = "", metadata: Optional[Dict[str, Any]] = None) -> None:
        idx = len(self.events) + 1
        event_str = f"[{idx:02d}] {event_name}"
        if description:
            event_str += f": {description}"
        self.events.append({
            "index": idx,
            "name": event_name,
            "description": description,
            "metadata": metadata or {},
            "formatted": event_str,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
        print(f"  {event_str}")

    def get_formatted_trace(self) -> List[str]:
        return [e["formatted"] for e in self.events]


def parse_arguments(cli_args: Optional[List[str]] = None) -> argparse.Namespace:
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
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic execution reproducibility",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Use deterministic MockModelProvider instead of live Nebius API",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="File path to write structured JSON telemetry report",
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
    return parser.parse_args(cli_args)


def sanitize_dict_for_telemetry(d: Any) -> Any:
    """Recursively removes any keys or values containing credential patterns."""
    forbidden = ["api_key", "bearer", "authorization", "secret", "token"]
    if isinstance(d, dict):
        clean = {}
        for k, v in d.items():
            if any(f in str(k).lower() for f in forbidden):
                continue
            clean[k] = sanitize_dict_for_telemetry(v)
        return clean
    elif isinstance(d, list):
        return [sanitize_dict_for_telemetry(x) for x in d]
    elif isinstance(d, str):
        return _sanitize_error_message(d)
    return d


def print_banner(text: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n  {text}\n{line}")


def print_section(title: str) -> None:
    print(f"\n--- {title} ---")


def execute_validation_run(args: argparse.Namespace) -> Tuple[bool, Dict[str, Any]]:
    """
    Executes the full SĀRTHI closed-loop MuJoCo validation run.
    Returns (success: bool, telemetry_dict: Dict[str, Any]).
    """
    trace = EventTraceRecorder()
    run_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # 1. Environment & Reproducibility Setup
    if args.seed is not None:
        np.random.seed(args.seed)

    trace.record("TASK_RECEIVED", args.instruction)

    # 2. Physics Engine Verification
    if not is_mujoco_available():
        raise RuntimeError("MuJoCo runtime is not installed or available.")

    # 3. Model Provider Setup
    provider_name: str
    model_name: str
    endpoint_url: str
    provider: ModelProvider

    if args.mock:
        provider = MockModelProvider()
        provider_name = "MockModelProvider"
        model_name = "mock/nemotron-deterministic"
        endpoint_url = "local://deterministic-mock"
    else:
        nebius_cfg = NebiusConfig.from_env()
        provider_name = "NebiusNemotronProvider"
        model_name = nebius_cfg.model or "nvidia/Nemotron-3-Ultra-550b-a55b"
        endpoint_url = nebius_cfg.base_url or "https://api.tokenfactory.nebius.com/v1"
        try:
            provider = NebiusNemotronProvider()
        except Exception as exc:
            sanitized = _sanitize_error_message(str(exc))
            print(f"\nERROR: Failed to initialize NebiusNemotronProvider: {sanitized}", file=sys.stderr)
            raise

    # 4. Cognitive Task Understanding
    service = TaskUnderstandingService(provider)
    start_cognition = time.time()
    try:
        tu: TaskUnderstanding = service.understand(args.instruction)
    except Exception as exc:
        sanitized = _sanitize_error_message(str(exc))
        print(f"\nERROR: Model Provider Failed to Understand Task: {sanitized}", file=sys.stderr)
        raise

    cognition_duration = time.time() - start_cognition
    trace.record(
        "NEMOTRON_TASK_UNDERSTANDING",
        f"target='{tu.target_object}', zone='{tu.target_location}', actions={tu.required_actions}",
        {"confidence": tu.confidence, "task_id": tu.task_id},
    )

    # 5. MuJoCo Physical Simulation Setup
    runtime = SarthiMuJoCoRuntime(
        config={
            "steps_per_action": args.steps_per_action,
            "convergence_tolerance": 0.03,
        }
    )
    sr: SarthiMuJoCoStateReader = runtime.state_reader
    initial_ws = runtime.get_world_state()

    # Optional passive viewer
    viewer = None
    if args.viewer:
        try:
            import mujoco.viewer as mj_viewer
            viewer = mj_viewer.launch_passive(runtime.model, runtime.data)
            viewer.sync()
            print("  MuJoCo passive viewer window launched.")
        except Exception as v_err:
            print(f"  Note: Passive viewer could not be launched ({v_err}). Continuing in headless mode.")
            viewer = None

    trace.record(
        "WORLD_STATE_OBSERVED",
        f"v{initial_ws.version}, Robot=({initial_ws.robot.position.x:.2f}, {initial_ws.robot.position.y:.2f}, {initial_ws.robot.position.z:.2f})",
        {"version": initial_ws.version},
    )

    # 6. Orchestration Setup
    engine = SarthiDecisionEngine()
    runner = SarthiTaskRunner(adapter=runtime, engine=engine, max_steps=args.max_steps)

    step_telemetry: List[Dict[str, Any]] = []
    candidate_evals_log: List[Dict[str, Any]] = []
    rejected_actions_log: List[Dict[str, Any]] = []
    selected_actions_log: List[Dict[str, Any]] = []

    causal_trace: Dict[str, Any] = {
        "disturbance_injected": False,
        "disturbed_step": None,
        "pre_disturbance_ws_version": None,
        "post_disturbance_ws_version": None,
        "obstacle_position": None,
        "rejected_blocked_candidates": [],
        "recovery_action_selected": None,
        "recovery_executed_success": False,
    }

    # Record initial decision
    initial_decision = engine.decide(initial_ws)
    trace.record(
        "DECISION_SELECTED",
        f"Selected {initial_decision.selected_action.action_type.value} ({initial_decision.selected_action.action_id})",
        {"action_type": initial_decision.selected_action.action_type.value},
    )

    # 7. Step Callback & Disturbance Injection Hook
    def on_step_executed(step_num: int, ws_after_action: WorldState) -> None:
        if viewer is not None:
            viewer.sync()

        last_action_res = runtime.execution_history[-1] if runtime.execution_history else None
        act_type = last_action_res.action_type if last_action_res else "UNKNOWN"
        act_id = last_action_res.action_id if last_action_res else "UNKNOWN"
        success = last_action_res.success if last_action_res else False

        selected_actions_log.append({"step": step_num, "action_type": act_type, "action_id": act_id, "success": success})

        record = {
            "step": step_num,
            "action_type": act_type,
            "action_id": act_id,
            "success": success,
            "failure_reason": last_action_res.failure_reason if last_action_res else None,
            "robot_pos": {"x": ws_after_action.robot.position.x, "y": ws_after_action.robot.position.y, "z": ws_after_action.robot.position.z},
            "gripper_open": ws_after_action.robot.gripper_open,
            "holding_object": ws_after_action.robot.holding_object_id,
            "object_pos": {
                "x": sr.get_red_object_position().x,
                "y": sr.get_red_object_position().y,
                "z": sr.get_red_object_position().z,
            },
            "object_state": sr.determine_object_state().value,
            "active_constraints": len(ws_after_action.active_constraints),
            "world_state_version": ws_after_action.version,
        }
        step_telemetry.append(record)

        # Emit distinct event traces based on action outcomes
        if act_type == "APPROACH" and success:
            trace.record("APPROACH_EXECUTED", f"EE reached standoff near object (v{ws_after_action.version})")
        elif act_type == "GRASP" and success:
            trace.record("GRASP_VERIFIED", f"Gripper clamped target object '{ws_after_action.robot.holding_object_id}'")
        elif act_type == "REPOSITION" and success:
            trace.record("REPOSITION_VERIFIED", f"Elevated object to clearance altitude z={record['robot_pos']['z']:.3f}m")
        elif act_type == "MOVE" and success:
            trace.record("MOVE_VERIFIED", f"Navigated over obstacle to target zone ({record['robot_pos']['x']:.2f}, {record['robot_pos']['y']:.2f})")
        elif act_type == "RELEASE" and success:
            trace.record("RELEASE_VERIFIED", "Gripper opened; object released in target zone")

        # Causal Disturbance Trigger: Immediately post-grasp (Step 2)
        if step_num == 2 and not causal_trace["disturbance_injected"]:
            causal_trace["disturbance_injected"] = True
            causal_trace["disturbed_step"] = step_num
            causal_trace["pre_disturbance_ws_version"] = ws_after_action.version

            # Physically inject obstacle in MuJoCo
            injected = runtime.inject_disturbance("PATH_BLOCKED")
            disturbed_ws = runtime.get_world_state()
            causal_trace["post_disturbance_ws_version"] = disturbed_ws.version
            obs = next((o for o in disturbed_ws.objects if o.is_obstacle), None)
            obs_pos = {"x": obs.position.x, "y": obs.position.y, "z": obs.position.z} if obs else {}
            causal_trace["obstacle_position"] = obs_pos

            trace.record("PATH_BLOCKED_INJECTED", f"Obstacle placed at ({obs_pos.get('x', 0):.3f}, {obs_pos.get('y', 0):.3f}, {obs_pos.get('z', 0):.3f})")
            trace.record("WORLD_STATE_CHANGED", f"Version advanced v{ws_after_action.version} -> v{disturbed_ws.version} with keepout constraint")

            if viewer is not None:
                viewer.sync()

            # Decision Engine evaluation on disturbed state
            peek_decision = engine.decide(disturbed_ws)
            evals_summary = []
            for cand in peek_decision.candidate_evaluations:
                eval_item = {
                    "step": step_num,
                    "action_id": cand.action.action_id,
                    "action_type": cand.action.action_type.value,
                    "is_valid": cand.is_valid,
                    "overall_score": cand.overall_score,
                    "rejection_reasons": cand.rejection_reasons,
                }
                evals_summary.append(eval_item)
                if not cand.is_valid:
                    rejected_actions_log.append(eval_item)
                    causal_trace["rejected_blocked_candidates"].append(eval_item)
                    if cand.action.action_type == ActionType.MOVE:
                        trace.record(
                            "MOVE_REJECTED_BLOCKED_PATH",
                            f"Constraint validation rejected '{cand.action.action_id}': {'; '.join(cand.rejection_reasons)}",
                        )

            candidate_evals_log.extend(evals_summary)
            causal_trace["recovery_action_selected"] = peek_decision.selected_action.action_type.value
            trace.record(
                "RECOVERY_DECISION_REPOSITION",
                f"Selected {peek_decision.selected_action.action_type.value} ({peek_decision.selected_action.action_id}) with clearance z={peek_decision.selected_action.target_position.z:.3f}m",
            )

    # 8. Run Closed-Loop Execution
    start_exec = time.time()
    task_result: TaskExecutionResult = runner.run_instruction(
        args.instruction,
        model_provider=provider,
        step_callback=on_step_executed,
    )
    total_exec_duration = time.time() - start_exec

    if viewer is not None:
        viewer.sync()

    # 9. Verify Recovery Execution
    repo_results = [r for r in runtime.execution_history if r.action_type == "REPOSITION"]
    if repo_results and repo_results[0].success:
        causal_trace["recovery_executed_success"] = True

    # 10. Physical Verification Measurements
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

    if within_tolerance and is_at_tabletop and final_obj_state == ObjectState.PLACED:
        trace.record(
            "FINAL_PLACEMENT_VERIFIED",
            f"Object at ({final_obj_pos.x:.3f}, {final_obj_pos.y:.3f}, {final_obj_pos.z:.3f}), error={horizontal_dist:.4f}m <= tol={tol:.4f}m",
        )
    else:
        trace.record(
            "FINAL_PLACEMENT_FAILED",
            f"Object error={horizontal_dist:.4f}m, tolerance={tol:.4f}m, state={final_obj_state}",
        )

    all_physical_criteria_met = (
        within_tolerance
        and is_at_tabletop
        and is_released
        and final_obj_state == ObjectState.PLACED
        and task_result.completed
        and causal_trace["disturbance_injected"]
        and causal_trace["recovery_executed_success"]
    )

    if all_physical_criteria_met:
        trace.record("TASK_COMPLETED", "Physical pick-and-place with obstacle avoidance succeeded.")
    else:
        trace.record("TASK_TERMINATED_INCOMPLETE", f"Failure reason: {task_result.failure_reason}")

    # 11. Compile Structured Telemetry
    telemetry: Dict[str, Any] = {
        "timestamp": run_timestamp,
        "project": "SĀRTHI",
        "simulator": "MuJoCo",
        "simulator_version": getattr(mujoco, "__version__", "unknown"),
        "robot": "Franka Emika Panda",
        "model": model_name,
        "model_provider": provider_name,
        "inference_endpoint": endpoint_url,
        "instruction": args.instruction,
        "task_understanding": {
            "task_id": tu.task_id,
            "objective": tu.objective,
            "target_object": tu.target_object,
            "target_location": tu.target_location,
            "required_actions": tu.required_actions,
            "confidence": tu.confidence,
            "reasoning_summary": tu.reasoning_summary,
        },
        "actions": task_result.executed_actions,
        "selected_actions": selected_actions_log,
        "candidate_evaluations": candidate_evals_log,
        "rejected_actions": rejected_actions_log,
        "disturbance": {
            "type": "PATH_BLOCKED",
            "injected": causal_trace["disturbance_injected"],
            "injected_at_step": causal_trace["disturbed_step"],
            "obstacle_position": causal_trace["obstacle_position"],
            "pre_disturbance_ws_version": causal_trace["pre_disturbance_ws_version"],
            "post_disturbance_ws_version": causal_trace["post_disturbance_ws_version"],
            "recovery_action": causal_trace["recovery_action_selected"],
        },
        "world_state_versions": [r["world_state_version"] for r in step_telemetry],
        "execution_durations": {
            "cognition_seconds": round(cognition_duration, 3),
            "wall_clock_seconds": round(total_exec_duration, 3),
            "simulation_physics_seconds": round(runtime.simulation_time, 3),
        },
        "recovery_count": task_result.recovery_count,
        "final_object_position": {"x": round(final_obj_pos.x, 4), "y": round(final_obj_pos.y, 4), "z": round(final_obj_pos.z, 4)},
        "target_position": {"x": round(tgt_pos.x, 4), "y": round(tgt_pos.y, 4), "z": round(tgt_pos.z, 4)},
        "placement_error": round(horizontal_dist, 4),
        "tolerance": round(tol, 4),
        "gripper_state": "OPEN" if is_released else "CLOSED",
        "gripper_released": is_released,
        "final_object_state": final_obj_state.value,
        "overall_result": "COMPLETED" if all_physical_criteria_met else "FAILED",
        "physical_verification_passed": all_physical_criteria_met,
        "event_trace": trace.get_formatted_trace(),
    }

    # Sanitize telemetry to guarantee zero secret leakage
    telemetry = sanitize_dict_for_telemetry(telemetry)

    # 12. Write Output JSON if requested
    if args.output:
        out_path = pathlib.Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(telemetry, f, indent=2)
        print(f"\nStructured telemetry written to: {out_path}")

    return all_physical_criteria_met, telemetry


def main() -> None:
    args = parse_arguments()

    print_banner("SĀRTHI MUJOCO DEMONSTRATION & VALIDATION RUNNER")
    print(f"Instruction:        \"{args.instruction}\"")
    print(f"Mode:               {'Interactive Viewer' if args.viewer else 'Headless'}")
    print(f"Model Provider:     {'MockModelProvider' if args.mock else 'NebiusNemotronProvider'}")
    print(f"Seed:               {args.seed if args.seed is not None else 'None (inherently deterministic)'}")
    print(f"Telemetry Output:   {args.output or 'None'}")

    print_section("HUMAN-READABLE EVENT TRACE")
    try:
        success, tel = execute_validation_run(args)
    except Exception as exc:
        print(f"\nFATAL RUNNER FAILURE: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print Validation Summary as requested
    print_section("VALIDATION SUMMARY")
    print(f"Simulator:         {tel['simulator']}")
    print(f"Robot:             {tel['robot']}")
    print(f"Model:             {tel['model']}")
    print(f"Task:              {tel['instruction']}")
    print(f"Disturbance:       {tel['disturbance']['type']}")
    print(f"Recovery:          {tel['disturbance'].get('recovery_action', 'REPOSITION')}")
    print(f"Recovery Count:    {tel['recovery_count']}")
    print(f"Placement Error:   {tel['placement_error']:.4f} m")
    print(f"Tolerance:         {tel['tolerance']:.4f} m")
    print(f"Object State:      {tel['final_object_state']}")
    print(f"Gripper Released:  {tel['gripper_released']}")
    print(f"Result:            {tel['overall_result']}")

    if success:
        print_banner("VALIDATION RUN COMPLETED SUCCESSFULLY (EXIT 0)")
        sys.exit(0)
    else:
        print_banner("VALIDATION RUN FAILED (EXIT 1)")
        sys.exit(1)


if __name__ == "__main__":
    main()
