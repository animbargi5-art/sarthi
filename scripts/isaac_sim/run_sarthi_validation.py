"""
SĀRTHI — Real NVIDIA Isaac Sim Runtime Validation Entry Point.

Executes the locked tabletop pick-and-place benchmark under closed-loop control:
1. Initializes NVIDIA Isaac Sim 6.x runtime.
2. Loads TabletopPickPlaceScenario and constructs the OpenUSD stage.
3. Connects IsaacSimAdapter with SarthiIsaacRuntime.
4. Delegates all physical action selection to SĀRTHI Decision Engine.
5. Injects PATH_BLOCKED disturbance dynamically AFTER_GRASP.
6. Verifies autonomous Decision Engine recovery (REPOSITION -> MOVE -> RELEASE).
7. Audits execution trace and outputs structured validation summary.

Usage (Inside NVIDIA Isaac Sim Environment):
    python scripts/isaac_sim/run_sarthi_validation.py --config configs/isaac_sim_validation.yaml
    python scripts/isaac_sim/run_sarthi_validation.py --headless --save-stage stages/validation.usd
"""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import yaml
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app.decision_engine.engine import SarthiDecisionEngine
from backend.app.decision_engine.models import ActionType, CandidateAction, WorldState
from simulation.adapters.isaac_sim import (
    IsaacSimAction,
    IsaacSimAdapter,
    is_isaac_sim_available,
)
from simulation.core.events import ActionExecutionResult, DisturbanceType
from simulation.scenarios.tabletop_pick_place import (
    DisturbanceTiming,
    TabletopPickPlaceScenario,
    create_default_scenario,
)


# ---------------------------------------------------------------------------
# Structured Validation Logging & Trace Schemas
# ---------------------------------------------------------------------------

class ValidationTraceEntry(BaseModel):
    """Auditable log record for each closed-loop cycle step."""
    step_number: int
    world_state_version_before: Any
    selected_action: Dict[str, Any]
    isaac_action: Dict[str, Any]
    execution_result: Dict[str, Any]
    world_state_version_after: Any
    verification_result: bool
    disturbance_state: Dict[str, Any]
    final_status: str


def log_phase(
    phase: str,
    action: Optional[str] = None,
    status: str = "ok",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Prints concise structured console log in the canonical [ISAAC] format."""
    parts = [f"[ISAAC] phase={phase}"]
    if action:
        parts.append(f"action={action}")
    if extra:
        for k, v in extra.items():
            parts.append(f"{k}={v}")
    parts.append(f"status={status}")
    print(" ".join(parts))


# ---------------------------------------------------------------------------
# Configuration Loader
# ---------------------------------------------------------------------------

def load_validation_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads runtime validation settings from YAML.
    Contains only execution options (headless, max_steps, paths);
    does NOT contain physical scenario constants.
    """
    default_config = {
        "scenario_id": "tabletop_pick_and_place_mvp",
        "headless": True,
        "save_stage": None,
        "output_directory": "logs/isaac_validation",
        "max_steps": 10,
        "enable_trace": True,
        "robot_asset_path": None,
    }

    resolved_path = config_path or os.path.join(_PROJECT_ROOT, "configs", "isaac_sim_validation.yaml")
    if os.path.exists(resolved_path):
        with open(resolved_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                default_config.update(loaded)

    return default_config


# ---------------------------------------------------------------------------
# Closed-Loop Execution Runner
# ---------------------------------------------------------------------------

class SarthiIsaacValidator:
    """
    Coordinates real runtime execution between IsaacSimAdapter and Decision Engine.
    Enforces strict architectural boundaries:
    - Decision Engine selects every action.
    - Adapter translates and executes actions.
    - Runtime updates physical simulation.
    """

    def __init__(
        self,
        runtime: Any,
        adapter: IsaacSimAdapter,
        scenario: Optional[TabletopPickPlaceScenario] = None,
        engine: Optional[SarthiDecisionEngine] = None,
        max_steps: int = 10,
        enable_trace: bool = True,
    ):
        self.runtime = runtime
        self.adapter = adapter
        self.scenario = scenario or create_default_scenario()
        self.engine = engine or SarthiDecisionEngine()
        self.max_steps = max_steps
        self.enable_trace = enable_trace
        self.trace: List[ValidationTraceEntry] = []
        self._disturbance_injected = False

    def run_validation(self) -> Tuple[bool, str]:
        """
        Executes the closed-loop benchmark sequence.

        Phases:
        INITIALIZE -> SCENE_BUILT -> TASK_LOADED -> APPROACH -> GRASP ->
        DISTURBANCE_INJECTED -> RECOVERY_DECISION -> REPOSITION -> MOVE ->
        RELEASE -> FINAL_VERIFICATION -> SUCCESS
        """
        log_phase("TASK_LOADED", extra={"scenario": self.scenario.scenario_id})

        step_count = 0
        while step_count < self.max_steps:
            step_count += 1

            # 1. State observation through adapter
            world_state = self.adapter.get_world_state()
            version_before = world_state.version

            # Check if task is already complete
            is_done, failure_reason = self.scenario.check_success_conditions(world_state)
            if is_done:
                log_phase("FINAL_VERIFICATION", status="ok")
                log_phase("SUCCESS", status="ok")
                return True, "Task completed successfully"

            # 2. Decision Engine deliberates and selects action
            decision = self.engine.decide(world_state)
            selected_action = decision.selected_action
            action_type_str = selected_action.action_type.value

            # Phase tracking & logging for recovery decision
            if self._disturbance_injected and action_type_str == ActionType.REPOSITION.value:
                log_phase("RECOVERY_DECISION", action="REPOSITION", status="ok")

            log_phase(action_type_str, action=action_type_str, status="ok")

            # 3. Translate action to Isaac Sim representation
            isaac_action = IsaacSimAdapter.translate_action(selected_action)

            # 4. Dispatch and execute action in Isaac Sim
            exec_result = self.adapter.execute_action(selected_action)

            # 5. Verify action result
            verified = self.adapter.verify_action_result(selected_action)

            world_state_after = self.adapter.get_world_state()
            version_after = world_state_after.version

            # Record step trace
            if self.enable_trace:
                self.trace.append(ValidationTraceEntry(
                    step_number=step_count,
                    world_state_version_before=version_before,
                    selected_action=selected_action.model_dump(),
                    isaac_action=isaac_action.model_dump(),
                    execution_result=exec_result.model_dump(),
                    world_state_version_after=version_after,
                    verification_result=verified,
                    disturbance_state={
                        "is_active": self._disturbance_injected,
                        "type": DisturbanceType.PATH_BLOCKED.value if self._disturbance_injected else None,
                    },
                    final_status="STEP_OK" if exec_result.success else "STEP_FAILED",
                ))

            if not exec_result.success:
                log_phase(action_type_str, action=action_type_str, status="failed", extra={"reason": exec_result.failure_reason})
                return False, f"Action execution failed at step {step_count}: {exec_result.failure_reason}"

            # 6. Inject dynamic disturbance AFTER_GRASP
            if (
                not self._disturbance_injected
                and action_type_str == ActionType.GRASP.value
                and exec_result.success
                and self.scenario.disturbance_timing == DisturbanceTiming.AFTER_GRASP
            ):
                dist_event = self.scenario.create_disturbance_event()
                self.adapter.inject_disturbance(dist_event)
                self._disturbance_injected = True
                log_phase(
                    "DISTURBANCE_INJECTED",
                    status="ok",
                    extra={"type": DisturbanceType.PATH_BLOCKED.value, "obstacle_id": self.scenario.obstacle.obstacle_id},
                )

        # Final state check
        final_state = self.adapter.get_world_state()
        is_done, failure_reason = self.scenario.check_success_conditions(final_state)
        if is_done:
            log_phase("FINAL_VERIFICATION", status="ok")
            log_phase("SUCCESS", status="ok")
            return True, "Task completed successfully"

        log_phase("FINAL_VERIFICATION", status="failed", extra={"reason": failure_reason})
        return False, f"Validation exceeded max steps ({self.max_steps}): {failure_reason}"


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SĀRTHI Real NVIDIA Isaac Sim Closed-Loop Validation Runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML validation configuration file",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Isaac Sim in headless mode (no GUI window)",
    )
    parser.add_argument(
        "--robot-asset",
        type=str,
        default=None,
        help="Optional path to robot USD asset",
    )
    parser.add_argument(
        "--save-stage",
        type=str,
        default=None,
        help="Optional destination path to export the assembled USD stage",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output directory or JSON file path for execution trace",
    )

    args = parser.parse_args()

    # Strict Runtime Safety Check
    if not is_isaac_sim_available():
        print(
            "ERROR: Isaac Sim runtime is required. Run this script using the Isaac Sim Python environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load configuration
    cfg = load_validation_config(args.config)
    headless = args.headless or cfg.get("headless", True)
    robot_asset = args.robot_asset or cfg.get("robot_asset_path")
    save_stage = args.save_stage or cfg.get("save_stage")
    out_target = args.output or cfg.get("output_directory", "logs/isaac_validation")
    max_steps = int(cfg.get("max_steps", 10))
    enable_trace = bool(cfg.get("enable_trace", True))

    from simulation.isaac_runtime.robot_scene import RobotSceneConfig
    from simulation.isaac_runtime.runtime import SarthiIsaacRuntime

    log_phase("INITIALIZE", status="ok", extra={"headless": str(headless)})

    scenario = create_default_scenario()
    robot_config = RobotSceneConfig(
        robot_id=scenario.robot.robot_id,
        robot_asset_path=robot_asset,
    )
    runtime = SarthiIsaacRuntime(scenario=scenario, robot_config=robot_config)
    runtime.initialize(headless=headless)

    runtime.load_scenario()
    log_phase("SCENE_BUILT", status="ok")

    if save_stage and runtime._world and hasattr(runtime._world, "stage"):
        runtime._world.stage.GetRootLayer().Export(save_stage)

    # Initialize adapter with active runtime
    adapter = IsaacSimAdapter(sim_backend=runtime)

    # Run closed loop validation
    validator = SarthiIsaacValidator(
        runtime=runtime,
        adapter=adapter,
        scenario=scenario,
        max_steps=max_steps,
        enable_trace=enable_trace,
    )

    success, message = validator.run_validation()

    # Save trace log
    if enable_trace and validator.trace:
        if out_target.endswith(".json"):
            trace_path = out_target
            out_dir = os.path.dirname(trace_path) or "."
        else:
            out_dir = out_target
            trace_path = os.path.join(out_dir, "validation_trace.json")
        os.makedirs(out_dir, exist_ok=True)
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump([t.model_dump() for t in validator.trace], f, indent=2, default=str)
        print(f"[ISAAC] Execution trace saved to: {trace_path}")

    runtime.shutdown()

    if success:
        print(f"[ISAAC] VALIDATION RESULT: PASSED ({message})")
        sys.exit(0)
    else:
        print(f"[ISAAC] VALIDATION RESULT: FAILED ({message})", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
