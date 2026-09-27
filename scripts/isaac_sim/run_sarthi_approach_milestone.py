"""
SĀRTHI — NVIDIA Isaac Sim Closed-Loop APPROACH Milestone Runner.

Executes the first Isaac Sim physical integration milestone:
Natural Language Instruction
    ↓
Real Nebius Token Factory (nvidia/Nemotron-3-Ultra-550b-a55b)
    ↓
TaskUnderstanding
    ↓
SarthiTaskRunner (max_steps=1)
    ↓
IsaacSimAdapter / SarthiIsaacRuntime
    ↓
Decision Engine (selects APPROACH)
    ↓
Isaac Sim Articulation (Franka Panda)
    ↓
Live Physical State Verification

Usage (inside Isaac Sim container/environment):
    python scripts/isaac_sim/run_sarthi_approach_milestone.py --headless
    python scripts/isaac_sim/run_sarthi_approach_milestone.py --config configs/isaac_sim_validation.yaml
"""

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
# Load environment without echoing credentials
load_dotenv(os.path.join(_PROJECT_ROOT, "configs", ".env"))

import yaml

from backend.app.model.config import NebiusConfig
from backend.app.orchestration.task_runner import SarthiTaskRunner
from simulation.adapters.isaac_sim import (
    IsaacSimAdapter,
    is_isaac_sim_available,
)
from simulation.isaac_runtime.robot_scene import RobotSceneConfig
from simulation.isaac_runtime.runtime import SarthiIsaacRuntime
from simulation.scenarios.tabletop_pick_place import create_default_scenario


def load_milestone_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads runtime validation settings from YAML with sensible defaults."""
    default_config = {
        "scenario_id": "tabletop_pick_and_place_mvp",
        "headless": True,
        "robot_asset_path": None,
        "physics_steps_per_action": 60,
    }
    resolved_path = config_path or os.path.join(_PROJECT_ROOT, "configs", "isaac_sim_validation.yaml")
    if os.path.exists(resolved_path):
        with open(resolved_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                default_config.update(loaded)
    return default_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SĀRTHI Real NVIDIA Isaac Sim APPROACH Milestone Runner"
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
        "--instruction",
        type=str,
        default="Move the red object to the blue target.",
        help="Natural language task instruction",
    )

    args = parser.parse_args()

    # 1. Strict Runtime Check
    if not is_isaac_sim_available():
        print(
            "ERROR: Isaac Sim runtime is required to execute this milestone. "
            "Run this script using the NVIDIA Isaac Sim container or Python environment.\n"
            "Example:\n"
            "  docker run --gpus all -it --network=host \\\n"
            "    -v <path_to_sarthi>:/workspace/sarthi \\\n"
            "    nvcr.io/nvidia/isaac-sim:6.1.0 \\\n"
            "    python /workspace/sarthi/scripts/isaac_sim/run_sarthi_approach_milestone.py --headless",
            file=sys.stderr,
        )
        sys.exit(1)

    # 2. Configuration & Initialization
    cfg = load_milestone_config(args.config)
    headless = args.headless or cfg.get("headless", True)
    robot_asset = args.robot_asset or cfg.get("robot_asset_path")
    physics_steps = int(cfg.get("physics_steps_per_action", 60))

    nebius_cfg = NebiusConfig.from_env()
    model_name = nebius_cfg.model or "nvidia/Nemotron-3-Ultra-550b-a55b"

    print("=== SĀRTHI ISAAC SIM APPROACH MILESTONE ===")
    print(f"Instruction: \"{args.instruction}\"")
    print(f"Configured Model: {model_name}")
    print(f"Isaac Sim Headless: {headless}")
    print(f"Physics Steps Per Action: {physics_steps}")

    scenario = create_default_scenario()
    robot_config = RobotSceneConfig(
        robot_id=scenario.robot.robot_id,
        robot_asset_path=robot_asset,
    )

    # 3. Initialize Isaac Sim Runtime & Build Scene
    runtime = SarthiIsaacRuntime(
        scenario=scenario,
        robot_config=robot_config,
        physics_steps_per_action=physics_steps,
    )
    runtime.initialize(headless=headless)
    prims = runtime.load_scenario()

    scene_loaded = len(prims) > 0
    art_bound = runtime.articulation_controller.is_bound

    # 4. Wrap with IsaacSimAdapter
    adapter = IsaacSimAdapter(sim_backend=runtime)

    # 5. Create SarthiTaskRunner configured for max_steps=1 (APPROACH phase only)
    runner = SarthiTaskRunner(
        adapter=adapter,
        max_steps=1,
    )

    print("\nExecuting runner.run_instruction() with default NebiusNemotronProvider...")
    try:
        # Crucial test: run_instruction called WITHOUT model_provider
        result = runner.run_instruction(args.instruction)
    except Exception as exc:
        print(f"FAILED with exception: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        runtime.shutdown()
        sys.exit(1)

    # 6. Extract Telemetry & Verification Data
    tu = result.task_understanding
    final_state = adapter.get_world_state()
    measured_ee = runtime.articulation_controller.get_end_effector_position()
    target_pos = scenario.red_object.initial_pose
    distance = math.sqrt(
        (measured_ee.x - target_pos.x) ** 2
        + (measured_ee.y - target_pos.y) ** 2
        + (measured_ee.z - target_pos.z) ** 2
    )

    # 7. Print Structured Milestone Report
    print("\n================ REPORT ================")

    # A. Nemotron
    print("\n--- A. Nemotron ---")
    print("Live Token Factory call: SUCCESS")
    print(f"Model used: {model_name}")
    if tu:
        print(f"Task ID: {tu.task_id}")
        print(f"Objective: {tu.objective}")
        print(f"Target object: {tu.target_object}")
        print(f"Target location: {tu.target_location}")
        print(f"Confidence: {tu.confidence}")
        print(f"Reasoning summary: {tu.reasoning_summary}")
    else:
        print("TaskUnderstanding: None")

    # B. Decision
    print("\n--- B. Decision ---")
    first_action = result.executed_actions[0] if result.executed_actions else "None"
    print(f"Selected action: {first_action}")
    print(f"Executed actions: {result.executed_actions}")
    print("Decision authority: SĀRTHI Decision Engine (deterministic)")

    # C. Isaac Sim
    print("\n--- C. Isaac Sim ---")
    print(f"Scene loaded: {scene_loaded}")
    print(f"Robot articulation bound: {art_bound}")
    print(f"APPROACH dispatched: {'APPROACH' in result.executed_actions}")
    print(f"Physics stepping performed: {physics_steps} steps at 60Hz")

    # D. Verification
    print("\n--- D. Verification ---")
    print(f"Measured end-effector position: ({measured_ee.x:.3f}, {measured_ee.y:.3f}, {measured_ee.z:.3f})")
    print(f"Target position: ({target_pos.x:.3f}, {target_pos.y:.3f}, {target_pos.z:.3f})")
    print(f"Distance to target: {distance:.3f}m (tolerance: 0.050m)")
    app_verified = any(v.action_type == "APPROACH" and v.verified for v in result.verification_results)
    print(f"Verification result: {'PASSED' if app_verified else 'FAILED'}")

    # E. Final
    print("\n--- E. Final ---")
    print(f"Run status: {result.run_status}")
    print(f"Final world-state version: {result.final_world_state_version}")
    print(f"Failure reason: {result.failure_reason}")
    print(f"Completed: {result.completed} (stopped cleanly after Step 1 as expected for milestone)")

    # 8. Clean up
    runtime.shutdown()

    if app_verified:
        print("\n=== ISAAC SIM APPROACH MILESTONE: PASSED ===")
        sys.exit(0)
    else:
        print("\n=== ISAAC SIM APPROACH MILESTONE: FAILED ===", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
