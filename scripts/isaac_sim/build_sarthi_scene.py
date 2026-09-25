"""
SĀRTHI — NVIDIA Isaac Sim Scene Assembly & Standalone Runner Script.

Usage (Inside NVIDIA Isaac Sim 6.x Environment):
    # Headless construction and stage save:
    python scripts/isaac_sim/build_sarthi_scene.py --headless --save-stage stages/sarthi_tabletop.usd

    # Interactive visual inspection:
    python scripts/isaac_sim/build_sarthi_scene.py --robot-asset /Isaac/Robots/Franka/franka.usd

SAFETY:
    If invoked outside an Isaac Sim runtime environment, this script fails immediately.
    It does NOT fall back to LocalSimulationAdapter and does NOT pretend execution succeeded.
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from simulation.adapters.isaac_sim import is_isaac_sim_available


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SĀRTHI NVIDIA Isaac Sim Scene Builder and Benchmark Runner"
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
        help="Path or Omniverse Nucleus URL to the robot USD asset",
    )
    parser.add_argument(
        "--save-stage",
        type=str,
        default=None,
        help="Optional destination path to export the assembled USD stage",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Simulation runtime in seconds before clean exit (default 5.0s)",
    )
    parser.add_argument(
        "--inject-disturbance",
        action="store_true",
        help="Test injecting the PATH_BLOCKED obstacle into the stage",
    )

    args = parser.parse_args()

    # Strict Runtime Safety Check
    if not is_isaac_sim_available():
        print(
            "ERROR: Isaac Sim runtime is required. Run this script using the Isaac Sim Python environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Initialize and execute real Isaac Sim runtime
    from simulation.isaac_runtime.robot_scene import RobotSceneConfig
    from simulation.isaac_runtime.runtime import SarthiIsaacRuntime

    robot_config = RobotSceneConfig(robot_asset_path=args.robot_asset)
    runtime = SarthiIsaacRuntime(robot_config=robot_config)

    print(f"[SĀRTHI] Initializing Isaac Sim (headless={args.headless})...")
    runtime.initialize(headless=args.headless)

    print("[SĀRTHI] Loading scenario: tabletop_pick_and_place_mvp...")
    runtime.load_scenario()

    if args.inject_disturbance:
        print("[SĀRTHI] Injecting PATH_BLOCKED dynamic obstacle...")
        runtime.inject_disturbance()

    if args.save_stage:
        print(f"[SĀRTHI] Saving assembled stage to '{args.save_stage}'...")
        if runtime._world and hasattr(runtime._world, "stage"):
            runtime._world.stage.GetRootLayer().Export(args.save_stage)

    # Simulation loop
    steps = int(args.duration * 60)
    print(f"[SĀRTHI] Advancing simulation for {steps} steps ({args.duration}s)...")
    for _ in range(steps):
        runtime.step(render=not args.headless)

    print("[SĀRTHI] Shutting down Isaac Sim cleanly.")
    runtime.shutdown()
    print("[SĀRTHI] Done.")


if __name__ == "__main__":
    main()
