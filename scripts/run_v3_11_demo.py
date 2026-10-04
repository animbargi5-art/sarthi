#!/usr/bin/env python3
"""
SĀRTHI V3-11 — Visual Demonstration & Telemetry Runner.

Usage:
    # Run full live demonstration and generate records/v3_11_demo_telemetry.json
    python scripts/run_v3_11_demo.py --mode live

    # Replay demonstration from saved telemetry
    python scripts/run_v3_11_demo.py --mode replay

    # Run failure/fallback demonstration
    python scripts/run_v3_11_demo.py --mode failure

    # Launch local visual dashboard
    python scripts/run_v3_11_demo.py --serve --port 8080
"""

import argparse
import json
import logging
import os
import sys

# Ensure root repository directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

try:
    from dotenv import load_dotenv
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    load_dotenv(os.path.join(_root, "configs", ".env"))
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    pass

from backend.app.demo.service import SarthiDemoService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("v3_11_demo_runner")


def run_live(output_path: str = "records/v3_11_demo_telemetry.json") -> None:
    print("=" * 70)
    print("SĀRTHI V3-11: LIVE DEMONSTRATION")
    print("Pipeline: Nemotron + Laya + Decision Engine + MuJoCo Panda")
    print("Disturbance: PATH_BLOCKED Obstacle")
    print("=" * 70)

    service = SarthiDemoService()
    session = service.run_live_demo(
        instruction="Move the red object to the blue target.",
        telemetry_output_path=output_path,
    )

    print("\n--- DEMONSTRATION EVENT STREAM ---")
    for ev in session.events:
        print(f"{ev.code:<36} | {ev.authority_classification:<35} | {ev.label}")

    print("\n--- RECOVERY TRACE ---")
    rec = session.recovery_visualization
    if rec:
        print(f"Disturbance        : {rec.disturbance_name} ({rec.obstacle_id})")
        print(f"Direct MOVE        : {rec.direct_move_status} ({rec.rejection_reason})")
        print(f"Available Options  : {rec.available_recovery_options}")
        print(f"Laya Decision      : {rec.laya_selected_option} [{rec.laya_authority_label}]")
        print(f"Validation         : {rec.validation_status} [{rec.validation_authority_label}]")
        print(f"Execution Result   : {rec.execution_result}")

    print("\n--- FINAL RESULT ---")
    fp = session.final_placement
    print(f"Task Completed     : {session.task_completed}")
    print(f"Final Position     : {fp.get('final_position')}")
    print(f"Target Position    : {fp.get('target_position')}")
    print(f"Placement Error    : {fp.get('placement_error_m')} m (tolerance: {fp.get('tolerance_m')} m)")
    print(f"Telemetry Saved To : {output_path}")
    print("=" * 70)


def run_replay(telemetry_path: str = "records/v3_11_demo_telemetry.json") -> None:
    print("=" * 70)
    print("SĀRTHI V3-11: REPLAY MODE — OFFLINE DEMONSTRATION")
    print(f"Source Telemetry: {telemetry_path}")
    print("CRITICAL: THIS IS OFFLINE PLAYBACK, NOT LIVE ROBOT EXECUTION")
    print("=" * 70)

    service = SarthiDemoService()
    session = service.run_replay_demo(telemetry_path=telemetry_path)

    print("\n--- REPLAY EVENT STREAM ---")
    for ev in session.events:
        print(f"{ev.code:<36} | [REPLAY] {ev.label}")

    print("\n--- REPLAY RECOVERY TRACE ---")
    rec = session.recovery_visualization
    if rec:
        print(f"Disturbance        : {rec.disturbance_name}")
        print(f"Direct MOVE        : {rec.direct_move_status}")
        print(f"Laya Proposal      : {rec.laya_selected_option} (NON-AUTHORITATIVE)")
        print(f"Engine Validation  : {rec.validation_status} (PHYSICAL AUTHORITY)")
        print(f"Execution Result   : {rec.execution_result}")

    print("\n--- REPLAY PLACEMENT VERIFICATION ---")
    fp = session.final_placement
    print(f"Task Completed     : {session.task_completed}")
    print(f"Placement Error    : {fp.get('placement_error_m')} m")
    print(f"Within Tolerance   : {fp.get('within_tolerance')}")
    print("=" * 70)


def run_failure() -> None:
    print("=" * 70)
    print("SĀRTHI V3-11: FAILURE & FALLBACK DEMONSTRATION")
    print("Scenario: Laya Decision Timeout -> Deterministic Recovery")
    print("=" * 70)

    service = SarthiDemoService()
    result = service.run_failure_demo()
    for k, v in result.items():
        print(f"{k:<25}: {v}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="SĀRTHI V3-11 Visual Demo & Telemetry Runner")
    parser.add_argument("--mode", choices=["live", "replay", "failure"], default="live", help="Demo execution mode")
    parser.add_argument("--output", default="records/v3_11_demo_telemetry.json", help="Telemetry output file path")
    parser.add_argument("--serve", action="store_true", help="Launch FastAPI visual dashboard server")
    parser.add_argument("--port", type=int, default=8080, help="Web dashboard server port")
    args = parser.parse_args()

    if args.serve:
        import uvicorn
        from backend.app.demo.server import app
        print(f"Starting SĀRTHI V3-11 Visual Dashboard at http://127.0.0.1:{args.port}")
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")
        return

    if args.mode == "live":
        run_live(output_path=args.output)
    elif args.mode == "replay":
        run_replay(telemetry_path=args.output)
    elif args.mode == "failure":
        run_failure()


if __name__ == "__main__":
    main()
