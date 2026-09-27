"""
SĀRTHI Phase G — MuJoCo Demonstration, Validation & Evidence Layer Tests.

Tests the CLI interface, event trace generation, structured JSON telemetry output,
reproducibility configurations, secret redaction, and failure handling.
All automated tests use MockModelProvider to guarantee deterministic offline execution.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from scripts.mujoco.run_sarthi_mujoco_e2e import (
    EventTraceRecorder,
    execute_validation_run,
    parse_arguments,
    sanitize_dict_for_telemetry,
)
from simulation.mujoco_runtime import require_mujoco


class TestSarthiMuJoCoValidation(unittest.TestCase):
    """Phase G Test Suite: Validates demonstration and evidence layer."""

    @classmethod
    def setUpClass(cls):
        require_mujoco()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    # -----------------------------------------------------------------------
    # Test 1: CLI Argument Parsing Defaults
    # -----------------------------------------------------------------------
    def test_01_cli_argument_parsing_defaults(self):
        """1. Validates default CLI arguments for headless runner."""
        args = parse_arguments([])
        self.assertTrue(args.headless)
        self.assertFalse(args.viewer)
        self.assertFalse(args.mock)
        self.assertIsNone(args.seed)
        self.assertIsNone(args.output)
        self.assertEqual(args.instruction, "Move the red object to the blue target.")
        self.assertEqual(args.max_steps, 10)
        self.assertEqual(args.steps_per_action, 400)

    # -----------------------------------------------------------------------
    # Test 2: CLI Argument Parsing Custom
    # -----------------------------------------------------------------------
    def test_02_cli_argument_parsing_custom(self):
        """2. Validates custom CLI argument overrides."""
        out_file = os.path.join(self.temp_dir.name, "report.json")
        args = parse_arguments([
            "--instruction", "Custom task instruction",
            "--output", out_file,
            "--seed", "123",
            "--mock",
            "--viewer",
            "--max-steps", "5",
            "--steps-per-action", "300",
        ])
        self.assertEqual(args.instruction, "Custom task instruction")
        self.assertEqual(args.output, out_file)
        self.assertEqual(args.seed, 123)
        self.assertTrue(args.mock)
        self.assertTrue(args.viewer)
        self.assertEqual(args.max_steps, 5)
        self.assertEqual(args.steps_per_action, 300)

    # -----------------------------------------------------------------------
    # Test 3: Successful Headless Validation Execution
    # -----------------------------------------------------------------------
    def test_03_successful_headless_validation_execution(self):
        """3. Executes headless validation with mock provider and verifies success."""
        args = parse_arguments(["--mock", "--headless"])
        success, telemetry = execute_validation_run(args)

        self.assertTrue(success)
        self.assertEqual(telemetry["overall_result"], "COMPLETED")
        self.assertTrue(telemetry["physical_verification_passed"])
        self.assertEqual(
            telemetry["actions"],
            ["APPROACH", "GRASP", "REPOSITION", "MOVE", "RELEASE"],
        )

    # -----------------------------------------------------------------------
    # Test 4: JSON Telemetry File Written
    # -----------------------------------------------------------------------
    def test_04_json_telemetry_file_written(self):
        """4. Validates that structured JSON telemetry is written to --output path."""
        out_path = os.path.join(self.temp_dir.name, "output_telemetry.json")
        args = parse_arguments(["--mock", "--output", out_path])
        success, telemetry = execute_validation_run(args)

        self.assertTrue(success)
        self.assertTrue(os.path.exists(out_path))

        with open(out_path, "r", encoding="utf-8") as f:
            loaded_json = json.load(f)

        self.assertEqual(loaded_json["project"], "SĀRTHI")
        self.assertEqual(loaded_json["simulator"], "MuJoCo")
        self.assertEqual(loaded_json["overall_result"], "COMPLETED")
        self.assertEqual(loaded_json["placement_error"], telemetry["placement_error"])

    # -----------------------------------------------------------------------
    # Test 5: Required Telemetry Fields Present & Typed
    # -----------------------------------------------------------------------
    def test_05_required_telemetry_fields_present_and_typed(self):
        """5. Verifies all required telemetry fields exist and match expected types."""
        args = parse_arguments(["--mock"])
        _, telemetry = execute_validation_run(args)

        required_keys = [
            "timestamp", "project", "simulator", "robot", "model",
            "model_provider", "instruction", "task_understanding",
            "actions", "selected_actions", "candidate_evaluations",
            "rejected_actions", "disturbance", "world_state_versions",
            "execution_durations", "recovery_count", "final_object_position",
            "target_position", "placement_error", "tolerance", "gripper_state",
            "gripper_released", "final_object_state", "overall_result",
            "physical_verification_passed", "event_trace",
        ]
        for key in required_keys:
            self.assertIn(key, telemetry, f"Missing required telemetry key: {key}")

        self.assertIsInstance(telemetry["actions"], list)
        self.assertIsInstance(telemetry["candidate_evaluations"], list)
        self.assertIsInstance(telemetry["rejected_actions"], list)
        self.assertIsInstance(telemetry["disturbance"], dict)
        self.assertIsInstance(telemetry["placement_error"], float)
        self.assertIsInstance(telemetry["gripper_released"], bool)
        self.assertIsInstance(telemetry["event_trace"], list)

    # -----------------------------------------------------------------------
    # Test 6: Secret Redaction in Telemetry
    # -----------------------------------------------------------------------
    def test_06_secret_redaction_in_telemetry(self):
        """6. Verifies sanitize_dict_for_telemetry removes API keys and tokens."""
        dirty_data = {
            "api_key": "v1.super_secret_token_12345",
            "authorization": "Bearer secret_abc_xyz",
            "safe_field": "public_data",
            "nested": {
                "user_token": "sk-123456789",
                "nested_safe": "harmless",
            },
        }
        sanitized = sanitize_dict_for_telemetry(dirty_data)

        self.assertNotIn("api_key", sanitized)
        self.assertNotIn("authorization", sanitized)
        self.assertNotIn("user_token", sanitized["nested"])
        self.assertEqual(sanitized["safe_field"], "public_data")
        self.assertEqual(sanitized["nested"]["nested_safe"], "harmless")

    # -----------------------------------------------------------------------
    # Test 7: Event Trace Completeness
    # -----------------------------------------------------------------------
    def test_07_event_trace_completeness(self):
        """7. Verifies the human-readable event trace records the complete lifecycle."""
        args = parse_arguments(["--mock"])
        _, telemetry = execute_validation_run(args)
        trace_lines = telemetry["event_trace"]

        self.assertGreaterEqual(len(trace_lines), 14)

        expected_events = [
            "TASK_RECEIVED",
            "NEMOTRON_TASK_UNDERSTANDING",
            "WORLD_STATE_OBSERVED",
            "DECISION_SELECTED",
            "APPROACH_EXECUTED",
            "GRASP_VERIFIED",
            "PATH_BLOCKED_INJECTED",
            "WORLD_STATE_CHANGED",
            "MOVE_REJECTED_BLOCKED_PATH",
            "RECOVERY_DECISION_REPOSITION",
            "REPOSITION_VERIFIED",
            "MOVE_VERIFIED",
            "RELEASE_VERIFIED",
            "FINAL_PLACEMENT_VERIFIED",
            "TASK_COMPLETED",
        ]
        trace_text = "\n".join(trace_lines)
        for evt in expected_events:
            self.assertIn(evt, trace_text, f"Missing event in trace: {evt}")

    # -----------------------------------------------------------------------
    # Test 8: Reproducibility Configuration
    # -----------------------------------------------------------------------
    def test_08_reproducibility_configuration(self):
        """8. Consecutive deterministic runs produce identical action sequences."""
        args1 = parse_arguments(["--mock", "--seed", "42"])
        _, tel1 = execute_validation_run(args1)

        args2 = parse_arguments(["--mock", "--seed", "42"])
        _, tel2 = execute_validation_run(args2)

        self.assertEqual(tel1["actions"], tel2["actions"])
        self.assertEqual(tel1["recovery_count"], tel2["recovery_count"])
        self.assertEqual(tel1["overall_result"], tel2["overall_result"])
        self.assertAlmostEqual(tel1["placement_error"], tel2["placement_error"], places=3)

    # -----------------------------------------------------------------------
    # Test 9: Failure Propagation Handling
    # -----------------------------------------------------------------------
    def test_09_failure_propagation_handling(self):
        """9. Unresolvable target destination results in non-zero failure result."""
        args = parse_arguments(["--mock", "--instruction", "Move the red object to the nonexistent zone."])
        success, tel = execute_validation_run(args)

        self.assertFalse(success)
        self.assertEqual(tel["overall_result"], "FAILED")
        self.assertFalse(tel["physical_verification_passed"])
        self.assertIn("TASK_TERMINATED_INCOMPLETE", "\n".join(tel["event_trace"]))


if __name__ == "__main__":
    unittest.main()
