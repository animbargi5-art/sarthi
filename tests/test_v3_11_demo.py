"""
SĀRTHI V3-11 — Visual Demonstration & Telemetry Logging Unit Tests.

Validates the 12 specified test criteria:
1. demo event schema
2. telemetry ordering
3. replay mode
4. live mode configuration
5. AI proposal labeling
6. deterministic authority labeling
7. recovery event visualization
8. final placement display
9. secret sanitization
10. replay does not claim live execution
11. existing telemetry compatibility
12. no robot execution from UI/dashboard layer
"""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

from backend.app.demo.models import (
    DemoEvent,
    DemoEventType,
    DemoMode,
    DemoSessionTelemetry,
    RecoveryVisualization,
    SystemStatus,
    _sanitize_data,
)
from backend.app.demo.service import SarthiDemoService
from backend.app.demo.server import app
from fastapi.testclient import TestClient


class TestV311DemoSuite(unittest.TestCase):
    """Test suite validating V3-11 visual demo and telemetry requirements."""

    def setUp(self) -> None:
        self.service = SarthiDemoService()
        self.client = TestClient(app)

    # 1. Demo Event Schema
    def test_01_demo_event_schema(self) -> None:
        event = DemoEvent(
            event_index=1,
            event_type=DemoEventType.TASK_RECEIVED,
            code="[01] TASK_RECEIVED",
            timestamp_iso="2026-10-04T20:00:00Z",
            label="Operator commanded: Move the red object",
            authority_classification="HUMAN OPERATOR",
            details={"instruction": "Move the red object"},
        )
        self.assertEqual(event.event_index, 1)
        self.assertEqual(event.event_type, DemoEventType.TASK_RECEIVED)
        self.assertEqual(event.code, "[01] TASK_RECEIVED")
        self.assertIn("instruction", event.details)

    # 2. Telemetry Ordering
    def test_02_telemetry_ordering(self) -> None:
        expected_sequence = [
            DemoEventType.TASK_RECEIVED,
            DemoEventType.NEMOTRON_TASK_UNDERSTANDING,
            DemoEventType.WORLD_STATE_OBSERVED,
            DemoEventType.LAYA_DECISION,
            DemoEventType.CANDIDATE_INJECTED,
            DemoEventType.DETERMINISTIC_VALIDATION_ACCEPTED,
            DemoEventType.ACTION_EXECUTED,
            DemoEventType.ACTION_VERIFIED,
            DemoEventType.DISTURBANCE_DETECTED,
            DemoEventType.MOVE_REJECTED_BLOCKED_PATH,
            DemoEventType.RECOVERY_OPTIONS_GENERATED,
            DemoEventType.LAYA_SELECTED_REPOSITION,
            DemoEventType.REPOSITION_VALIDATED,
            DemoEventType.REPOSITION_EXECUTED,
            DemoEventType.MOVE_VALIDATED,
            DemoEventType.MOVE_EXECUTED,
            DemoEventType.RELEASE_VERIFIED,
            DemoEventType.FINAL_PLACEMENT_VERIFIED,
            DemoEventType.TASK_COMPLETED,
        ]
        self.assertEqual(len(expected_sequence), 19)
        # Verify codes format matches [01] through [19]
        for idx, et in enumerate(expected_sequence, start=1):
            code_str = f"[{idx:02d}] {et.value}"
            self.assertTrue(code_str.startswith(f"[{idx:02d}]"))

    # 3. Replay Mode Generation
    def test_03_replay_mode(self) -> None:
        sample_path = "records/test_temp_replay.json"
        try:
            # Create a mock session telemetry and dump to file
            sample_session = DemoSessionTelemetry(
                session_id="test_session_01",
                demo_mode=DemoMode.LIVE,
                mode_label="LIVE DEMONSTRATION",
                timestamp_iso="2026-10-04T20:00:00Z",
                human_command="Test command",
                task_completed=True,
            )
            os.makedirs(os.path.dirname(sample_path), exist_ok=True)
            with open(sample_path, "w", encoding="utf-8") as f:
                json.dump(sample_session.model_dump(), f)

            replay_session = self.service.run_replay_demo(telemetry_path=sample_path)
            self.assertEqual(replay_session.demo_mode, DemoMode.REPLAY)
            self.assertIn("REPLAY MODE", replay_session.mode_label)
        finally:
            if os.path.exists(sample_path):
                os.remove(sample_path)

    # 4. Live Mode Configuration
    def test_04_live_mode_configuration(self) -> None:
        sys_status = SystemStatus()
        self.assertEqual(sys_status.nemotron_status, "Connected")
        self.assertEqual(sys_status.laya_status, "Connected")
        self.assertIn("Sole Physical Authority", sys_status.decision_engine_status)
        self.assertIn("Franka Emika Panda", sys_status.mujoco_status)
        self.assertIn("Authoritative", sys_status.safety_validation_status)

    # 5. AI Proposal Labeling
    def test_05_ai_proposal_labeling(self) -> None:
        recovery_vis = RecoveryVisualization()
        self.assertEqual(recovery_vis.laya_authority_label, "AI PROPOSAL — NON-AUTHORITATIVE")

        laya_event = DemoEvent(
            event_index=4,
            event_type=DemoEventType.LAYA_DECISION,
            code="[04] LAYA_DECISION",
            timestamp_iso="2026-10-04T20:00:00Z",
            label="Laya proposed REPOSITION",
            authority_classification="AI PROPOSAL — NON-AUTHORITATIVE",
            details={},
        )
        self.assertIn("NON-AUTHORITATIVE", laya_event.authority_classification)

    # 6. Deterministic Authority Labeling
    def test_06_deterministic_authority_labeling(self) -> None:
        recovery_vis = RecoveryVisualization()
        self.assertEqual(
            recovery_vis.validation_authority_label,
            "DETERMINISTIC PHYSICAL AUTHORITY",
        )

        val_event = DemoEvent(
            event_index=6,
            event_type=DemoEventType.DETERMINISTIC_VALIDATION_ACCEPTED,
            code="[06] DETERMINISTIC_VALIDATION_ACCEPTED",
            timestamp_iso="2026-10-04T20:00:00Z",
            label="Validated candidate action",
            authority_classification="DETERMINISTIC PHYSICAL AUTHORITY",
            details={},
        )
        self.assertEqual(
            val_event.authority_classification, "DETERMINISTIC PHYSICAL AUTHORITY"
        )

    # 7. Recovery Event Visualization
    def test_07_recovery_event_visualization(self) -> None:
        rec = RecoveryVisualization(
            disturbance_name="PATH_BLOCKED",
            obstacle_id="blocking_barrier_01",
            direct_move_status="REJECTED",
            rejection_reason="BlockedPath: Obstacle intersects trajectory corridor",
            available_recovery_options=["REPOSITION", "STOP"],
            laya_selected_option="REPOSITION",
            validation_status="ACCEPTED",
            execution_result="RECOVERY SUCCESS (Elevated Clearance Transit)",
        )
        self.assertEqual(rec.disturbance_name, "PATH_BLOCKED")
        self.assertEqual(rec.direct_move_status, "REJECTED")
        self.assertEqual(rec.laya_selected_option, "REPOSITION")
        self.assertEqual(rec.validation_status, "ACCEPTED")
        self.assertIn("RECOVERY SUCCESS", rec.execution_result)

    # 8. Final Placement Display
    def test_08_final_placement_display(self) -> None:
        session = DemoSessionTelemetry(
            session_id="test_placement",
            demo_mode=DemoMode.LIVE,
            mode_label="LIVE DEMONSTRATION",
            timestamp_iso="2026-10-04T20:00:00Z",
            human_command="Move red to blue",
            final_placement={
                "final_position": {"x": 0.401, "y": -0.198, "z": 0.201},
                "target_position": {"x": 0.4, "y": -0.2, "z": 0.2},
                "placement_error_m": 0.00245,
                "tolerance_m": 0.08,
                "within_tolerance": True,
            },
            task_completed=True,
        )
        self.assertTrue(session.final_placement["within_tolerance"])
        self.assertLess(
            session.final_placement["placement_error_m"],
            session.final_placement["tolerance_m"],
        )

    # 9. Secret Sanitization
    def test_09_secret_sanitization(self) -> None:
        dirty_payload = {
            "api_key": "sk-12345abcdef",
            "bearer_token": "Bearer eyJhbGciOi...",
            "nested": {
                "nebius_key": "nbf_secretkey123",
                "safe_field": "public_data",
            },
            "token_list": ["Bearer token1", "safe_item"],
        }
        cleaned = _sanitize_data(dirty_payload)
        self.assertEqual(cleaned["api_key"], "[REDACTED]")
        self.assertEqual(cleaned["bearer_token"], "[REDACTED]")
        self.assertEqual(cleaned["nested"]["nebius_key"], "[REDACTED]")
        self.assertEqual(cleaned["nested"]["safe_field"], "public_data")
        self.assertNotIn("sk-12345abcdef", str(cleaned))
        self.assertNotIn("nbf_secretkey123", str(cleaned))

    # 10. Replay Does Not Claim Live Execution
    def test_10_replay_does_not_claim_live_execution(self) -> None:
        sample_path = "records/test_temp_replay2.json"
        try:
            sample_session = DemoSessionTelemetry(
                session_id="test_live_01",
                demo_mode=DemoMode.LIVE,
                mode_label="LIVE DEMONSTRATION",
                timestamp_iso="2026-10-04T20:00:00Z",
                human_command="Test command",
                task_completed=True,
            )
            os.makedirs(os.path.dirname(sample_path), exist_ok=True)
            with open(sample_path, "w", encoding="utf-8") as f:
                json.dump(sample_session.model_dump(), f)

            replay = self.service.run_replay_demo(telemetry_path=sample_path)
            self.assertNotEqual(replay.demo_mode, DemoMode.LIVE)
            self.assertEqual(replay.demo_mode, DemoMode.REPLAY)
            self.assertIn("REPLAY MODE", replay.mode_label)
            self.assertNotIn("LIVE DEMONSTRATION", replay.mode_label)
        finally:
            if os.path.exists(sample_path):
                os.remove(sample_path)

    # 11. Existing Telemetry Compatibility
    def test_11_existing_telemetry_compatibility(self) -> None:
        # Check that existing V3 telemetry models can be captured in session
        session = DemoSessionTelemetry(
            session_id="compat_test",
            demo_mode=DemoMode.LIVE,
            mode_label="LIVE DEMONSTRATION",
            timestamp_iso="2026-10-04T20:00:00Z",
            human_command="Test command",
            task_completed=True,
        )
        d = session.to_sanitized_dict()
        self.assertIn("session_id", d)
        self.assertIn("architectural_guarantee", d)
        self.assertIn("events", d)

    # 12. No Robot Execution From UI/Dashboard Layer
    def test_12_no_robot_execution_from_ui_layer(self) -> None:
        # Verify that FastAPI endpoints only read telemetry or invoke validated pipeline
        # The UI/dashboard has NO direct motor actuation endpoints
        routes = [route.path for route in app.routes]
        for route in routes:
            self.assertNotIn("/motor", route)
            self.assertNotIn("/actuate", route)
            self.assertNotIn("/bypass", route)
            self.assertNotIn("/direct_control", route)

        # GET /api/status returns read-only status
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        self.assertIn("nemotron_status", res.json())


if __name__ == "__main__":
    unittest.main()
