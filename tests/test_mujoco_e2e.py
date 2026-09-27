"""
SĀRTHI Phase F — MuJoCo End-to-End Closed-Loop Integration + Disturbance Recovery Tests.

Tests the full autonomous closed-loop orchestration stack with MuJoCo physics:
  Natural language
  -> TaskUnderstandingService (via ModelProvider boundary)
  -> Canonical WorldState (observed from live MuJoCo mjData)
  -> SarthiDecisionEngine (deterministic candidate action evaluation)
  -> SarthiMuJoCoRuntime (SimulationAdapter contract & physical Panda physics)
  -> Disturbance injection (PATH_BLOCKED post-grasp)
  -> Autonomous recovery replanning (Decision Engine selects REPOSITION)
  -> Execution to completion (MOVE -> RELEASE)
  -> Physical placement verification (measured coordinates within tolerance)

Strict Architectural Constraints:
- Zero cognitive or replanning logic inside simulation/articulation/runtime.
- SarthiDecisionEngine is the sole action authority.
- No hardcoded recovery shortcuts ("if blocked: action = REPOSITION").
- Deterministic offline tests use MockModelProvider (live Nebius test run separately).
- Zero secret or API-key leakage in telemetry, events, or execution results.
"""

import math
import unittest

from backend.app.decision_engine.engine import SarthiDecisionEngine
from backend.app.decision_engine.models import (
    ActionType,
    CandidateAction,
    ObjectState,
    Point3D,
    WorldState,
)
from backend.app.model.mock_provider import MockModelProvider
from backend.app.model.models import TaskUnderstanding
from backend.app.model.task_understanding import TaskUnderstandingService
from backend.app.orchestration.execution_result import TaskExecutionResult
from backend.app.orchestration.loop import TaskRunStatus
from backend.app.orchestration.task_runner import SarthiTaskRunner
from simulation.adapters.base import SimulationAdapter
from simulation.mujoco_runtime import require_mujoco
from simulation.mujoco_runtime.runtime import SarthiMuJoCoRuntime


class TestSarthiMuJoCoEndToEnd(unittest.TestCase):
    """
    Phase F Test Suite: Validates closed-loop integration of SarthiTaskRunner,
    Decision Engine, and SarthiMuJoCoRuntime with dynamic obstacle recovery.
    """

    @classmethod
    def setUpClass(cls):
        require_mujoco()

    def setUp(self):
        """Initializes a fresh, deterministic MuJoCo runtime and TaskRunner."""
        self.runtime = SarthiMuJoCoRuntime(
            config={"steps_per_action": 400, "convergence_tolerance": 0.03}
        )
        self.engine = SarthiDecisionEngine()
        self.runner = SarthiTaskRunner(adapter=self.runtime, engine=self.engine, max_steps=10)
        self.provider = MockModelProvider()
        self.instruction = "Move the red object to the blue target."

    # -----------------------------------------------------------------------
    # Test 1: TaskRunner accepts MuJoCo SimulationAdapter
    # -----------------------------------------------------------------------
    def test_01_task_runner_accepts_mujoco_simulation_adapter(self):
        """1. TaskRunner accepts SarthiMuJoCoRuntime as a compliant SimulationAdapter."""
        self.assertIsInstance(self.runner.adapter, SimulationAdapter)
        self.assertIsInstance(self.runner.adapter, SarthiMuJoCoRuntime)
        self.assertTrue(callable(getattr(self.runner.adapter, "get_world_state", None)))
        self.assertTrue(callable(getattr(self.runner.adapter, "execute_action", None)))
        self.assertTrue(callable(getattr(self.runner.adapter, "inject_disturbance", None)))
        self.assertTrue(callable(getattr(self.runner.adapter, "verify_action_result", None)))

    # -----------------------------------------------------------------------
    # Test 2: End-to-end orchestration starts correctly
    # -----------------------------------------------------------------------
    def test_02_e2e_orchestration_starts_correctly(self):
        """2. End-to-end orchestration initializes and begins cycle execution."""
        initial_ws = self.runtime.get_world_state()
        self.assertIsNotNone(initial_ws)
        self.assertEqual(initial_ws.version, 1)
        self.assertFalse(initial_ws.robot.is_moving)
        self.assertIsNone(initial_ws.robot.holding_object_id)

        # Execute first step manually to verify initial action dispatch
        decision = self.engine.decide(initial_ws)
        self.assertIsNotNone(decision.selected_action)
        self.assertEqual(decision.selected_action.action_type, ActionType.APPROACH)

        res = self.runtime.execute_action(decision.selected_action)
        self.assertTrue(res.success)
        self.assertEqual(res.new_world_state_version, 2)

    # -----------------------------------------------------------------------
    # Test 3: Task understanding is connected through existing provider boundary
    # -----------------------------------------------------------------------
    def test_03_task_understanding_connected_through_provider_boundary(self):
        """3. TaskUnderstandingService parses natural language via ModelProvider boundary."""
        service = TaskUnderstandingService(self.provider)
        tu = service.understand(self.instruction)

        self.assertIsInstance(tu, TaskUnderstanding)
        self.assertEqual(tu.target_object, "red_object")
        self.assertEqual(tu.target_location, "blue_target")
        self.assertGreater(tu.confidence, 0.5)

    # -----------------------------------------------------------------------
    # Test 4: WorldState is read from physical MuJoCo state
    # -----------------------------------------------------------------------
    def test_04_world_state_read_from_physical_mujoco_state(self):
        """4. Canonical WorldState matches actual MuJoCo physical coordinates."""
        ws = self.runtime.get_world_state()
        sr = self.runtime.state_reader

        # Robot position matches end-effector
        ee_pos = sr.get_end_effector_position()
        self.assertAlmostEqual(ws.robot.position.x, ee_pos.x, places=3)
        self.assertAlmostEqual(ws.robot.position.y, ee_pos.y, places=3)
        self.assertAlmostEqual(ws.robot.position.z, ee_pos.z, places=3)

        # Target object matches physical red object geom/body
        red_obj = next(o for o in ws.objects if o.is_target)
        phys_obj = sr.get_red_object_position()
        self.assertAlmostEqual(red_obj.position.x, phys_obj.x, places=3)
        self.assertAlmostEqual(red_obj.position.y, phys_obj.y, places=3)
        self.assertAlmostEqual(red_obj.position.z, phys_obj.z, places=3)

        # Target zone matches blue target
        tgt_pos = sr.get_blue_target_position()
        self.assertAlmostEqual(ws.target.position.x, tgt_pos.x, places=3)
        self.assertAlmostEqual(ws.target.position.y, tgt_pos.y, places=3)

    # -----------------------------------------------------------------------
    # Test 5: PATH_BLOCKED changes WorldState
    # -----------------------------------------------------------------------
    def test_05_path_blocked_changes_physical_world_state(self):
        """5. Calling inject_disturbance(PATH_BLOCKED) modifies MuJoCo physical scene and WorldState."""
        ws_before = self.runtime.get_world_state()
        obs_before = [o for o in ws_before.objects if o.is_obstacle]
        self.assertEqual(len(obs_before), 0)

        # Inject disturbance
        success = self.runtime.inject_disturbance("PATH_BLOCKED")
        self.assertTrue(success)

        # Verify physical scene changed
        ws_after = self.runtime.get_world_state()
        self.assertGreater(ws_after.version, ws_before.version)
        obs_after = [o for o in ws_after.objects if o.is_obstacle]
        self.assertEqual(len(obs_after), 1)

        # Obstacle is positioned at canonical workspace blocking coordinates
        obs = obs_after[0]
        self.assertAlmostEqual(obs.position.x, 0.325, places=2)
        self.assertAlmostEqual(obs.position.y, -0.025, places=2)
        self.assertAlmostEqual(obs.position.z, 0.12, places=2)

        # Keepout constraint generated
        constraints = self.runtime.state_reader.get_active_constraints()
        self.assertTrue(any("obstacle" in c.constraint_id.lower() for c in constraints))

    # -----------------------------------------------------------------------
    # Test 6: Decision Engine receives changed state
    # -----------------------------------------------------------------------
    def test_06_decision_engine_receives_changed_state(self):
        """6. Decision Engine receives the disturbed WorldState with obstacle object."""
        self.runtime.inject_disturbance("PATH_BLOCKED")
        disturbed_ws = self.runtime.get_world_state()

        # Decision Engine evaluates the disturbed WorldState
        decision = self.engine.decide(disturbed_ws)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.world_state_version, disturbed_ws.version)

    # -----------------------------------------------------------------------
    # Test 7: Blocked candidate action is rejected by constraints
    # -----------------------------------------------------------------------
    def test_07_blocked_candidate_action_rejected_by_constraints(self):
        """7. Nominal ground-level transit candidate is rejected when obstacle is active."""
        # 1. Approach and grasp object
        self.runtime.execute_action(
            CandidateAction(
                action_type=ActionType.APPROACH,
                action_id="act_app",
                target_position=Point3D(x=0.25, y=0.15, z=0.20),
            )
        )
        self.runtime.execute_action(
            CandidateAction(
                action_type=ActionType.GRASP,
                action_id="act_gr",
                target_position=Point3D(x=0.25, y=0.15, z=0.20),
                target_object_id="red_object_01",
            )
        )

        # 2. Inject disturbance
        self.runtime.inject_disturbance("PATH_BLOCKED")
        disturbed_ws = self.runtime.get_world_state()

        # 3. Evaluate candidate actions
        decision = self.engine.decide(disturbed_ws)

        # Find direct move candidate
        direct_move = next(
            (c for c in decision.candidate_evaluations if c.action.action_type == ActionType.MOVE),
            None,
        )
        self.assertIsNotNone(direct_move)
        # Direct move intersects obstacle and MUST be rejected as invalid
        self.assertFalse(direct_move.is_valid)
        reasons_text = " ".join(direct_move.rejection_reasons).lower()
        self.assertTrue(
            "obstacle" in reasons_text or "blocked" in reasons_text or "collision" in reasons_text
        )

    # -----------------------------------------------------------------------
    # Test 8: Recovery action is selected by Decision Engine
    # -----------------------------------------------------------------------
    def test_08_recovery_action_selected_by_decision_engine(self):
        """8. Decision Engine selects REPOSITION autonomously (no integration shortcut)."""
        # Execute approach + grasp
        self.runtime.execute_action(
            CandidateAction(
                action_type=ActionType.APPROACH,
                action_id="act_app",
                target_position=Point3D(x=0.25, y=0.15, z=0.20),
            )
        )
        self.runtime.execute_action(
            CandidateAction(
                action_type=ActionType.GRASP,
                action_id="act_gr",
                target_position=Point3D(x=0.25, y=0.15, z=0.20),
                target_object_id="red_object_01",
            )
        )
        self.runtime.inject_disturbance("PATH_BLOCKED")
        ws = self.runtime.get_world_state()

        decision = self.engine.decide(ws)
        # Deterministic evaluation selects recovery action REPOSITION
        self.assertEqual(decision.selected_action.action_type, ActionType.REPOSITION)
        self.assertTrue(decision.selected_action.target_position.z > 0.30)

    # -----------------------------------------------------------------------
    # Test 9: Recovery action reaches the runtime and executes
    # -----------------------------------------------------------------------
    def test_09_recovery_action_reaches_runtime_and_executes(self):
        """9. Selected REPOSITION action reaches MuJoCo runtime and physically lifts the object."""
        self.runtime.execute_action(
            CandidateAction(
                action_type=ActionType.APPROACH,
                action_id="act_app",
                target_position=Point3D(x=0.25, y=0.15, z=0.20),
            )
        )
        self.runtime.execute_action(
            CandidateAction(
                action_type=ActionType.GRASP,
                action_id="act_gr",
                target_position=Point3D(x=0.25, y=0.15, z=0.20),
                target_object_id="red_object_01",
            )
        )
        self.runtime.inject_disturbance("PATH_BLOCKED")
        ws = self.runtime.get_world_state()

        decision = self.engine.decide(ws)
        repo_action = decision.selected_action
        self.assertEqual(repo_action.action_type, ActionType.REPOSITION)

        # Dispatch to runtime
        result = self.runtime.execute_action(repo_action)
        self.assertTrue(result.success)
        self.assertEqual(result.action_type, "REPOSITION")

        # Physical elevation check
        ee = self.runtime.state_reader.get_end_effector_position()
        self.assertGreater(ee.z, 0.30)
        self.assertTrue(self.runtime.state_reader.is_object_grasped())

    # -----------------------------------------------------------------------
    # Test 10: Full E2E closed-loop execution and physical verification
    # -----------------------------------------------------------------------
    def test_10_full_e2e_closed_loop_and_physical_verification(self):
        """10. Full closed-loop execution completes with physical object placement inside tolerance."""
        def disturbance_cb(step_num: int, ws: WorldState):
            if step_num == 2:
                # Disturbance injected post-grasp
                self.runtime.inject_disturbance("PATH_BLOCKED")

        result: TaskExecutionResult = self.runner.run_instruction(
            self.instruction,
            model_provider=self.provider,
            step_callback=disturbance_cb,
        )

        # Orchestration level verification
        self.assertTrue(result.completed)
        self.assertEqual(result.run_status, TaskRunStatus.COMPLETED.value)
        self.assertEqual(result.recovery_count, 1)
        self.assertEqual(
            result.executed_actions,
            ["APPROACH", "GRASP", "REPOSITION", "MOVE", "RELEASE"],
        )

        # Ground-truth physical verification from MuJoCo mjData
        sr = self.runtime.state_reader
        final_obj_pos = sr.get_red_object_position()
        tgt_pos = sr.get_blue_target_position()
        tol = sr.get_blue_target_tolerance()

        h_dist = math.sqrt(
            (final_obj_pos.x - tgt_pos.x) ** 2 + (final_obj_pos.y - tgt_pos.y) ** 2
        )
        self.assertLessEqual(
            h_dist,
            tol,
            f"Object final position ({final_obj_pos.x:.3f}, {final_obj_pos.y:.3f}) exceeds tolerance {tol:.3f}m. Dist: {h_dist:.4f}m",
        )
        self.assertEqual(sr.determine_object_state(), ObjectState.PLACED)
        self.assertFalse(sr.is_object_grasped())

    # -----------------------------------------------------------------------
    # Test 11: No hidden hardcoded recovery selection
    # -----------------------------------------------------------------------
    def test_11_no_hidden_hardcoded_recovery_selection(self):
        """11. Verifies recovery action is decided exclusively through candidate evaluation."""
        self.runtime.inject_disturbance("PATH_BLOCKED")
        disturbed_ws = self.runtime.get_world_state()

        # Engine produces evaluations for all candidates
        decision = self.engine.decide(disturbed_ws)
        self.assertGreater(len(decision.candidate_evaluations), 1)

        # Every candidate has an explicit score and validity flag
        for eval_res in decision.candidate_evaluations:
            self.assertIsInstance(eval_res.is_valid, bool)
            self.assertIsInstance(eval_res.overall_score, float)

        # Selected action is the highest-ranked valid candidate
        valid_candidates = [c for c in decision.candidate_evaluations if c.is_valid]
        best_candidate = max(valid_candidates, key=lambda c: c.overall_score)
        self.assertEqual(decision.selected_action.action_id, best_candidate.action.action_id)

    # -----------------------------------------------------------------------
    # Test 12: Zero secret or API-key leakage in telemetry
    # -----------------------------------------------------------------------
    def test_12_no_secret_or_api_key_leakage(self):
        """12. Telemetry, event logs, and string outputs never contain secrets or API keys."""
        forbidden_tokens = ["NEBIUS_API_KEY", "Bearer", "secret", "sk-"]

        def check_no_leak(text: str):
            for token in forbidden_tokens:
                self.assertNotIn(token, text)

        # Check execution history details
        for res in self.runtime.execution_history:
            check_no_leak(str(res.details))
            check_no_leak(str(res.failure_reason))

        # Check task runner events
        for event in self.runner.events:
            check_no_leak(str(event.payload))
            check_no_leak(str(event.message))

    # -----------------------------------------------------------------------
    # Test 13: Failure propagation on missing target object
    # -----------------------------------------------------------------------
    def test_13_failure_propagation_missing_target(self):
        """13. Non-existent target location terminates cleanly with structured diagnostic."""
        bogus_instruction = "Move the red object to the golden altar."
        result = self.runner.run_instruction(
            bogus_instruction,
            model_provider=self.provider,
        )

        self.assertFalse(result.completed)
        self.assertIn("TARGET_LOCATION_NOT_FOUND", str(result.failure_reason))
        # Zero physical actions executed in MuJoCo
        self.assertEqual(len(result.executed_actions), 0)
        self.assertEqual(len(self.runtime.execution_history), 0)


if __name__ == "__main__":
    unittest.main()
