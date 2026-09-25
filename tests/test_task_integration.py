"""
SĀRTHI Phase 4B — Task Integration Test Suite.

Tests the complete pipeline:
  Natural language -> TaskUnderstanding -> Target Resolution
  -> Decision Engine -> Simulation -> Verification -> TaskExecutionResult

Tests:
  1.  Natural language produces valid TaskUnderstanding.
  2.  Complete nominal task: APPROACH -> GRASP -> MOVE -> RELEASE.
  3.  Unknown target object: structured failure, zero physical actions.
  4.  Unknown target location: structured failure, zero physical actions.
  5.  Model layer cannot execute physical actions.
  6.  Decision Engine remains physical action authority.
  7.  PATH_BLOCKED causes MOVE rejection and reassessment.
  8.  Recovery via REPOSITION completes task.
  9.  Identical instruction produces deterministic result.
  10. Existing tests still pass (import check).
"""

import unittest
from typing import Any, Dict, Optional

from simulation.core.geometry import SimDimensions3D, SimPoint3D, WorkspaceBounds
from simulation.core.robot import SimulatedRobot
from simulation.core.objects import SimulatedObject, SimulatedTargetZone
from simulation.core.events import DisturbanceEvent
from simulation.core.world import SimulationWorld
from simulation.adapters.base import LocalSimulationAdapter

from backend.app.decision_engine.models import ActionType
from backend.app.model.models import TaskUnderstanding
from backend.app.model.provider import ModelProvider
from backend.app.model.mock_provider import MockModelProvider
from backend.app.model.task_understanding import TaskUnderstandingService
from backend.app.orchestration.task_interpreter import TaskInterpreter
from backend.app.orchestration.execution_result import TaskExecutionResult
from backend.app.orchestration.task_runner import SarthiTaskRunner
from backend.app.orchestration.loop import TaskRunStatus


# ---------------------------------------------------------------------------
# Shared world-setup helper
# ---------------------------------------------------------------------------

def _make_standard_world():
    """
    Creates a clean, canonical test environment:
      - Robot at origin
      - red_object_01 flagged is_target=True at (0.25, 0.15, 0.20)
      - blue_target_zone at (0.40, -0.20, 0.20)
    """
    bounds = WorkspaceBounds(min_x=-0.8, max_x=0.8, min_y=-0.8, max_y=0.8, min_z=0.0, max_z=1.0)
    robot = SimulatedRobot(
        robot_id="sarthi_franka_arm",
        position=SimPoint3D(x=0.0, y=0.0, z=0.20),
        max_reach=0.85,
        max_payload_kg=3.0,
        workspace_limits=bounds,
    )
    world = SimulationWorld(robot=robot, workspace_bounds=bounds)

    red_obj = SimulatedObject(
        id="red_object_01",
        name="Red Cylinder",
        position=SimPoint3D(x=0.25, y=0.15, z=0.20),
        dimensions=SimDimensions3D(length_x=0.05, width_y=0.05, height_z=0.06),
        mass=0.5,
        is_target=True,
    )
    blue_target = SimulatedTargetZone(
        id="blue_target_zone",
        name="Blue Destination Zone",
        position=SimPoint3D(x=0.40, y=-0.20, z=0.20),
        tolerance_radius=0.06,
    )
    world.add_object(red_obj)
    world.add_target_zone(blue_target)
    adapter = LocalSimulationAdapter(world)
    return world, adapter


def _make_empty_world():
    """World with NO objects (no target object, no target zone objects)."""
    bounds = WorkspaceBounds(min_x=-0.8, max_x=0.8, min_y=-0.8, max_y=0.8, min_z=0.0, max_z=1.0)
    robot = SimulatedRobot(
        robot_id="sarthi_franka_arm",
        position=SimPoint3D(x=0.0, y=0.0, z=0.20),
        max_reach=0.85,
        max_payload_kg=3.0,
        workspace_limits=bounds,
    )
    world = SimulationWorld(robot=robot, workspace_bounds=bounds)
    # No objects, no target zones added
    adapter = LocalSimulationAdapter(world)
    return world, adapter


def _make_world_no_location():
    """
    World with target object but the target zone id does NOT match 'blue_target',
    so location resolution should fail.
    """
    bounds = WorkspaceBounds(min_x=-0.8, max_x=0.8, min_y=-0.8, max_y=0.8, min_z=0.0, max_z=1.0)
    robot = SimulatedRobot(
        robot_id="sarthi_franka_arm",
        position=SimPoint3D(x=0.0, y=0.0, z=0.20),
        max_reach=0.85,
        max_payload_kg=3.0,
        workspace_limits=bounds,
    )
    world = SimulationWorld(robot=robot, workspace_bounds=bounds)
    red_obj = SimulatedObject(
        id="red_object_01",
        name="Red Cylinder",
        position=SimPoint3D(x=0.25, y=0.15, z=0.20),
        dimensions=SimDimensions3D(length_x=0.05, width_y=0.05, height_z=0.06),
        mass=0.5,
        is_target=True,
    )
    world.add_object(red_obj)
    # Intentionally omit add_target_zone -- world uses the default zone id "default_zone"
    # which does NOT contain "blue_target"
    adapter = LocalSimulationAdapter(world)
    return world, adapter


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

class TestTaskIntegration(unittest.TestCase):
    """Phase 4B integration tests: natural language through physical execution."""

    # -------------------------------------------------------------------
    # Test 1: Natural language -> TaskUnderstanding
    # -------------------------------------------------------------------

    def test_01_natural_language_to_task_understanding(self):
        """Natural language instruction must produce a valid, structured TaskUnderstanding."""
        provider = MockModelProvider()
        service = TaskUnderstandingService(provider)
        interpreter = TaskInterpreter(service)

        understanding = interpreter.interpret("Move the red object to the blue target.")

        self.assertIsInstance(understanding, TaskUnderstanding)
        self.assertEqual(understanding.target_object, "red_object")
        self.assertEqual(understanding.target_location, "blue_target")
        self.assertEqual(
            understanding.required_actions,
            ["APPROACH", "GRASP", "MOVE", "RELEASE"],
        )
        self.assertGreaterEqual(understanding.confidence, 0.0)
        self.assertLessEqual(understanding.confidence, 1.0)
        self.assertTrue(understanding.task_id.startswith("task_"))

    # -------------------------------------------------------------------
    # Test 2: Complete nominal task
    # -------------------------------------------------------------------

    def test_02_complete_nominal_task(self):
        """
        run_instruction() with standard world must produce:
        APPROACH -> GRASP -> MOVE -> RELEASE and completed=True.
        """
        _, adapter = _make_standard_world()
        runner = SarthiTaskRunner(adapter=adapter, max_steps=10)

        result = runner.run_instruction(
            instruction="Move the red object to the blue target.",
            model_provider=MockModelProvider(),
        )

        self.assertIsInstance(result, TaskExecutionResult)
        self.assertTrue(result.completed, f"Task not completed. failure_reason={result.failure_reason}")
        self.assertEqual(
            result.executed_actions,
            ["APPROACH", "GRASP", "MOVE", "RELEASE"],
        )
        self.assertEqual(result.run_status, TaskRunStatus.COMPLETED.value)
        self.assertIsNone(result.failure_reason)
        self.assertIsNotNone(result.task_understanding)
        self.assertEqual(result.instruction, "Move the red object to the blue target.")

    # -------------------------------------------------------------------
    # Test 3: Unknown target object -> structured failure, zero physical actions
    # -------------------------------------------------------------------

    def test_03_unknown_target_object_structured_failure(self):
        """
        When the target object is NOT in the WorldState, run_instruction must return
        a structured failure with code TARGET_OBJECT_NOT_FOUND and execute ZERO physical actions.
        """
        _, adapter = _make_empty_world()
        runner = SarthiTaskRunner(adapter=adapter, max_steps=10)

        result = runner.run_instruction(
            instruction="Move the red object to the blue target.",
            model_provider=MockModelProvider(),
        )

        self.assertFalse(result.completed)
        self.assertIsNotNone(result.failure_reason)
        self.assertIn("TARGET_OBJECT_NOT_FOUND", result.failure_reason)
        # Zero physical actions must have been executed
        self.assertEqual(result.executed_actions, [])
        self.assertEqual(result.failed_actions, [])
        self.assertEqual(result.recovery_count, 0)

    # -------------------------------------------------------------------
    # Test 4: Unknown target location -> structured failure, zero physical actions
    # -------------------------------------------------------------------

    def test_04_unknown_target_location_structured_failure(self):
        """
        When the target location is NOT in the WorldState, run_instruction must return
        a structured failure with code TARGET_LOCATION_NOT_FOUND and execute ZERO physical actions.
        """
        _, adapter = _make_world_no_location()
        runner = SarthiTaskRunner(adapter=adapter, max_steps=10)

        result = runner.run_instruction(
            instruction="Move the red object to the blue target.",
            model_provider=MockModelProvider(),
        )

        self.assertFalse(result.completed)
        self.assertIsNotNone(result.failure_reason)
        self.assertIn("TARGET_LOCATION_NOT_FOUND", result.failure_reason)
        # Zero physical actions
        self.assertEqual(result.executed_actions, [])
        self.assertEqual(result.failed_actions, [])

    # -------------------------------------------------------------------
    # Test 5: Model cannot execute physical actions
    # -------------------------------------------------------------------

    def test_05_model_cannot_execute_physical_actions(self):
        """
        TaskUnderstanding, MockModelProvider, TaskInterpreter, and TaskUnderstandingService
        must NOT expose any execution or actuation methods.
        """
        prohibited_methods = [
            "execute", "execute_action", "actuate", "step", "move_motor",
            "set_position", "command", "apply_force", "inject_disturbance",
        ]
        model_classes = [
            TaskUnderstanding,
            MockModelProvider,
            TaskInterpreter,
            TaskUnderstandingService,
        ]
        for cls in model_classes:
            for method in prohibited_methods:
                self.assertFalse(
                    hasattr(cls, method),
                    f"'{cls.__name__}' violates model boundary: exposes '{method}'",
                )

    # -------------------------------------------------------------------
    # Test 6: Decision Engine is the physical action authority
    # -------------------------------------------------------------------

    def test_06_decision_engine_is_physical_action_authority(self):
        """
        The model layer (TaskUnderstanding, required_actions) must NEVER directly
        invoke execute_action(). Only SimulationAdapter.execute_action() via the
        Decision Engine is allowed.
        """
        provider = MockModelProvider()
        service = TaskUnderstandingService(provider)
        interpreter = TaskInterpreter(service)
        understanding = interpreter.interpret("Move the red object to the blue target.")

        # required_actions are semantic strings, NOT executable objects
        for action_str in understanding.required_actions:
            self.assertIsInstance(action_str, str,
                "required_actions entries must be plain strings, not executable objects")
            # Must not be callable action objects
            self.assertFalse(callable(action_str),
                f"required_action '{action_str}' must NOT be callable")

        # TaskUnderstanding does not expose execute_action
        self.assertFalse(hasattr(understanding, "execute_action"))
        self.assertFalse(hasattr(understanding, "execute"))

        # Confirm execute_action lives only on the adapter
        _, adapter = _make_standard_world()
        self.assertTrue(hasattr(adapter, "execute_action"),
            "execute_action must exist on SimulationAdapter")
        self.assertFalse(hasattr(understanding, "execute_action"),
            "execute_action must NOT exist on TaskUnderstanding")

    # -------------------------------------------------------------------
    # Test 7: PATH_BLOCKED causes MOVE rejection and reassessment
    # -------------------------------------------------------------------

    def test_07_path_blocked_causes_move_rejection(self):
        """
        After GRASP, injecting PATH_BLOCKED must cause the Decision Engine to
        reject the direct MOVE and select an alternative (REPOSITION).
        recovery_count must be >= 1.
        """
        world, adapter = _make_standard_world()
        runner = SarthiTaskRunner(adapter=adapter, max_steps=12)

        def inject_after_grasp(step_number: int, world_state):
            if step_number == 2:  # After GRASP
                dist = DisturbanceEvent.create_path_blocked(
                    event_id="dist_test_7",
                    obstacle_id="blocking_barrier_07",
                    position=SimPoint3D(x=0.325, y=-0.025, z=0.12),
                    dimensions=SimDimensions3D(length_x=0.06, width_y=0.06, height_z=0.08),
                )
                adapter.inject_disturbance(dist)

        result = runner.run_instruction(
            instruction="Move the red object to the blue target.",
            model_provider=MockModelProvider(),
            step_callback=inject_after_grasp,
        )

        # The Decision Engine must have rejected MOVE and selected REPOSITION
        self.assertIn("REPOSITION", result.executed_actions,
            "REPOSITION must appear after PATH_BLOCKED -- Decision Engine must reassess")
        # recovery_count tracks REPOSITION steps
        self.assertGreaterEqual(result.recovery_count, 1)

    # -------------------------------------------------------------------
    # Test 8: Recovery completes task
    # -------------------------------------------------------------------

    def test_08_recovery_completes_task(self):
        """
        Even with PATH_BLOCKED disturbance injected after GRASP,
        the task must reach completed=True via the REPOSITION recovery path.
        Expected trace: APPROACH -> GRASP -> REPOSITION -> MOVE -> RELEASE.
        """
        world, adapter = _make_standard_world()
        runner = SarthiTaskRunner(adapter=adapter, max_steps=12)

        def inject_after_grasp(step_number: int, world_state):
            if step_number == 2:
                dist = DisturbanceEvent.create_path_blocked(
                    event_id="dist_test_8",
                    obstacle_id="blocking_barrier_08",
                    position=SimPoint3D(x=0.325, y=-0.025, z=0.12),
                    dimensions=SimDimensions3D(length_x=0.06, width_y=0.06, height_z=0.08),
                )
                adapter.inject_disturbance(dist)

        result = runner.run_instruction(
            instruction="Move the red object to the blue target.",
            model_provider=MockModelProvider(),
            step_callback=inject_after_grasp,
        )

        self.assertTrue(result.completed,
            f"Task must complete despite PATH_BLOCKED. failure={result.failure_reason}")
        self.assertIn("APPROACH", result.executed_actions)
        self.assertIn("GRASP", result.executed_actions)
        self.assertIn("REPOSITION", result.executed_actions)
        self.assertIn("MOVE", result.executed_actions)
        self.assertIn("RELEASE", result.executed_actions)
        self.assertGreaterEqual(result.recovery_count, 1)
        self.assertEqual(result.run_status, TaskRunStatus.COMPLETED.value)

    # -------------------------------------------------------------------
    # Test 9: Deterministic result for identical instruction
    # -------------------------------------------------------------------

    def test_09_identical_instruction_produces_deterministic_result(self):
        """
        Two independent runs with the same instruction and identical initial world
        must produce bit-identical task_id, executed_actions, and recovery_count.
        """
        def run_once():
            _, adapter = _make_standard_world()
            runner = SarthiTaskRunner(adapter=adapter, max_steps=10)
            return runner.run_instruction(
                instruction="Move the red object to the blue target.",
                model_provider=MockModelProvider(),
            )

        r1 = run_once()
        r2 = run_once()

        self.assertEqual(r1.task_id, r2.task_id,
            "task_id must be deterministic for identical instruction")
        self.assertEqual(r1.executed_actions, r2.executed_actions,
            "Action sequence must be deterministic")
        self.assertEqual(r1.recovery_count, r2.recovery_count)
        self.assertEqual(r1.completed, r2.completed)
        self.assertEqual(r1.run_status, r2.run_status)
        self.assertEqual(
            r1.task_understanding.model_dump(),
            r2.task_understanding.model_dump(),
            "TaskUnderstanding must be deterministic for identical instruction",
        )

    # -------------------------------------------------------------------
    # Test 10: Existing tests still importable (regression guard)
    # -------------------------------------------------------------------

    def test_10_existing_modules_still_importable(self):
        """
        All existing Phase 3B and Phase 4A modules must remain importable
        without modification errors.
        """
        # Phase 3B orchestration
        from backend.app.orchestration.loop import ClosedLoopCycle, TaskRunResult, TaskRunStatus
        from backend.app.orchestration.task_runner import SarthiTaskRunner
        # Phase 3A decision engine
        from backend.app.decision_engine.engine import SarthiDecisionEngine
        from backend.app.decision_engine.models import WorldState, ActionType
        # Phase 4A model layer
        from backend.app.model.models import TaskUnderstanding
        from backend.app.model.mock_provider import MockModelProvider
        from backend.app.model.task_understanding import TaskUnderstandingService
        # Phase 4B new modules
        from backend.app.orchestration.task_interpreter import TaskInterpreter
        from backend.app.orchestration.execution_result import TaskExecutionResult

        # All imports succeeded -- no regression
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
