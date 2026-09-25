"""
SĀRTHI Model Integration — Unit Test Suite.
Validates provider abstraction, deterministic mock outputs, Pydantic validation boundaries,
and structural decoupling between semantic language understanding and physical execution.
"""

import unittest
from typing import Any, Dict, Optional
from pydantic import ValidationError

from backend.app.model.models import TaskUnderstanding
from backend.app.model.provider import ModelProvider
from backend.app.model.mock_provider import MockModelProvider
from backend.app.model.task_understanding import TaskUnderstandingService
from backend.app.decision_engine.models import Point3D, RobotState, WorldState, TargetZone, EnvironmentState, LastActionOutcome


class TestModelIntegration(unittest.TestCase):
    """Test suite covering Phase 4A model integration boundary."""

    def test_valid_structured_task_understanding(self):
        """Verify standard instruction parses into expected structured TaskUnderstanding fields."""
        provider = MockModelProvider()
        service = TaskUnderstandingService(provider)

        instruction = "Move the red object to the blue target."
        understanding = service.understand(instruction)

        self.assertIsInstance(understanding, TaskUnderstanding)
        self.assertEqual(understanding.objective, "Move the red object to the blue target.")
        self.assertEqual(understanding.target_object, "red_object")
        self.assertEqual(understanding.target_location, "blue_target")
        self.assertEqual(
            understanding.required_actions,
            ["APPROACH", "GRASP", "MOVE", "RELEASE"]
        )
        self.assertGreaterEqual(understanding.confidence, 0.0)
        self.assertLessEqual(understanding.confidence, 1.0)
        self.assertGreater(len(understanding.constraints), 0)
        self.assertGreater(len(understanding.success_conditions), 0)
        self.assertGreater(len(understanding.reasoning_summary), 0)

    def test_invalid_provider_output_rejected(self):
        """Verify that malformed or non-compliant provider output is rejected by Pydantic validation."""
        # 1. Provider returning invalid confidence (> 1.0)
        class OutOfRangeConfidenceProvider(ModelProvider):
            def understand_task(self, instruction: str, world_context: Optional[Dict[str, Any]] = None):
                return {
                    "task_id": "bad_task",
                    "objective": "invalid confidence test",
                    "confidence": 1.5,  # Violates le=1.0
                    "reasoning_summary": "too confident",
                }

        service_bad_conf = TaskUnderstandingService(OutOfRangeConfidenceProvider())
        with self.assertRaises(ValidationError):
            service_bad_conf.understand("any instruction")

        # 2. Provider returning missing required fields
        class MissingFieldsProvider(ModelProvider):
            def understand_task(self, instruction: str, world_context: Optional[Dict[str, Any]] = None):
                return {
                    "target_object": "red_box"
                    # missing task_id, objective, confidence, reasoning_summary
                }

        service_missing = TaskUnderstandingService(MissingFieldsProvider())
        with self.assertRaises(ValidationError):
            service_missing.understand("any instruction")

        # 3. Provider returning wrong type (e.g. int or string)
        class MalformedTypeProvider(ModelProvider):
            def understand_task(self, instruction: str, world_context: Optional[Dict[str, Any]] = None):
                return "Not a valid model output"

        service_wrong_type = TaskUnderstandingService(MalformedTypeProvider())
        with self.assertRaises(TypeError):
            service_wrong_type.understand("any instruction")

    def test_deterministic_mock_provider(self):
        """Verify that multiple invocations of MockModelProvider produce 100% bit-for-bit identical outputs."""
        provider = MockModelProvider()
        service = TaskUnderstandingService(provider)

        instruction = "Move the red object to the blue target."
        res1 = service.understand(instruction)
        res2 = service.understand(instruction)
        res3 = service.understand(instruction)

        self.assertEqual(res1, res2)
        self.assertEqual(res2, res3)
        self.assertEqual(res1.model_dump(), res2.model_dump())

    def test_service_uses_provider_correctly(self):
        """Verify service properly dispatches instruction and context to underlying provider."""
        captured_calls = []

        class SpyProvider(ModelProvider):
            def understand_task(self, instruction: str, world_context: Optional[Dict[str, Any]] = None):
                captured_calls.append({"instruction": instruction, "world_context": world_context})
                return TaskUnderstanding(
                    task_id="spy_task",
                    objective=instruction,
                    confidence=0.95,
                    reasoning_summary="Spy rationale",
                )

        spy = SpyProvider()
        service = TaskUnderstandingService(spy)

        context = {"workspace_id": "ws_123", "slip_risk": 0.2}
        result = service.understand("Pick up widget", world_context=context)

        self.assertEqual(len(captured_calls), 1)
        self.assertEqual(captured_calls[0]["instruction"], "Pick up widget")
        self.assertEqual(captured_calls[0]["world_context"], context)
        self.assertEqual(result.task_id, "spy_task")

    def test_model_layer_cannot_directly_execute_robot_actions(self):
        """
        Verify that the model integration boundary is strictly computational.
        Ensures model classes have no execution or actuation methods and cannot mutate WorldState.
        """
        prohibited_methods = [
            "execute", "execute_action", "actuate", "step", "move_motor",
            "set_position", "command", "apply_force"
        ]

        model_classes = [
            TaskUnderstanding,
            ModelProvider,
            MockModelProvider,
            TaskUnderstandingService,
        ]

        for cls in model_classes:
            for method_name in prohibited_methods:
                self.assertFalse(
                    hasattr(cls, method_name),
                    f"Class '{cls.__name__}' violates safety boundary by exposing '{method_name}'"
                )

        # Verify that generating a TaskUnderstanding cannot mutate a WorldState
        world_state = WorldState(
            version="ws_frozen_test",
            robot=RobotState(
                position=Point3D(x=0.0, y=0.0, z=0.2),
                gripper_open=True,
            ),
            target=TargetZone(id="z1", position=Point3D(x=0.4, y=-0.2, z=0.1)),
            environment=EnvironmentState(),
            last_action_outcome=LastActionOutcome(),
        )

        service = TaskUnderstandingService(MockModelProvider())
        understanding = service.understand("Move the red object to the blue target.")

        # Ensure WorldState remains unmodified
        self.assertEqual(world_state.version, "ws_frozen_test")
        self.assertEqual(world_state.robot.position, Point3D(x=0.0, y=0.0, z=0.2))

        # TaskUnderstanding is frozen/immutable
        with self.assertRaises(ValidationError):
            understanding.confidence = 0.5  # Attempt to mutate frozen model

    def test_empty_instruction_rejected(self):
        """Verify that blank or whitespace-only instructions are rejected."""
        service = TaskUnderstandingService(MockModelProvider())
        with self.assertRaises(ValueError):
            service.understand("")
        with self.assertRaises(ValueError):
            service.understand("   ")


if __name__ == "__main__":
    unittest.main()
