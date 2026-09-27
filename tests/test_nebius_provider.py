"""
SĀRTHI Phase 5A — Nebius Token Factory Nemotron Provider Unit & Integration Tests.

IMPORTANT:
- ZERO live Nebius API calls.
- Does NOT require a real NEBIUS_API_KEY.
- All OpenAI client / network calls are strictly mocked.
- Verifies cognitive-only boundary (provider never executes physical actions).
"""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

import openai

from backend.app.decision_engine.models import ActionType
from backend.app.model.config import NebiusConfig, DEFAULT_NEBIUS_BASE_URL, DEFAULT_NEBIUS_TIMEOUT
from backend.app.model.models import TaskUnderstanding
from backend.app.model.nebius_provider import (
    NebiusAPIError,
    NebiusConfigurationError,
    NebiusNemotronProvider,
    NebiusNetworkError,
    NebiusParsingError,
    NebiusProviderError,
    NebiusTimeoutError,
    NebiusValidationError,
    _extract_json,
    _sanitize_error_message,
)
from backend.app.model.provider import ModelProvider
from backend.app.model.task_understanding import TaskUnderstandingService
from backend.app.orchestration.loop import TaskRunStatus
from backend.app.orchestration.task_runner import SarthiTaskRunner
from simulation.adapters.base import LocalSimulationAdapter
from simulation.core.geometry import SimDimensions3D, SimPoint3D, WorkspaceBounds
from simulation.core.objects import SimulatedObject, SimulatedTargetZone
from simulation.core.robot import SimulatedRobot
from simulation.core.world import SimulationWorld


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _make_mock_completion(content: str) -> MagicMock:
    """Builds a mock OpenAI ChatCompletion response structure."""
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


def _make_standard_world():
    """Builds standard world with robot, red_object_01, and blue_target_zone."""
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


CANONICAL_VALID_PAYLOAD = {
    "task_id": "task_nebius_101",
    "objective": "Move the red object to the blue target.",
    "target_object": "red_object",
    "target_location": "blue_target",
    "required_actions": ["APPROACH", "GRASP", "MOVE", "RELEASE"],
    "constraints": ["maintain_clearance", "avoid_workspace_breach"],
    "success_conditions": ["red_object_at_blue_target", "gripper_released"],
    "confidence": 0.98,
    "reasoning_summary": "Identified pick-and-place operation from human instruction.",
}


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

class TestNebiusNemotronProvider(unittest.TestCase):
    """
    Complete unit test suite for NebiusNemotronProvider.
    Guarantees no live network calls and strict cognitive decoupling.
    """

    def setUp(self):
        # Ensure clean environment without real credentials
        self.env_patcher = patch.dict(os.environ, {}, clear=True)
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    # 1. Provider configuration
    def test_01_provider_configuration(self):
        """Verify provider and NebiusConfig configure properly from params and env."""
        # Via explicit parameters
        mock_client = MagicMock()
        provider = NebiusNemotronProvider(
            api_key="mock-key-123",
            base_url="https://api.tokenfactory.nebius.com/v1",
            model="nvidia/nemotron-4-340b-instruct",
            timeout=45.0,
            client=mock_client,
        )
        self.assertEqual(provider.model_name, "nvidia/nemotron-4-340b-instruct")
        self.assertEqual(provider.base_url, "https://api.tokenfactory.nebius.com/v1")
        self.assertEqual(provider.config.timeout, 45.0)

        # Via environment variables
        with patch.dict(os.environ, {
            "NEBIUS_API_KEY": "env-key-abc",
            "NEBIUS_BASE_URL": "https://custom.tokenfactory.nebius.com/v1",
            "NEBIUS_MODEL": "nvidia/nemotron-4-mini-instruct",
            "NEBIUS_TIMEOUT": "20.5",
        }):
            provider_env = NebiusNemotronProvider(client=mock_client)
            self.assertEqual(provider_env.model_name, "nvidia/nemotron-4-mini-instruct")
            self.assertEqual(provider_env.base_url, "https://custom.tokenfactory.nebius.com/v1")
            self.assertEqual(provider_env.config.timeout, 20.5)

    # 2. Missing API key
    def test_02_missing_api_key(self):
        """Verify clear configuration error if NEBIUS_API_KEY is missing or empty."""
        mock_client = MagicMock()
        # Empty env, no api_key passed
        with self.assertRaises(NebiusConfigurationError) as ctx:
            NebiusNemotronProvider(
                model="nvidia/nemotron-4-340b-instruct",
                client=mock_client,
            )
        self.assertIn("NEBIUS_API_KEY is missing", str(ctx.exception))

        # Blank/whitespace api_key passed
        with self.assertRaises(NebiusConfigurationError) as ctx2:
            NebiusNemotronProvider(
                api_key="   ",
                model="nvidia/nemotron-4-340b-instruct",
                client=mock_client,
            )
        self.assertIn("NEBIUS_API_KEY is missing", str(ctx2.exception))

    # 3. Missing model ID
    def test_03_missing_model_id(self):
        """Verify clear configuration error if NEBIUS_MODEL is missing or empty."""
        mock_client = MagicMock()
        with self.assertRaises(NebiusConfigurationError) as ctx:
            NebiusNemotronProvider(
                api_key="mock-valid-key",
                client=mock_client,
            )
        self.assertIn("NEBIUS_MODEL is missing", str(ctx.exception))

        # Blank/whitespace model passed
        with self.assertRaises(NebiusConfigurationError) as ctx2:
            NebiusNemotronProvider(
                api_key="mock-valid-key",
                model="   ",
                client=mock_client,
            )
        self.assertIn("NEBIUS_MODEL is missing", str(ctx2.exception))

    # 4. Correct Nebius base URL
    def test_04_correct_nebius_base_url(self):
        """Verify Nebius Token Factory base URL defaults correctly and can be customized."""
        self.assertEqual(DEFAULT_NEBIUS_BASE_URL, "https://api.tokenfactory.nebius.com/v1")

        with patch("openai.OpenAI") as mock_openai_cls:
            # Default base URL
            NebiusNemotronProvider(
                api_key="test-key",
                model="nvidia/nemotron-4-340b-instruct",
            )
            mock_openai_cls.assert_called_with(
                api_key="test-key",
                base_url="https://api.tokenfactory.nebius.com/v1",
                timeout=DEFAULT_NEBIUS_TIMEOUT,
            )

        with patch("openai.OpenAI") as mock_openai_cls2:
            # Custom base URL
            NebiusNemotronProvider(
                api_key="test-key",
                base_url="https://alt.api.nebius.com/v1",
                model="nvidia/nemotron-4-340b-instruct",
            )
            mock_openai_cls2.assert_called_with(
                api_key="test-key",
                base_url="https://alt.api.nebius.com/v1",
                timeout=DEFAULT_NEBIUS_TIMEOUT,
            )

    # 5. Correct configured model ID
    def test_05_correct_configured_model_id(self):
        """Verify that provider dispatches requests with the exact configured model ID."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_completion(
            json.dumps(CANONICAL_VALID_PAYLOAD)
        )
        provider = NebiusNemotronProvider(
            api_key="mock-key",
            model="nvidia/nemotron-4-340b-instruct",
            client=mock_client,
        )

        provider.understand_task("Move the red object to the blue target.")

        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "nvidia/nemotron-4-340b-instruct")
        self.assertEqual(call_kwargs["temperature"], 0.0)

    # 6. Valid structured response
    def test_06_valid_structured_response(self):
        """Verify provider parses pure JSON response into validated TaskUnderstanding."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_completion(
            json.dumps(CANONICAL_VALID_PAYLOAD)
        )
        provider = NebiusNemotronProvider(
            api_key="mock-key",
            model="nvidia/nemotron-4-340b-instruct",
            client=mock_client,
        )

        understanding = provider.understand_task("Move the red object to the blue target.")

        self.assertIsInstance(understanding, TaskUnderstanding)
        self.assertEqual(understanding.task_id, "task_nebius_101")
        self.assertEqual(understanding.objective, "Move the red object to the blue target.")
        self.assertEqual(understanding.target_object, "red_object")
        self.assertEqual(understanding.target_location, "blue_target")
        self.assertEqual(understanding.required_actions, ["APPROACH", "GRASP", "MOVE", "RELEASE"])
        self.assertEqual(understanding.confidence, 0.98)
        self.assertEqual(understanding.constraints, ["maintain_clearance", "avoid_workspace_breach"])
        self.assertEqual(understanding.success_conditions, ["red_object_at_blue_target", "gripper_released"])

    # 7. Markdown-fenced JSON response
    def test_07_markdown_fenced_json_response(self):
        """Verify provider un-fences ```json ... ``` and ``` ... ``` markdown safely."""
        mock_client = MagicMock()
        provider = NebiusNemotronProvider(
            api_key="mock-key",
            model="nvidia/nemotron-4-340b-instruct",
            client=mock_client,
        )

        # Test case A: ```json fence
        fenced_json_1 = f"```json\n{json.dumps(CANONICAL_VALID_PAYLOAD)}\n```"
        mock_client.chat.completions.create.return_value = _make_mock_completion(fenced_json_1)
        res1 = provider.understand_task("Move the red object to the blue target.")
        self.assertIsInstance(res1, TaskUnderstanding)
        self.assertEqual(res1.target_object, "red_object")

        # Test case B: ``` bare fence
        fenced_json_2 = f"```\n{json.dumps(CANONICAL_VALID_PAYLOAD)}\n```"
        mock_client.chat.completions.create.return_value = _make_mock_completion(fenced_json_2)
        res2 = provider.understand_task("Move the red object to the blue target.")
        self.assertIsInstance(res2, TaskUnderstanding)
        self.assertEqual(res2.target_location, "blue_target")

    # 8. Invalid JSON
    def test_08_invalid_json_rejected(self):
        """Verify malformed JSON or natural language commentary raises NebiusParsingError."""
        mock_client = MagicMock()
        provider = NebiusNemotronProvider(
            api_key="mock-key",
            model="nvidia/nemotron-4-340b-instruct",
            client=mock_client,
        )

        # Test case A: Plain text chatter
        mock_client.chat.completions.create.return_value = _make_mock_completion(
            "Sure! I can help you move the red object to the blue target. Here is what I think..."
        )
        with self.assertRaises(NebiusParsingError):
            provider.understand_task("Move the red object to the blue target.")

        # Test case B: Broken JSON syntax
        mock_client.chat.completions.create.return_value = _make_mock_completion(
            '{"task_id": "bad", "objective": "broken'
        )
        with self.assertRaises(NebiusParsingError):
            provider.understand_task("Move the red object to the blue target.")

        # Test case C: JSON array instead of object
        mock_client.chat.completions.create.return_value = _make_mock_completion(
            '["APPROACH", "GRASP", "MOVE", "RELEASE"]'
        )
        with self.assertRaises(NebiusParsingError):
            provider.understand_task("Move the red object to the blue target.")

    # 9. Schema validation failure
    def test_09_schema_validation_failure(self):
        """Verify JSON failing TaskUnderstanding schema raises NebiusValidationError."""
        mock_client = MagicMock()
        provider = NebiusNemotronProvider(
            api_key="mock-key",
            model="nvidia/nemotron-4-340b-instruct",
            client=mock_client,
        )

        # Missing required task_id and objective
        invalid_payload_1 = {
            "target_object": "red_object",
            "target_location": "blue_target",
        }
        mock_client.chat.completions.create.return_value = _make_mock_completion(
            json.dumps(invalid_payload_1)
        )
        with self.assertRaises(NebiusValidationError):
            provider.understand_task("Move the red object to the blue target.")

        # Out-of-bounds confidence (> 1.0)
        invalid_payload_2 = dict(CANONICAL_VALID_PAYLOAD)
        invalid_payload_2["confidence"] = 1.85
        mock_client.chat.completions.create.return_value = _make_mock_completion(
            json.dumps(invalid_payload_2)
        )
        with self.assertRaises(NebiusValidationError):
            provider.understand_task("Move the red object to the blue target.")

    # 10. Network/API failure
    def test_10_network_api_failures(self):
        """Verify network transport, timeout, and API status failures are caught and mapped."""
        mock_client = MagicMock()
        provider = NebiusNemotronProvider(
            api_key="mock-key",
            model="nvidia/nemotron-4-340b-instruct",
            client=mock_client,
        )

        # Connection error -> NebiusNetworkError
        mock_client.chat.completions.create.side_effect = openai.APIConnectionError(
            request=MagicMock()
        )
        with self.assertRaises(NebiusNetworkError):
            provider.understand_task("Move the red object to the blue target.")

        # Timeout error -> NebiusTimeoutError
        mock_client.chat.completions.create.side_effect = openai.APITimeoutError(
            request=MagicMock()
        )
        with self.assertRaises(NebiusTimeoutError):
            provider.understand_task("Move the red object to the blue target.")

        # Status error -> NebiusAPIError
        mock_resp = MagicMock()
        mock_resp.status_code = 502
        mock_client.chat.completions.create.side_effect = openai.APIStatusError(
            message="Bad Gateway",
            response=mock_resp,
            body={"error": "server error"},
        )
        with self.assertRaises(NebiusAPIError):
            provider.understand_task("Move the red object to the blue target.")

    # 11. API key is never included in error output
    def test_11_api_key_never_included_in_error_output(self):
        """Verify that secret API key is scrubbed from exception messages, logs, and repr."""
        secret_key = "sk-nebius-secret-super-sensitive-token-9876543210"
        mock_client = MagicMock()

        # Simulate exception containing the secret key in message
        mock_client.chat.completions.create.side_effect = openai.APIConnectionError(
            message=f"Failed request with Bearer {secret_key}",
            request=MagicMock(),
        )

        provider = NebiusNemotronProvider(
            api_key=secret_key,
            model="nvidia/nemotron-4-340b-instruct",
            client=mock_client,
        )

        # Check that exception message does NOT contain secret_key
        try:
            provider.understand_task("Move the red object to the blue target.")
            self.fail("Expected NebiusNetworkError was not raised")
        except NebiusNetworkError as exc:
            exc_str = str(exc)
            self.assertNotIn(secret_key, exc_str)
            self.assertIn("[REDACTED", exc_str)

        # Check repr of provider
        self.assertNotIn(secret_key, repr(provider))
        self.assertNotIn(secret_key, str(provider))

        # Check repr of config
        self.assertNotIn(secret_key, repr(provider.config))
        self.assertNotIn(secret_key, str(provider.config))

    # 12. Provider never executes physical actions
    def test_12_provider_never_executes_physical_actions(self):
        """Verify NebiusNemotronProvider has strictly no physical actuation capabilities."""
        mock_client = MagicMock()
        provider = NebiusNemotronProvider(
            api_key="mock-key",
            model="nvidia/nemotron-4-340b-instruct",
            client=mock_client,
        )

        self.assertIsInstance(provider, ModelProvider)

        # Disallowed execution methods
        forbidden_methods = [
            "execute_action",
            "step",
            "actuate",
            "move_arm",
            "set_gripper",
            "modify_world",
            "apply_action",
            "run_loop",
        ]
        for method in forbidden_methods:
            self.assertFalse(
                hasattr(provider, method),
                f"NebiusNemotronProvider violates boundary: has '{method}' method!",
            )

        # Ensure provider has no reference to SimulationAdapter or WorldState
        self.assertFalse(hasattr(provider, "adapter"))
        self.assertFalse(hasattr(provider, "world_state"))
        self.assertFalse(hasattr(provider, "engine"))

    # Empty instruction check
    def test_empty_instruction_rejected(self):
        """Verify empty or whitespace instruction is rejected before network invocation."""
        mock_client = MagicMock()
        provider = NebiusNemotronProvider(
            api_key="mock-key",
            model="nvidia/nemotron-4-340b-instruct",
            client=mock_client,
        )

        with self.assertRaises(ValueError):
            provider.understand_task("")

        with self.assertRaises(ValueError):
            provider.understand_task("   \n\t  ")

        mock_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Integration Test (Section 12)
# ---------------------------------------------------------------------------

class TestNebiusNemotronIntegration(unittest.TestCase):
    """
    Mocked end-to-end integration test:
    instruction
        ↓
    NebiusNemotronProvider (mocked response)
        ↓
    TaskUnderstandingService
        ↓
    TaskRunner.run_instruction()
        ↓
    Decision Engine
        ↓
    Simulation
    """

    def test_mocked_end_to_end_instruction_run(self):
        """
        Runs full instruction pipeline with mocked Nebius response.
        Expected physical action sequence: APPROACH -> GRASP -> MOVE -> RELEASE.
        """
        world, adapter = _make_standard_world()

        # Mock Nebius client returning structured task understanding
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_completion(
            json.dumps(CANONICAL_VALID_PAYLOAD)
        )

        provider = NebiusNemotronProvider(
            api_key="mock-token-xyz",
            model="nvidia/nemotron-4-340b-instruct",
            client=mock_client,
        )

        # Coordinate through SarthiTaskRunner
        runner = SarthiTaskRunner(adapter=adapter)
        instruction = "Move the red object to the blue target."

        result = runner.run_instruction(
            instruction=instruction,
            model_provider=provider,
        )

        # Assert full task completion
        self.assertTrue(result.completed, f"Task did not complete: {result.failure_reason}")
        self.assertEqual(result.run_status, TaskRunStatus.COMPLETED.value)
        self.assertEqual(
            result.executed_actions,
            [
                ActionType.APPROACH.value,
                ActionType.GRASP.value,
                ActionType.MOVE.value,
                ActionType.RELEASE.value,
            ],
        )
        self.assertEqual(result.failed_actions, [])
        self.assertEqual(result.recovery_count, 0)
        self.assertTrue(runner.is_task_completed(adapter.get_world_state()))

        # Verify Decision Engine governed physical actions, not provider
        self.assertEqual(len(result.executed_actions), 4)
        mock_client.chat.completions.create.assert_called_once()

    def test_default_model_provider_is_nebius_provider(self):
        """
        Verify SarthiTaskRunner defaults to NebiusNemotronProvider when no model_provider is provided.
        Ensures NO live API call occurs by patching the provider class.
        """
        world, adapter = _make_standard_world()
        runner = SarthiTaskRunner(adapter=adapter)

        class _MockNebiusProvider(ModelProvider):
            def understand_task(self, instruction, world_context=None):
                return TaskUnderstanding.model_validate(CANONICAL_VALID_PAYLOAD)

        with patch("backend.app.orchestration.task_runner.NebiusNemotronProvider") as mock_prov_cls:
            mock_prov_cls.return_value = _MockNebiusProvider()

            result = runner.run_instruction("Move the red object to the blue target.")
            mock_prov_cls.assert_called_once()
            self.assertTrue(result.completed)


if __name__ == "__main__":
    unittest.main()
