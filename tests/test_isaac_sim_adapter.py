"""
SĀRTHI Phase 6A — NVIDIA Isaac Sim Adapter Unit & Boundary Tests.

IMPORTANT:
- ZERO dependencies on NVIDIA Isaac Sim packages.
- Tests run completely offline and in lightweight CPU / CI environments.
- Verifies adapter contract, lazy dependency guard, deterministic translations,
  and strict architectural boundary separating simulation from decision engine.
"""

import sys
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock

from backend.app.decision_engine.models import (
    ActionType,
    CandidateAction,
    ObjectState,
    Point3D,
    WorldObject,
    WorldState,
)
from simulation.adapters.base import LocalSimulationAdapter, SimulationAdapter
from simulation.adapters.isaac_sim import (
    IsaacSimAction,
    IsaacSimAdapter,
    IsaacSimDisturbance,
    IsaacSimUnavailableError,
    is_isaac_sim_available,
)
from simulation.core.events import (
    ActionExecutionResult,
    DisturbanceEvent,
    DisturbanceType,
)
from simulation.core.geometry import SimDimensions3D, SimPoint3D


class TestIsaacSimAdapter(unittest.TestCase):
    """
    Unit test suite verifying Phase 6A Isaac Sim Adapter Boundary.
    """

    # 1. Adapter can be imported without Isaac Sim installed
    def test_01_adapter_importable_without_isaac_sim(self):
        """Verify IsaacSimAdapter is importable without requiring Isaac Sim packages."""
        # Ensure 'isaacsim' is not present in sys.modules
        self.assertNotIn("isaacsim", sys.modules)

        # Confirm inheritance and interface compliance
        self.assertTrue(issubclass(IsaacSimAdapter, SimulationAdapter))
        adapter = IsaacSimAdapter()
        self.assertIsInstance(adapter, SimulationAdapter)

    # 2. Missing Isaac Sim produces IsaacSimUnavailableError only when requested
    def test_02_missing_isaac_sim_produces_unavailable_error_on_demand(self):
        """Verify IsaacSimUnavailableError is raised only when live functionality is invoked."""
        adapter = IsaacSimAdapter()

        # Instantiation does NOT raise
        self.assertFalse(adapter.is_connected)

        # Invoking live methods without a backend must raise IsaacSimUnavailableError
        with self.assertRaises(IsaacSimUnavailableError) as ctx_ws:
            adapter.get_world_state()
        self.assertIn("Isaac Sim is not installed in this environment", str(ctx_ws.exception))

        candidate = CandidateAction(
            action_id="act_test_01",
            action_type=ActionType.APPROACH,
            target_object_id="red_object_01",
            target_position=Point3D(x=0.25, y=0.15, z=0.20),
        )
        with self.assertRaises(IsaacSimUnavailableError) as ctx_exec:
            adapter.execute_action(candidate)
        self.assertIn("Isaac Sim runtime environment", str(ctx_exec.exception))

        disturbance = DisturbanceEvent.create_path_blocked(
            event_id="dist_01",
            obstacle_id="obs_01",
            position=SimPoint3D(x=0.3, y=0.0, z=0.2),
        )
        with self.assertRaises(IsaacSimUnavailableError):
            adapter.inject_disturbance(disturbance)

        with self.assertRaises(IsaacSimUnavailableError):
            adapter.verify_action_result(candidate)

    # 3. SĀRTHI action vocabulary is preserved
    def test_03_sarthi_action_vocabulary_preserved(self):
        """Verify all 6 canonical SĀRTHI actions are supported and mapped."""
        all_actions = [
            ActionType.APPROACH,
            ActionType.REPOSITION,
            ActionType.GRASP,
            ActionType.MOVE,
            ActionType.RELEASE,
            ActionType.STOP,
        ]

        for act_type in all_actions:
            candidate = CandidateAction(
                action_id=f"act_{act_type.value.lower()}",
                action_type=act_type,
                target_position=Point3D(x=0.1, y=0.2, z=0.3),
                expected_force_n=10.0,
                speed_scale=0.8,
            )
            translated = IsaacSimAdapter.translate_action(candidate)

            self.assertIsInstance(translated, IsaacSimAction)
            self.assertEqual(translated.action_type, act_type)
            self.assertEqual(translated.action_id, f"act_{act_type.value.lower()}")
            self.assertEqual(translated.speed_scale, 0.8)
            self.assertEqual(translated.expected_force_n, 10.0)

    # 4. Action translation mapping is deterministic
    def test_04_action_translation_mapping_deterministic(self):
        """Verify translation mapping produces identical, correct controller primitives."""
        candidate = CandidateAction(
            action_id="act_approach_det",
            action_type=ActionType.APPROACH,
            target_object_id="red_cylinder",
            target_position=Point3D(x=0.25, y=0.15, z=0.20),
            speed_scale=0.9,
            expected_force_n=5.0,
        )

        res1 = IsaacSimAdapter.translate_action(candidate)
        res2 = IsaacSimAdapter.translate_action(candidate)

        self.assertEqual(res1.model_dump(), res2.model_dump())
        self.assertEqual(res1.primitive_name, "cartesian_approach")
        self.assertEqual(res1.gripper_target, "OPEN")
        self.assertEqual(res1.target_position, {"x": 0.25, "y": 0.15, "z": 0.20})

        # Test specific primitive mappings
        expected_mappings = {
            ActionType.APPROACH: ("cartesian_approach", "OPEN"),
            ActionType.REPOSITION: ("cartesian_reposition", "HOLD"),
            ActionType.GRASP: ("gripper_close", "CLOSE"),
            ActionType.MOVE: ("cartesian_move", "HOLD"),
            ActionType.RELEASE: ("gripper_open", "OPEN"),
            ActionType.STOP: ("emergency_hold", "HOLD"),
        }

        for act_type, (expected_prim, expected_gripper) in expected_mappings.items():
            act = CandidateAction(
                action_id="test_act",
                action_type=act_type,
            )
            mapped = IsaacSimAdapter.translate_action(act)
            self.assertEqual(mapped.primitive_name, expected_prim)
            self.assertEqual(mapped.gripper_target, expected_gripper)

    # 5. PATH_BLOCKED disturbance representation is supported
    def test_05_path_blocked_disturbance_representation(self):
        """Verify PATH_BLOCKED disturbance translates to USD collision entity specification."""
        disturbance = DisturbanceEvent.create_path_blocked(
            event_id="dist_block_1",
            obstacle_id="blocker_cube",
            position=SimPoint3D(x=0.32, y=-0.05, z=0.22),
            dimensions=SimDimensions3D(length_x=0.10, width_y=0.10, height_z=0.25),
        )

        translated = IsaacSimAdapter.translate_disturbance(disturbance)

        self.assertIsInstance(translated, IsaacSimDisturbance)
        self.assertEqual(translated.event_id, "dist_block_1")
        self.assertEqual(translated.disturbance_type, DisturbanceType.PATH_BLOCKED.value)
        self.assertEqual(translated.usd_prim_path, "/World/Obstacles/blocker_cube")
        self.assertEqual(translated.usd_prim_type, "Cube")
        self.assertEqual(translated.position, {"x": 0.32, "y": -0.05, "z": 0.22})
        self.assertEqual(translated.dimensions, {"length_x": 0.10, "width_y": 0.10, "height_z": 0.25})
        self.assertTrue(translated.is_active)
        self.assertTrue(translated.physics_enabled)

    # 6. Isaac adapter does not contain Decision Engine logic
    def test_06_isaac_adapter_does_not_contain_decision_engine_logic(self):
        """Verify IsaacSimAdapter does not implement decision engine scoring or evaluation."""
        adapter = IsaacSimAdapter()

        forbidden_cognitive_methods = [
            "decide",
            "evaluate_candidates",
            "calculate_responsibility",
            "calculate_relevance",
            "evaluate_consequences",
            "check_constraints",
            "generate_candidates",
            "evaluate_candidate",
        ]

        for method in forbidden_cognitive_methods:
            self.assertFalse(
                hasattr(adapter, method),
                f"IsaacSimAdapter violates architectural boundary: contains cognitive method '{method}'",
            )

    # 7. Isaac adapter does not directly select actions
    def test_07_isaac_adapter_does_not_select_actions(self):
        """Verify IsaacSimAdapter strictly accepts actions from outside and does not choose actions."""
        adapter = IsaacSimAdapter()

        # The adapter must not have an action selection method
        self.assertFalse(hasattr(adapter, "select_action"))
        self.assertFalse(hasattr(adapter, "choose_action"))
        self.assertFalse(hasattr(adapter, "propose_action"))

        # execute_action requires an action passed into it
        import inspect
        sig = inspect.signature(adapter.execute_action)
        self.assertIn("action", sig.parameters)

    # 8. WorldState translation from raw simulator dict
    def test_08_world_state_translation_reuses_canonical_model(self):
        """Verify IsaacSimAdapter.translate_world_state builds canonical SĀRTHI WorldState."""
        raw_state: Dict[str, Any] = {
            "version": 4,
            "timestamp_ns": 40000000,
            "robot": {
                "position": {"x": 0.25, "y": 0.15, "z": 0.20},
                "gripper_open": False,
                "holding_object_id": "red_cylinder_01",
                "payload_mass_kg": 0.5,
                "is_moving": False,
                "max_payload_kg": 3.0,
                "max_reach_m": 0.85,
            },
            "objects": [
                {
                    "id": "red_cylinder_01",
                    "name": "Red Cylinder",
                    "position": {"x": 0.25, "y": 0.15, "z": 0.20},
                    "bounding_radius_m": 0.05,
                    "mass_kg": 0.5,
                    "state": "GRASPED",
                    "is_target": True,
                    "is_obstacle": False,
                },
                {
                    "id": "blocker_obs_01",
                    "name": "Dynamic Obstacle",
                    "position": {"x": 0.35, "y": 0.0, "z": 0.20},
                    "bounding_radius_m": 0.08,
                    "mass_kg": 5.0,
                    "state": "FREE",
                    "is_target": False,
                    "is_obstacle": True,
                },
            ],
            "target": {
                "id": "blue_target_zone",
                "position": {"x": 0.40, "y": -0.20, "z": 0.20},
                "tolerance_radius_m": 0.06,
            },
            "environment": {
                "min_x": -0.8,
                "max_x": 0.8,
                "min_y": -0.8,
                "max_y": 0.8,
                "min_z": 0.0,
                "max_z": 1.2,
                "dynamic_obstacles_detected": True,
                "slip_risk_level": 0.0,
                "friction_coefficient": 0.6,
            },
            "task_objective": "PICK_AND_PLACE",
            "last_action_outcome": {
                "action_type": "GRASP",
                "status": "SUCCESS",
                "error_message": None,
                "contact_force_delta": 4.8,
            },
        }

        world_state = IsaacSimAdapter.translate_world_state(raw_state)

        self.assertIsInstance(world_state, WorldState)
        self.assertEqual(world_state.version, 4)
        self.assertEqual(world_state.robot.holding_object_id, "red_cylinder_01")
        self.assertFalse(world_state.robot.gripper_open)
        self.assertEqual(len(world_state.objects), 2)
        self.assertTrue(world_state.objects[0].is_target)
        self.assertTrue(world_state.objects[1].is_obstacle)
        self.assertTrue(world_state.environment.dynamic_obstacles_detected)
        self.assertEqual(world_state.last_action_outcome.action_type, ActionType.GRASP)

    # 9. Adapter with mocked simulation backend operates end-to-end
    def test_09_adapter_with_mock_backend(self):
        """Verify IsaacSimAdapter functions seamlessly when connected to a simulation backend."""
        mock_backend = MagicMock()
        mock_backend.get_world_state.return_value = {
            "version": 1,
            "timestamp_ns": 0,
            "robot": {
                "position": {"x": 0.0, "y": 0.0, "z": 0.20},
                "gripper_open": True,
                "holding_object_id": None,
            },
            "objects": [],
            "target": {"id": "zone_1", "position": {"x": 0.4, "y": -0.2, "z": 0.2}},
            "environment": {},
            "task_objective": "PICK_AND_PLACE",
            "last_action_outcome": {},
        }
        mock_backend.execute_action.return_value = ActionExecutionResult(
            success=True,
            action_type="APPROACH",
            action_id="act_app_1",
            simulation_time=0.5,
            previous_world_state_version=1,
            new_world_state_version=2,
            failure_reason=None,
            details={},
        )
        mock_backend.inject_disturbance.return_value = True
        mock_backend.verify_action_result.return_value = True

        adapter = IsaacSimAdapter(sim_backend=mock_backend)
        self.assertTrue(adapter.is_connected)

        # 1. State retrieval
        ws = adapter.get_world_state()
        self.assertIsInstance(ws, WorldState)
        mock_backend.get_world_state.assert_called_once()

        # 2. Action execution
        act = CandidateAction(
            action_id="act_app_1",
            action_type=ActionType.APPROACH,
            target_position=Point3D(x=0.25, y=0.15, z=0.20),
        )
        exec_res = adapter.execute_action(act)
        self.assertTrue(exec_res.success)
        mock_backend.execute_action.assert_called_once()
        # Verify backend received translated IsaacSimAction
        dispatched_arg = mock_backend.execute_action.call_args[0][0]
        self.assertIsInstance(dispatched_arg, IsaacSimAction)
        self.assertEqual(dispatched_arg.primitive_name, "cartesian_approach")

        # 3. Disturbance injection
        dist = DisturbanceEvent.create_path_blocked(
            event_id="dist_1",
            obstacle_id="block_1",
            position=SimPoint3D(x=0.3, y=0.0, z=0.2),
        )
        inj_res = adapter.inject_disturbance(dist)
        self.assertTrue(inj_res)
        mock_backend.inject_disturbance.assert_called_once()

        # 4. Action verification
        ver_res = adapter.verify_action_result(act)
        self.assertTrue(ver_res)

    # 10. LocalSimulationAdapter existing behavior unchanged
    def test_10_local_simulation_adapter_unchanged(self):
        """Verify LocalSimulationAdapter behavior and interface remain 100% intact."""
        local_adapter = LocalSimulationAdapter()
        ws = local_adapter.get_world_state()
        self.assertIsInstance(ws, WorldState)
        self.assertTrue(hasattr(local_adapter, "world"))


if __name__ == "__main__":
    unittest.main()
