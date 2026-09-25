"""
SĀRTHI Orchestration — Unit Test Suite.
Validates end-to-end task coordination across Decision Engine and Simulation Adapter interfaces,
testing nominal execution, state version propagation, verification gating,
dynamic PATH_BLOCKED disturbance re-planning, recovery, safety stop, and repeatability.
"""

import unittest
from simulation.core.geometry import SimDimensions3D, SimPoint3D, WorkspaceBounds
from simulation.core.robot import SimulatedRobot
from simulation.core.objects import SimulatedObject, SimulatedTargetZone
from simulation.core.events import DisturbanceEvent
from simulation.core.world import SimulationWorld
from simulation.adapters.base import LocalSimulationAdapter

from backend.app.decision_engine.models import ActionType, TaskObjective
from backend.app.orchestration.task_runner import SarthiTaskRunner
from backend.app.orchestration.loop import TaskEventType, TaskRunStatus


class TestSarthiOrchestration(unittest.TestCase):
    """Test suite covering Phase 3B closed-loop orchestration integration."""

    def _create_standard_environment(self):
        """Helper to assemble a clean test world for object transfer."""
        bounds = WorkspaceBounds(min_x=-0.8, max_x=0.8, min_y=-0.8, max_y=0.8, min_z=0.0, max_z=1.0)
        robot = SimulatedRobot(
            robot_id="sarthi_franka_arm",
            position=SimPoint3D(x=0.0, y=0.0, z=0.20),
            max_reach=0.85,
            max_payload_kg=3.0,
            workspace_limits=bounds,
        )
        world = SimulationWorld(robot=robot, workspace_bounds=bounds)

        # Red object at initial position
        red_obj = SimulatedObject(
            id="red_object_01",
            name="Red Cylinder",
            position=SimPoint3D(x=0.25, y=0.15, z=0.20),
            dimensions=SimDimensions3D(length_x=0.05, width_y=0.05, height_z=0.06),
            mass=0.5,
            is_target=True,
        )

        # Blue target zone at destination
        blue_target = SimulatedTargetZone(
            id="blue_target_zone",
            name="Blue Destination Zone",
            position=SimPoint3D(x=0.40, y=-0.20, z=0.20),
            tolerance_radius=0.06,
        )

        world.add_object(red_obj)
        world.add_target_zone(blue_target)
        adapter = LocalSimulationAdapter(world)
        return world, adapter, red_obj, blue_target

    def test_complete_nominal_task(self):
        """A, B, D. Complete nominal task: APPROACH -> GRASP -> MOVE -> RELEASE with verification."""
        world, adapter, red_obj, blue_target = self._create_standard_environment()
        runner = SarthiTaskRunner(adapter=adapter, max_steps=10)

        result = runner.run_until_complete(task_id="test_nominal_transfer")

        # Must reach completed status
        self.assertEqual(result.status, TaskRunStatus.COMPLETED)
        self.assertIsNone(result.failure_reason)

        # Exactly 4 steps in canonical order
        expected_sequence = [
            ActionType.APPROACH,
            ActionType.GRASP,
            ActionType.MOVE,
            ActionType.RELEASE,
        ]
        self.assertEqual(len(result.ordered_step_results), 4)

        for idx, (step_res, expected_type) in enumerate(zip(result.ordered_step_results, expected_sequence), 1):
            self.assertEqual(step_res.step_number, idx)
            self.assertEqual(step_res.selected_action.action_type, expected_type)
            self.assertTrue(step_res.action_result.success)
            self.assertTrue(step_res.verification_result)

        # Final world state checks
        final_state = adapter.get_world_state()
        self.assertIsNone(final_state.robot.holding_object_id)
        self.assertTrue(final_state.robot.gripper_open)
        self.assertTrue(blue_target.contains(world.objects["red_object_01"].position))

        # Event stream verification
        event_types = [e.event_type for e in result.events]
        self.assertIn(TaskEventType.WORLD_OBSERVED, event_types)
        self.assertIn(TaskEventType.DECISION_MADE, event_types)
        self.assertIn(TaskEventType.ACTION_EXECUTED, event_types)
        self.assertIn(TaskEventType.ACTION_VERIFIED, event_types)
        self.assertIn(TaskEventType.TASK_COMPLETED, event_types)

    def test_world_state_version_propagation(self):
        """C. World-state version propagation: verifies continuous monotonically increasing version chain."""
        _, adapter, _, _ = self._create_standard_environment()
        runner = SarthiTaskRunner(adapter=adapter, max_steps=10)

        result = runner.run_until_complete()
        self.assertEqual(result.status, TaskRunStatus.COMPLETED)

        # Initial to final version must advance
        self.assertGreater(result.final_world_state_version, result.initial_world_state_version)

        # Each step before version must match previous step after version
        for i in range(len(result.ordered_step_results) - 1):
            curr_step = result.ordered_step_results[i]
            next_step = result.ordered_step_results[i + 1]
            self.assertGreater(curr_step.world_state_version_after, curr_step.world_state_version_before)
            self.assertEqual(curr_step.world_state_version_after, next_step.world_state_version_before)

    def test_path_blocked_disturbance_and_recovery(self):
        """E, F, G. Injects PATH_BLOCKED disturbance; verifies Decision Engine re-decision, detour, and completion."""
        world, adapter, red_obj, blue_target = self._create_standard_environment()
        runner = SarthiTaskRunner(adapter=adapter, max_steps=10)

        def inject_disturbance_after_grasp(step_number: int, world_state):
            # Injected after GRASP (step 2), before MOVE
            if step_number == 2:
                dist = DisturbanceEvent.create_path_blocked(
                    event_id="dist_block_direct_path",
                    obstacle_id="blocking_barrier_01",
                    position=SimPoint3D(x=0.325, y=-0.025, z=0.12),
                    dimensions=SimDimensions3D(length_x=0.06, width_y=0.06, height_z=0.08),
                )
                adapter.inject_disturbance(dist)

        result = runner.run_until_complete(
            task_id="test_disturbance_recovery",
            step_callback=inject_disturbance_after_grasp,
        )

        # The task must successfully complete despite the disturbance!
        self.assertEqual(result.status, TaskRunStatus.COMPLETED)

        # 5 steps executed: APPROACH -> GRASP -> REPOSITION -> MOVE -> RELEASE
        step_actions = [s.selected_action.action_type for s in result.ordered_step_results]
        self.assertEqual(
            step_actions,
            [
                ActionType.APPROACH,
                ActionType.GRASP,
                ActionType.REPOSITION,  # Adaptive re-decision!
                ActionType.MOVE,
                ActionType.RELEASE,
            ],
        )

        # Step 3 must have rejected direct MOVE with BlockedPath reason
        step_3 = result.ordered_step_results[2]
        self.assertEqual(step_3.selected_action.action_type, ActionType.REPOSITION)
        self.assertIsNotNone(step_3.decision)
        rejection_reasons = step_3.decision.rejection_reasons
        self.assertIn("act_move_to_target_zone", rejection_reasons)
        self.assertTrue(any("BlockedPath" in r for r in rejection_reasons["act_move_to_target_zone"]))

        # Verify task objective remained PICK_AND_PLACE throughout
        step_3_state = step_3.decision.world_state_version
        self.assertEqual(adapter.get_world_state().task_objective, TaskObjective.PICK_AND_PLACE)

    def test_maximum_step_safety_limit(self):
        """H. Maximum-step safety limit: terminates safely if task does not conclude within budget."""
        _, adapter, _, _ = self._create_standard_environment()
        # Restrict max_steps to 2 (insufficient to finish 4-step task)
        short_runner = SarthiTaskRunner(adapter=adapter, max_steps=2)

        result = short_runner.run_until_complete(task_id="test_step_exhaustion")

        self.assertEqual(result.status, TaskRunStatus.MAX_STEPS_REACHED)
        self.assertEqual(len(result.ordered_step_results), 2)
        self.assertIn("Exceeded maximum step count", result.failure_reason)

        event_types = [e.event_type for e in result.events]
        self.assertIn(TaskEventType.TASK_FAILED, event_types)

    def test_deterministic_repeated_task_runs(self):
        """I. Deterministic repeated task runs: two separate runs produce identical results."""
        def run_once():
            _, adapter, _, _ = self._create_standard_environment()
            runner = SarthiTaskRunner(adapter=adapter, max_steps=10)
            return runner.run_until_complete()

        run1 = run_once()
        run2 = run_once()

        self.assertEqual(run1.status, run2.status)
        self.assertEqual(run1.initial_world_state_version, run2.initial_world_state_version)
        self.assertEqual(run1.final_world_state_version, run2.final_world_state_version)
        self.assertEqual(len(run1.ordered_step_results), len(run2.ordered_step_results))

        for s1, s2 in zip(run1.ordered_step_results, run2.ordered_step_results):
            self.assertEqual(s1.step_number, s2.step_number)
            self.assertEqual(s1.selected_action.action_type, s2.selected_action.action_type)
            self.assertEqual(s1.selected_action.action_id, s2.selected_action.action_id)
            self.assertEqual(s1.world_state_version_before, s2.world_state_version_before)
            self.assertEqual(s1.world_state_version_after, s2.world_state_version_after)
            self.assertEqual(s1.action_result.success, s2.action_result.success)
            self.assertEqual(s1.verification_result, s2.verification_result)


if __name__ == "__main__":
    unittest.main()
