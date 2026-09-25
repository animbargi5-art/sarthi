"""
SĀRTHI Simulation Core — Unit Test Suite.
Validates deterministic physics and state tracking across entity creation,
kinematic actions, payload attachment/transport, state versioning,
PATH_BLOCKED disturbance injection, and repeatability.
"""

import unittest
from simulation.core.geometry import SimDimensions3D, SimPoint3D, WorkspaceBounds
from simulation.core.robot import SimulatedRobot
from simulation.core.objects import SimulatedObject, SimulatedObstacle, SimulatedTargetZone
from simulation.core.events import DisturbanceEvent, DisturbanceType
from simulation.core.world import SimulationWorld
from simulation.adapters.base import LocalSimulationAdapter

from backend.app.decision_engine.models import ActionType, CandidateAction


class TestSimulationCore(unittest.TestCase):
    """Test suite covering Phase 3A simulation-neutral world and environment core."""

    def setUp(self):
        self.bounds = WorkspaceBounds(min_x=-0.8, max_x=0.8, min_y=-0.8, max_y=0.8, min_z=0.0, max_z=1.0)
        self.robot = SimulatedRobot(
            robot_id="test_arm",
            position=SimPoint3D(x=0.0, y=0.0, z=0.2),
            max_reach=0.85,
            max_payload_kg=3.0,
            workspace_limits=self.bounds,
        )
        self.world = SimulationWorld(robot=self.robot, workspace_bounds=self.bounds)

    def test_initial_world_creation(self):
        """A. Initial world creation: verify entity registration, clock, and state export."""
        obj = SimulatedObject(
            id="cube_01",
            name="Test Cube",
            position=SimPoint3D(x=0.3, y=0.2, z=0.1),
            dimensions=SimDimensions3D(length_x=0.05, width_y=0.05, height_z=0.05),
            mass=0.6,
            is_target=True,
        )
        zone = SimulatedTargetZone(
            id="target_zone_A",
            name="Zone A",
            position=SimPoint3D(x=0.4, y=-0.2, z=0.1),
            tolerance_radius=0.05,
        )
        obs = SimulatedObstacle(
            id="barrier_01",
            position=SimPoint3D(x=0.2, y=0.0, z=0.2),
        )

        initial_v = self.world.world_state_version
        self.world.add_object(obj)
        self.world.add_target_zone(zone)
        self.world.add_obstacle(obs)

        # Entity additions must increment version
        self.assertEqual(self.world.world_state_version, initial_v + 3)
        self.assertIn("cube_01", self.world.objects)
        self.assertIn("target_zone_A", self.world.target_zones)
        self.assertIn("barrier_01", self.world.obstacles)

        # Export to decision engine WorldState
        de_state = self.world.to_decision_world_state()
        self.assertEqual(de_state.version, self.world.world_state_version)
        self.assertEqual(len(de_state.objects), 2)  # cube_01 + active barrier_01

    def test_robot_movement(self):
        """B. Robot movement: verify APPROACH / MOVE sets new coordinates and increments version."""
        start_v = self.world.world_state_version
        dest = SimPoint3D(x=0.25, y=0.15, z=0.30)

        action = CandidateAction(
            action_id="act_move_test",
            action_type=ActionType.MOVE,
            target_position={"x": dest.x, "y": dest.y, "z": dest.z},
        )
        result = self.world.execute_action(action)

        self.assertTrue(result.success)
        self.assertIsNone(result.failure_reason)
        self.assertEqual(result.previous_world_state_version, start_v)
        self.assertEqual(result.new_world_state_version, start_v + 1)
        self.assertEqual(self.world.robot.position, dest)

    def test_grasping_object(self):
        """C. Grasping an object: verify gripper closes, holding state updates, and object links to robot."""
        obj = SimulatedObject(
            id="pick_target",
            name="Target Part",
            position=SimPoint3D(x=0.20, y=0.10, z=0.20),
            mass=0.5,
        )
        self.world.add_object(obj)

        # Move robot directly adjacent to object (within 0.05m grasp proximity)
        self.world.robot.move_to(SimPoint3D(x=0.20, y=0.10, z=0.22))
        self.assertTrue(self.world.robot.gripper_open)

        grasp_action = CandidateAction(
            action_id="act_grasp_01",
            action_type=ActionType.GRASP,
            target_object_id="pick_target",
        )
        result = self.world.execute_action(grasp_action)

        self.assertTrue(result.success)
        self.assertFalse(self.world.robot.gripper_open)
        self.assertEqual(self.world.robot.carrying_object_id, "pick_target")
        self.assertEqual(obj.current_holder, self.world.robot.robot_id)

    def test_moving_while_carrying_object(self):
        """D. Moving while carrying an object: object position tracks robot end-effector coordinates."""
        obj = SimulatedObject(
            id="transport_obj",
            name="Transport Part",
            position=SimPoint3D(x=0.20, y=0.10, z=0.20),
            mass=0.7,
        )
        self.world.add_object(obj)
        self.world.robot.move_to(SimPoint3D(x=0.20, y=0.10, z=0.20))
        self.world.robot.attach_object("transport_obj")
        obj.current_holder = self.world.robot.robot_id

        # Move to new waypoint
        waypoint = SimPoint3D(x=0.40, y=-0.10, z=0.35)
        move_action = CandidateAction(
            action_id="act_transport_move",
            action_type=ActionType.MOVE,
            target_position={"x": waypoint.x, "y": waypoint.y, "z": waypoint.z},
        )
        result = self.world.execute_action(move_action)

        self.assertTrue(result.success)
        self.assertEqual(self.world.robot.position, waypoint)
        self.assertEqual(obj.position, waypoint)  # Object tracked end-effector!

    def test_releasing_object_at_target(self):
        """E. Releasing an object at a target: detaches payload, opens gripper, leaves object at destination."""
        zone = SimulatedTargetZone(
            id="drop_zone",
            position=SimPoint3D(x=0.35, y=-0.25, z=0.10),
            tolerance_radius=0.06,
        )
        obj = SimulatedObject(
            id="releasable_obj",
            name="Drop Part",
            position=SimPoint3D(x=0.35, y=-0.25, z=0.10),
            mass=0.4,
        )
        self.world.add_target_zone(zone)
        self.world.add_object(obj)

        self.world.robot.move_to(SimPoint3D(x=0.35, y=-0.25, z=0.10))
        self.world.robot.attach_object("releasable_obj")
        obj.current_holder = self.world.robot.robot_id

        release_action = CandidateAction(
            action_id="act_release_01",
            action_type=ActionType.RELEASE,
        )
        result = self.world.execute_action(release_action)

        self.assertTrue(result.success)
        self.assertTrue(self.world.robot.gripper_open)
        self.assertIsNone(self.world.robot.carrying_object_id)
        self.assertIsNone(obj.current_holder)
        self.assertTrue(zone.contains(obj.position))

    def test_world_state_version_increments(self):
        """F. World-state version increments: increments on state change, remains static on failure."""
        v0 = self.world.world_state_version

        # Successful move increments version
        res1 = self.world.execute_action(CandidateAction(
            action_id="act_v_1",
            action_type=ActionType.MOVE,
            target_position={"x": 0.1, "y": 0.1, "z": 0.2},
        ))
        self.assertTrue(res1.success)
        v1 = self.world.world_state_version
        self.assertEqual(v1, v0 + 1)

        # Disturbance injection increments version
        self.world.inject_disturbance(DisturbanceEvent.create_path_blocked(
            event_id="dist_01",
            obstacle_id="blocker_01",
            position=SimPoint3D(x=0.3, y=0.1, z=0.2),
        ))
        v2 = self.world.world_state_version
        self.assertEqual(v2, v1 + 1)

        # Failed action (e.g. out of bounds) MUST NOT increment version
        res_fail = self.world.execute_action(CandidateAction(
            action_id="act_fail_oob",
            action_type=ActionType.MOVE,
            target_position={"x": 5.0, "y": 0.0, "z": 0.2},
        ))
        self.assertFalse(res_fail.success)
        v3 = self.world.world_state_version
        self.assertEqual(v3, v2)  # Version preserved!

    def test_path_blocked_disturbance(self):
        """G. PATH_BLOCKED disturbance: obstacle inserted into path halts trajectory with BlockedPath."""
        start = SimPoint3D(x=0.0, y=0.0, z=0.2)
        goal = SimPoint3D(x=0.5, y=0.0, z=0.2)
        self.world.robot.move_to(start)

        # Inject PATH_BLOCKED disturbance midway at (0.25, 0.0, 0.2)
        event = DisturbanceEvent.create_path_blocked(
            event_id="evt_block_midway",
            obstacle_id="dynamic_wall",
            position=SimPoint3D(x=0.25, y=0.0, z=0.2),
            dimensions=SimDimensions3D(length_x=0.10, width_y=0.10, height_z=0.20),
        )
        injected = self.world.inject_disturbance(event)
        self.assertTrue(injected)
        self.assertIn("dynamic_wall", self.world.obstacles)

        # Attempt to move through blocked trajectory
        blocked_move = CandidateAction(
            action_id="act_through_obstacle",
            action_type=ActionType.MOVE,
            target_position={"x": goal.x, "y": goal.y, "z": goal.z},
        )
        result = self.world.execute_action(blocked_move)

        self.assertFalse(result.success)
        self.assertIn("BlockedPath", result.failure_reason)
        # Robot must remain safe at start position
        self.assertEqual(self.world.robot.position, start)

    def test_failed_action_structured_result(self):
        """H. Failed action returns a structured failure result without corrupting world state."""
        prev_v = self.world.world_state_version

        # Attempt to release when not holding anything
        premature_release = CandidateAction(
            action_id="act_premature_release",
            action_type=ActionType.RELEASE,
        )
        result = self.world.execute_action(premature_release)

        self.assertFalse(result.success)
        self.assertEqual(result.action_type, ActionType.RELEASE.value)
        self.assertEqual(result.action_id, "act_premature_release")
        self.assertEqual(result.previous_world_state_version, prev_v)
        self.assertEqual(result.new_world_state_version, prev_v)
        self.assertIsNotNone(result.failure_reason)
        self.assertIn("not carrying any object", result.failure_reason)

    def test_repeated_identical_simulation_runs(self):
        """I. Repeated identical simulation runs produce bit-for-bit identical results."""
        def run_standard_pipeline():
            bounds = WorkspaceBounds()
            robot = SimulatedRobot(position=SimPoint3D(x=0.0, y=0.0, z=0.2))
            world = SimulationWorld(robot=robot, workspace_bounds=bounds)

            obj = SimulatedObject(
                id="test_part",
                name="Part 1",
                position=SimPoint3D(x=0.25, y=0.10, z=0.20),
                mass=0.5,
            )
            zone = SimulatedTargetZone(
                id="zone_dest",
                position=SimPoint3D(x=0.40, y=-0.15, z=0.20),
            )
            world.add_object(obj)
            world.add_target_zone(zone)

            # Step 1: Approach
            r1 = world.execute_action(CandidateAction(
                action_id="a1", action_type=ActionType.APPROACH,
                target_position={"x": 0.25, "y": 0.10, "z": 0.20}
            ))
            # Step 2: Grasp
            r2 = world.execute_action(CandidateAction(
                action_id="a2", action_type=ActionType.GRASP, target_object_id="test_part"
            ))
            # Step 3: Move to zone
            r3 = world.execute_action(CandidateAction(
                action_id="a3", action_type=ActionType.MOVE,
                target_position={"x": 0.40, "y": -0.15, "z": 0.20}
            ))
            # Step 4: Release
            r4 = world.execute_action(CandidateAction(
                action_id="a4", action_type=ActionType.RELEASE
            ))

            return world, [r1, r2, r3, r4]

        world_a, results_a = run_standard_pipeline()
        world_b, results_b = run_standard_pipeline()

        # Compare world states
        self.assertEqual(world_a.world_state_version, world_b.world_state_version)
        self.assertEqual(world_a.simulation_time, world_b.simulation_time)
        self.assertEqual(world_a.robot.position, world_b.robot.position)
        self.assertEqual(world_a.objects["test_part"].position, world_b.objects["test_part"].position)

        # Compare all action execution results
        self.assertEqual(len(results_a), len(results_b))
        for r_a, r_b in zip(results_a, results_b):
            self.assertEqual(r_a.success, r_b.success)
            self.assertEqual(r_a.action_type, r_b.action_type)
            self.assertEqual(r_a.action_id, r_b.action_id)
            self.assertEqual(r_a.simulation_time, r_b.simulation_time)
            self.assertEqual(r_a.previous_world_state_version, r_b.previous_world_state_version)
            self.assertEqual(r_a.new_world_state_version, r_b.new_world_state_version)
            self.assertEqual(r_a.failure_reason, r_b.failure_reason)

    def test_simulation_adapter_interface(self):
        """Verify LocalSimulationAdapter fulfills SimulationAdapter contract."""
        adapter = LocalSimulationAdapter(self.world)
        state = adapter.get_world_state()
        self.assertIsNotNone(state.robot)

        # Execute action via adapter
        action = CandidateAction(
            action_id="act_adapter_test",
            action_type=ActionType.MOVE,
            target_position={"x": 0.1, "y": 0.1, "z": 0.2},
        )
        res = adapter.execute_action(action)
        self.assertTrue(res.success)
        self.assertTrue(adapter.verify_action_result(action, {"position": SimPoint3D(x=0.1, y=0.1, z=0.2)}))


if __name__ == "__main__":
    unittest.main()
