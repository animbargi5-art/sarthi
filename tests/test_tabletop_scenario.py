"""
SĀRTHI Phase 6B — Tabletop Pick-and-Place Benchmark Scenario Tests.

IMPORTANT:
- ZERO dependencies on NVIDIA Isaac Sim packages.
- Tests run completely offline and in lightweight CPU / CI environments.
- Verifies deterministic scenario creation, geometric path intersection,
  WorldState translation, and success/failure criteria.
"""

import sys
import unittest

from backend.app.decision_engine.models import (
    ActionType,
    LastActionOutcome,
    LastActionStatus,
    ObjectState,
    Point3D,
    WorldObject,
    WorldState,
)
from simulation.core.events import DisturbanceType
from simulation.scenarios.tabletop_pick_place import (
    DisturbanceTiming,
    ObstacleSpecification,
    RobotConfiguration,
    ScenarioFailureReason,
    TabletopPickPlaceScenario,
    create_default_scenario,
    create_tabletop_world_state,
)


class TestTabletopScenario(unittest.TestCase):
    """
    Unit test suite covering Phase 6B Physical AI Scenario Specification.
    """

    # 1. Deterministic scenario creation
    def test_01_deterministic_scenario_creation(self):
        """Verify two independent scenario instantiations yield identical definitions."""
        scen1 = create_default_scenario()
        scen2 = create_default_scenario()

        self.assertEqual(scen1.model_dump(), scen2.model_dump())
        self.assertEqual(scen1.scenario_id, "tabletop_pick_and_place_mvp")
        self.assertEqual(scen1.instruction, "Move the red object to the blue target.")

    # 2. Correct red object definition
    def test_02_correct_red_object_definition(self):
        """Verify red movable object properties, mass, graspability, and initial coordinates."""
        scenario = create_default_scenario()
        obj = scenario.red_object

        self.assertEqual(obj.object_id, "red_object_01")
        self.assertTrue(obj.is_movable)
        self.assertTrue(obj.is_graspable)
        self.assertEqual(obj.initial_pose.x, 0.25)
        self.assertEqual(obj.initial_pose.y, 0.15)
        self.assertEqual(obj.initial_pose.z, 0.20)
        self.assertEqual(obj.mass_kg, 0.5)

    # 3. Correct blue target definition
    def test_03_correct_blue_target_definition(self):
        """Verify blue target zone properties, tolerance, and coordinates."""
        scenario = create_default_scenario()
        target = scenario.blue_target

        self.assertEqual(target.target_id, "blue_target_zone")
        self.assertFalse(target.is_movable)
        self.assertEqual(target.target_pose.x, 0.40)
        self.assertEqual(target.target_pose.y, -0.20)
        self.assertEqual(target.target_pose.z, 0.20)
        self.assertEqual(target.tolerance_radius_m, 0.06)

    # 4. Robot configuration exists
    def test_04_robot_configuration_exists(self):
        """Verify manipulator configuration, reach, degrees of freedom, and configurability."""
        scenario = create_default_scenario(robot_id="custom_franka_arm")
        robot = scenario.robot

        self.assertEqual(robot.robot_id, "custom_franka_arm")
        self.assertEqual(robot.dof, 7)
        self.assertEqual(robot.base_pose.x, 0.0)
        self.assertEqual(robot.base_pose.y, 0.0)
        self.assertEqual(robot.base_pose.z, 0.20)
        self.assertEqual(robot.max_reach_m, 0.85)
        self.assertEqual(robot.max_payload_kg, 3.0)

    # 5. Workspace is valid
    def test_05_workspace_is_valid(self):
        """Verify workspace limits are geometrically consistent and enclose all entities."""
        scenario = create_default_scenario()
        ws = scenario.workspace

        # Bound consistency
        self.assertLess(ws.min_x, ws.max_x)
        self.assertLess(ws.min_y, ws.max_y)
        self.assertLess(ws.min_z, ws.max_z)

        # Entity containment checks
        for entity_pos in [
            scenario.robot.base_pose,
            scenario.red_object.initial_pose,
            scenario.blue_target.target_pose,
            scenario.obstacle.position,
        ]:
            self.assertGreaterEqual(entity_pos.x, ws.min_x)
            self.assertLessEqual(entity_pos.x, ws.max_x)
            self.assertGreaterEqual(entity_pos.y, ws.min_y)
            self.assertLessEqual(entity_pos.y, ws.max_y)
            self.assertGreaterEqual(entity_pos.z, ws.min_z)
            self.assertLessEqual(entity_pos.z, ws.max_z)

    # 6. Payload is within SĀRTHI limits
    def test_06_payload_is_within_sarthi_limits(self):
        """Verify red object mass does not exceed the robot's rated payload capacity."""
        scenario = create_default_scenario()
        self.assertLessEqual(scenario.red_object.mass_kg, scenario.robot.max_payload_kg)
        self.assertGreater(scenario.red_object.mass_kg, 0.0)

    # 7. Disturbance is defined
    def test_07_disturbance_is_defined(self):
        """Verify obstacle and PATH_BLOCKED disturbance event definitions."""
        scenario = create_default_scenario()
        obs = scenario.obstacle

        self.assertEqual(obs.obstacle_id, "blocking_barrier_01")
        self.assertTrue(obs.is_dynamic)

        event = scenario.create_disturbance_event()
        self.assertEqual(event.disturbance_type, DisturbanceType.PATH_BLOCKED)
        self.assertEqual(event.parameters["obstacle_id"], "blocking_barrier_01")
        self.assertEqual(event.parameters["position"]["x"], obs.position.x)
        self.assertEqual(event.parameters["position"]["y"], obs.position.y)

    # 8. Disturbance occurs after GRASP
    def test_08_disturbance_occurs_after_grasp(self):
        """Verify that the benchmark disturbance timing is configured AFTER_GRASP."""
        scenario = create_default_scenario()
        self.assertEqual(scenario.disturbance_timing, DisturbanceTiming.AFTER_GRASP)

    # 9. Obstacle intersects the nominal path
    def test_09_obstacle_intersects_nominal_path(self):
        """Verify geometric algorithm confirms the obstacle obstructs nominal trajectory."""
        scenario = create_default_scenario()
        # Default obstacle at (0.325, -0.025) is directly on the line from (0.25, 0.15) to (0.40, -0.20)
        self.assertTrue(scenario.obstacle_intersects_nominal_path())

        # An obstacle placed far away should NOT intersect
        cleared_obstacle = ObstacleSpecification(
            obstacle_id="remote_obs",
            position=Point3D(x=0.70, y=0.70, z=0.20),
        )
        scen_cleared = TabletopPickPlaceScenario(obstacle=cleared_obstacle)
        self.assertFalse(scen_cleared.obstacle_intersects_nominal_path())

    # 10. Scenario converts to existing WorldState
    def test_10_scenario_converts_to_existing_worldstate(self):
        """Verify conversion to canonical SĀRTHI WorldState without schema duplication."""
        scenario = create_default_scenario()

        # Without disturbance
        ws_nominal = scenario.to_world_state(with_disturbance=False)
        self.assertIsInstance(ws_nominal, WorldState)
        self.assertEqual(len(ws_nominal.objects), 1)
        self.assertEqual(ws_nominal.objects[0].id, "red_object_01")
        self.assertTrue(ws_nominal.objects[0].is_target)
        self.assertFalse(ws_nominal.objects[0].is_obstacle)
        self.assertFalse(ws_nominal.environment.dynamic_obstacles_detected)

        # With disturbance
        ws_disturbed = scenario.to_world_state(with_disturbance=True)
        self.assertIsInstance(ws_disturbed, WorldState)
        self.assertEqual(len(ws_disturbed.objects), 2)
        self.assertTrue(ws_disturbed.objects[1].is_obstacle)
        self.assertEqual(ws_disturbed.objects[1].id, "blocking_barrier_01")
        self.assertTrue(ws_disturbed.environment.dynamic_obstacles_detected)

        # Convenience function
        ws_conv = create_tabletop_world_state(with_disturbance=True)
        self.assertIsInstance(ws_conv, WorldState)
        self.assertEqual(len(ws_conv.objects), 2)

    # 11. Success conditions are represented
    def test_11_success_conditions_represented(self):
        """Verify check_success_conditions evaluates task criteria correctly."""
        scenario = create_default_scenario()
        ws_initial = scenario.to_world_state()

        # Initial state should not be completed (object not at target zone)
        success_init, reason_init = scenario.check_success_conditions(ws_initial)
        self.assertFalse(success_init)
        self.assertIsNotNone(reason_init)

        # Build a valid completed WorldState
        completed_robot = ws_initial.robot.model_copy(
            update={"gripper_open": True, "holding_object_id": None}
        )
        completed_object = WorldObject(
            id=scenario.red_object.object_id,
            name=scenario.red_object.name,
            position=scenario.blue_target.target_pose,  # At blue target
            bounding_radius_m=0.05,
            mass_kg=0.5,
            state=ObjectState.FREE,
            is_target=True,
            is_obstacle=False,
        )
        completed_outcome = LastActionOutcome(
            action_type=ActionType.RELEASE,
            status=LastActionStatus.SUCCESS,
        )
        ws_completed = ws_initial.model_copy(
            update={
                "robot": completed_robot,
                "objects": [completed_object],
                "last_action_outcome": completed_outcome,
            }
        )

        success_done, reason_done = scenario.check_success_conditions(ws_completed)
        self.assertTrue(success_done)
        self.assertIsNone(reason_done)

        # Verify structured failure reasons enum exists
        self.assertEqual(ScenarioFailureReason.OBJECT_UNREACHABLE.value, "OBJECT_UNREACHABLE")
        self.assertEqual(ScenarioFailureReason.BLOCKED_PATH_NO_ALTERNATIVE.value, "BLOCKED_PATH_NO_ALTERNATIVE")

    # 12. No Isaac Sim installation is required
    def test_12_no_isaac_sim_installation_required(self):
        """Verify scenario tests operate without requiring Isaac Sim packages."""
        self.assertNotIn("isaacsim", sys.modules)
        self.assertNotIn("omni.isaac.core", sys.modules)


if __name__ == "__main__":
    unittest.main()
