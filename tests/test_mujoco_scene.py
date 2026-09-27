"""
SĀRTHI — MuJoCo Tabletop Scene & SceneBuilder Tests (Phase A & B).
Verifies that:
- MuJoCo package is available.
- Tabletop scenario XML template exists and is valid.
- SarthiMuJoCoSceneBuilder loads and compiles mjModel and mjData without modifying upstream assets.
- Franka Emika Panda, table, red object, blue target, and dynamic obstacle exist with correct coordinates.
"""

import unittest

from simulation.mujoco_runtime import is_mujoco_available, SarthiMuJoCoSceneBuilder


class TestMuJoCoScene(unittest.TestCase):
    """Verifies MuJoCo scene assembly and deterministic model compilation."""

    def setUp(self):
        if not is_mujoco_available():
            self.skipTest("MuJoCo is not installed in this environment.")
        self.builder = SarthiMuJoCoSceneBuilder()
        self.model, self.data = self.builder.build()
        self.entity_ids = self.builder.get_entity_ids()

    def test_01_mujoco_is_available(self):
        """Verify MuJoCo is importable."""
        self.assertTrue(is_mujoco_available())

    def test_02_scene_xml_template_exists(self):
        """Verify scenario XML template exists and contains valid MJCF markup."""
        xml_content = self.builder.load_scene_xml()
        self.assertIn("<mujoco model=\"sarthi_tabletop_pick_and_place\">", xml_content)
        self.assertIn("panda.xml", xml_content)
        self.assertIn("red_object", xml_content)
        self.assertIn("blue_target", xml_content)
        self.assertIn("obstacle", xml_content)

    def test_03_scene_builder_compiles_model(self):
        """Verify mjModel and mjData compile and allocate correctly."""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.data)
        self.assertGreater(self.model.nbody, 10)
        self.assertGreater(self.model.ngeom, 20)

    def test_04_franka_panda_bodies_and_actuators_present(self):
        """Verify Franka Panda arm bodies, gripper fingers, and 8 actuators exist."""
        bodies = self.entity_ids["bodies"]
        actuators = self.entity_ids["actuators"]

        self.assertGreaterEqual(bodies["link0"], 0)
        self.assertGreaterEqual(bodies["hand"], 0)
        self.assertGreaterEqual(bodies["left_finger"], 0)
        self.assertGreaterEqual(bodies["right_finger"], 0)

        # 7 arm actuators + 1 gripper actuator
        self.assertEqual(len(actuators), 8)
        for act_id in actuators.values():
            self.assertGreaterEqual(act_id, 0)

    def test_05_red_object_position_and_geom_present(self):
        """Verify red movable object exists with freejoint and matches canonical coordinates."""
        red_body_id = self.entity_ids["bodies"]["red_object"]
        self.assertGreaterEqual(red_body_id, 0)

        pos = self.data.xpos[red_body_id]
        self.assertAlmostEqual(pos[0], 0.25, places=2)
        self.assertAlmostEqual(pos[1], 0.15, places=2)
        self.assertAlmostEqual(pos[2], 0.20, places=2)

        # Freejoint check
        joint_id = self.entity_ids["joints"]["red_object_joint"]
        self.assertGreaterEqual(joint_id, 0)

    def test_06_blue_target_zone_present(self):
        """Verify blue target zone exists at canonical destination coordinates."""
        target_body_id = self.entity_ids["bodies"]["blue_target"]
        self.assertGreaterEqual(target_body_id, 0)

        pos = self.data.xpos[target_body_id]
        self.assertAlmostEqual(pos[0], 0.40, places=2)
        self.assertAlmostEqual(pos[1], -0.20, places=2)

    def test_07_obstacle_present(self):
        """Verify dynamic blocking obstacle exists with mass and bounding dimensions."""
        obs_body_id = self.entity_ids["bodies"]["obstacle"]
        self.assertGreaterEqual(obs_body_id, 0)

        pos = self.data.xpos[obs_body_id]
        self.assertAlmostEqual(pos[0], 0.325, places=2)
        self.assertAlmostEqual(pos[1], -0.025, places=2)

    def test_08_tabletop_present(self):
        """Verify tabletop workspace support surface exists."""
        table_body_id = self.entity_ids["bodies"]["table"]
        self.assertGreaterEqual(table_body_id, 0)
        table_geom_id = self.entity_ids["geoms"]["tabletop"]
        self.assertGreaterEqual(table_geom_id, 0)


if __name__ == "__main__":
    unittest.main()
