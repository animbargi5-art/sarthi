"""
SĀRTHI MuJoCo Runtime — Scene Builder.
Assembles the complete tabletop pick-and-place scene in MuJoCo:
- Franka Emika Panda 7-DOF manipulator with 2-finger parallel gripper
- Tabletop workspace
- Red graspable dynamic cylinder
- Blue static destination zone
- Dynamic path-blocking obstacle
- Calibrated studio lighting and demo camera
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from simulation.mujoco_runtime import require_mujoco
from simulation.scenarios.tabletop_pick_place import (
    TabletopPickPlaceScenario,
    create_default_scenario,
)


class SarthiMuJoCoSceneBuilder:
    """
    Constructs and compiles the SĀRTHI benchmark scene in MuJoCo.
    Loads the Menagerie Franka Panda model without modifying upstream assets.
    """

    DEFAULT_SCENARIO_XML_NAME = "tabletop_pick_and_place_mvp.xml"

    def __init__(
        self,
        scenario: Optional[TabletopPickPlaceScenario] = None,
        menagerie_panda_dir: Optional[str] = None,
        scenario_xml_path: Optional[str] = None,
    ):
        require_mujoco()
        import mujoco

        self.scenario = scenario or create_default_scenario()

        # Resolve project root and paths
        module_dir = Path(__file__).resolve().parent
        self.project_root = module_dir.parent.parent

        self.menagerie_dir = (
            Path(menagerie_panda_dir)
            if menagerie_panda_dir
            else self.project_root / "mujoco_menagerie" / "franka_emika_panda"
        )
        self.scenario_xml_path = (
            Path(scenario_xml_path)
            if scenario_xml_path
            else module_dir / "scenarios" / self.DEFAULT_SCENARIO_XML_NAME
        )

        self._model: Optional[Any] = None
        self._data: Optional[Any] = None

    def _load_menagerie_assets(self) -> Dict[str, bytes]:
        """
        Loads all mesh and XML assets from MuJoCo Menagerie into a memory buffer.
        Enables seamless inclusion regardless of platform or filesystem path encoding.
        """
        assets: Dict[str, bytes] = {}
        assets_dir = self.menagerie_dir / "assets"

        if assets_dir.exists():
            for filename in os.listdir(assets_dir):
                filepath = assets_dir / filename
                if filepath.is_file():
                    with open(filepath, "rb") as fp:
                        assets[f"assets/{filename}"] = fp.read()

        panda_xml = self.menagerie_dir / "panda.xml"
        if panda_xml.exists():
            with open(panda_xml, "rb") as fp:
                assets["panda.xml"] = fp.read()
        else:
            raise FileNotFoundError(f"Franka Panda model not found at {panda_xml}")

        return assets

    def load_scene_xml(self) -> str:
        """Reads the SĀRTHI tabletop scenario XML template."""
        if not self.scenario_xml_path.exists():
            raise FileNotFoundError(
                f"Scenario XML template not found at {self.scenario_xml_path}"
            )
        with open(self.scenario_xml_path, "r", encoding="utf-8") as fp:
            return fp.read()

    def build(self) -> Tuple[Any, Any]:
        """
        Compiles the complete MuJoCo model and allocates runtime data.
        Returns:
            Tuple of (mjModel, mjData)
        """
        import mujoco

        xml_string = self.load_scene_xml()
        assets = self._load_menagerie_assets()

        self._model = mujoco.MjModel.from_xml_string(xml_string, assets=assets)
        self._data = mujoco.MjData(self._model)

        # Forward kinematics and collision geometry initialization
        mujoco.mj_step(self._model, self._data)

        # Validate presence of canonical scene components
        self._validate_model_components()

        return self._model, self._data

    def _validate_model_components(self) -> None:
        """Asserts that all essential SĀRTHI benchmark bodies and actuators exist."""
        import mujoco

        required_bodies = [
            "link0",        # Robot base
            "hand",         # Robot end-effector
            "left_finger",  # Gripper jaw 1
            "right_finger", # Gripper jaw 2
            "table",        # Support workspace
            "red_object",   # Movable target object
            "blue_target",  # Destination zone
            "obstacle",     # Dynamic blocking obstacle
        ]

        for body_name in required_bodies:
            body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(
                    f"Required body '{body_name}' was not found in compiled MuJoCo model."
                )

        # Verify Franka Panda actuators (7 arm + 1 gripper)
        if self._model.nu < 8:
            raise ValueError(
                f"Expected at least 8 actuators for Franka Panda, found {self._model.nu}."
            )

    def get_entity_ids(self) -> Dict[str, Dict[str, int]]:
        """
        Returns a lookup map of canonical entity names to their MuJoCo index IDs.
        """
        if self._model is None:
            raise RuntimeError("Model has not been built yet. Call build() first.")

        import mujoco

        def get_id(obj_type, name):
            return mujoco.mj_name2id(self._model, obj_type, name)

        return {
            "bodies": {
                "link0": get_id(mujoco.mjtObj.mjOBJ_BODY, "link0"),
                "hand": get_id(mujoco.mjtObj.mjOBJ_BODY, "hand"),
                "left_finger": get_id(mujoco.mjtObj.mjOBJ_BODY, "left_finger"),
                "right_finger": get_id(mujoco.mjtObj.mjOBJ_BODY, "right_finger"),
                "table": get_id(mujoco.mjtObj.mjOBJ_BODY, "table"),
                "red_object": get_id(mujoco.mjtObj.mjOBJ_BODY, "red_object"),
                "blue_target": get_id(mujoco.mjtObj.mjOBJ_BODY, "blue_target"),
                "obstacle": get_id(mujoco.mjtObj.mjOBJ_BODY, "obstacle"),
            },
            "geoms": {
                "tabletop": get_id(mujoco.mjtObj.mjOBJ_GEOM, "tabletop"),
                "red_object_geom": get_id(mujoco.mjtObj.mjOBJ_GEOM, "red_object_geom"),
                "blue_target_zone": get_id(mujoco.mjtObj.mjOBJ_GEOM, "blue_target_zone"),
                "obstacle_geom": get_id(mujoco.mjtObj.mjOBJ_GEOM, "obstacle_geom"),
            },
            "joints": {
                "joint1": get_id(mujoco.mjtObj.mjOBJ_JOINT, "joint1"),
                "joint2": get_id(mujoco.mjtObj.mjOBJ_JOINT, "joint2"),
                "joint3": get_id(mujoco.mjtObj.mjOBJ_JOINT, "joint3"),
                "joint4": get_id(mujoco.mjtObj.mjOBJ_JOINT, "joint4"),
                "joint5": get_id(mujoco.mjtObj.mjOBJ_JOINT, "joint5"),
                "joint6": get_id(mujoco.mjtObj.mjOBJ_JOINT, "joint6"),
                "joint7": get_id(mujoco.mjtObj.mjOBJ_JOINT, "joint7"),
                "finger_joint1": get_id(mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1"),
                "finger_joint2": get_id(mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2"),
                "red_object_joint": get_id(mujoco.mjtObj.mjOBJ_JOINT, "red_object_joint"),
            },
            "actuators": {
                "actuator1": get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator1"),
                "actuator2": get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator2"),
                "actuator3": get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator3"),
                "actuator4": get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator4"),
                "actuator5": get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator5"),
                "actuator6": get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator6"),
                "actuator7": get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator7"),
                "actuator8": get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"),
            },
        }
