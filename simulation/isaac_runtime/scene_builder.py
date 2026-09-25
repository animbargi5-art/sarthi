"""
SĀRTHI Isaac Sim Runtime — Scene Builder.
Assembles the complete tabletop pick-and-place scene in NVIDIA Isaac Sim 6.x.

Scene composition:
- Tabletop workspace
- Articulated robot manipulator
- Red graspable dynamic cylinder
- Blue static destination zone
- Calibrated studio lighting (key + ambient)
- Perspective demo camera
"""

from typing import Any, Dict, Optional
from backend.app.decision_engine.models import Point3D
from simulation.adapters.isaac_sim import is_isaac_sim_available
from simulation.isaac_runtime.objects import BlueTargetPrim, RedObjectPrim, TablePrim
from simulation.isaac_runtime.robot_scene import RobotSceneConfig, SarthiRobotPrim
from simulation.scenarios.tabletop_pick_place import (
    TabletopPickPlaceScenario,
    create_default_scenario,
)


class SceneCameraConfig:
    """
    Deterministic perspective camera configuration.
    Framed to capture: robot arm, red object, blue target, tabletop,
    and the intermediate trajectory zone where the dynamic obstacle appears.
    """

    def __init__(
        self,
        eye: Optional[Point3D] = None,
        target: Optional[Point3D] = None,
        fov_deg: float = 55.0,
        focal_length_mm: float = 24.0,
        usd_prim_path: str = "/World/Camera",
    ):
        self.eye = eye or Point3D(x=1.15, y=0.45, z=0.85)
        self.target = target or Point3D(x=0.30, y=-0.05, z=0.20)
        self.fov_deg = fov_deg
        self.focal_length_mm = focal_length_mm
        self.usd_prim_path = usd_prim_path


class SarthiSceneBuilder:
    """
    Constructs the SĀRTHI benchmark scene inside an active Isaac Sim stage.
    Physical values are strictly sourced from TabletopPickPlaceScenario.
    """

    def __init__(
        self,
        scenario: Optional[TabletopPickPlaceScenario] = None,
        robot_config: Optional[RobotSceneConfig] = None,
        camera_config: Optional[SceneCameraConfig] = None,
    ):
        self.scenario = scenario or create_default_scenario()
        self.robot_config = robot_config or RobotSceneConfig(
            robot_id=self.scenario.robot.robot_id
        )
        self.camera_config = camera_config or SceneCameraConfig()

        # Component prim builders initialized with locked scenario parameters
        self.table_prim = TablePrim(surface_z=self.scenario.red_object.initial_pose.z)
        self.robot_prim = SarthiRobotPrim(
            scenario=self.scenario,
            config=self.robot_config,
        )
        self.red_object_prim = RedObjectPrim(scenario=self.scenario)
        self.blue_target_prim = BlueTargetPrim(scenario=self.scenario)

        self._world: Optional[Any] = None
        self._built_prims: Dict[str, Any] = {}

    def create_world(self, stage_units_in_meters: float = 1.0) -> Any:
        """Initializes the Isaac Sim World context with 60Hz physics stepping."""
        if not is_isaac_sim_available():
            from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
            raise IsaacSimRuntimeError(
                "Isaac Sim runtime is required to create an Isaac Sim World."
            )

        from omni.isaac.core import World

        self._world = World(
            stage_units_in_meters=stage_units_in_meters,
            physics_dt=1.0 / 60.0,
            rendering_dt=1.0 / 60.0,
        )
        return self._world

    def create_table(self) -> Any:
        """Constructs the tabletop support surface."""
        prim = self.table_prim.create(self._world)
        self._built_prims["table"] = prim
        return prim

    def create_robot(self) -> Any:
        """Loads and positions the configured robot arm."""
        prim = self.robot_prim.create(self._world)
        self._built_prims["robot"] = prim
        return prim

    def create_red_object(self) -> Any:
        """Spawns the red movable target cylinder."""
        prim = self.red_object_prim.create(self._world)
        self._built_prims["red_object"] = prim
        return prim

    def create_blue_target(self) -> Any:
        """Spawns the static blue destination zone."""
        prim = self.blue_target_prim.create(self._world)
        self._built_prims["blue_target"] = prim
        return prim

    def create_lighting(self) -> Dict[str, Any]:
        """Creates key directional light and ambient dome light."""
        if not is_isaac_sim_available():
            from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
            raise IsaacSimRuntimeError(
                "Isaac Sim runtime is required to configure scene lighting."
            )

        import omni.kit.commands
        from pxr import UsdLux

        lights: Dict[str, Any] = {}

        # 1. Distant Key Light
        omni.kit.commands.execute(
            "CreatePrim",
            prim_type="DistantLight",
            prim_path="/World/KeyLight",
            attributes={
                UsdLux.Tokens.inputsIntensity: 2500.0,
                UsdLux.Tokens.inputsAngle: 1.0,
            },
        )

        # 2. Dome Ambient Light
        omni.kit.commands.execute(
            "CreatePrim",
            prim_type="DomeLight",
            prim_path="/World/DomeLight",
            attributes={
                UsdLux.Tokens.inputsIntensity: 800.0,
            },
        )

        lights["key_light"] = "/World/KeyLight"
        lights["dome_light"] = "/World/DomeLight"
        self._built_prims["lighting"] = lights
        return lights

    def create_camera(self) -> Any:
        """Sets up the benchmark perspective demonstration camera."""
        if not is_isaac_sim_available():
            from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
            raise IsaacSimRuntimeError(
                "Isaac Sim runtime is required to create a viewport camera."
            )

        from omni.isaac.core.utils.rotations import lookat_to_quatf
        from pxr import Gf, UsdGeom

        # Calculate look-at quaternion from camera eye to target
        cam_eye = Gf.Vec3d(
            self.camera_config.eye.x,
            self.camera_config.eye.y,
            self.camera_config.eye.z,
        )
        cam_target = Gf.Vec3d(
            self.camera_config.target.x,
            self.camera_config.target.y,
            self.camera_config.target.z,
        )

        stage = self._world.stage if self._world else None
        if stage:
            cam_prim = UsdGeom.Camera.Define(stage, self.camera_config.usd_prim_path)
            cam_prim.GetFocalLengthAttr().Set(self.camera_config.focal_length_mm)
            self._built_prims["camera"] = cam_prim
            return cam_prim

        return None

    def build(self) -> Dict[str, Any]:
        """
        Executes complete scene assembly in proper dependency order.
        """
        self.create_world()
        self.create_table()
        self.create_robot()
        self.create_red_object()
        self.create_blue_target()
        self.create_lighting()
        self.create_camera()
        return self._built_prims
