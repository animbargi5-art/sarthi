"""
SĀRTHI Isaac Sim Runtime — Robot Manipulator Scene Integration.
Manages loading, articulation binding, and base pose placement of the
configured tabletop manipulator inside NVIDIA Isaac Sim.

Strict Architecture:
- Robot identifier defaults to 'tabletop_manipulator'.
- Robot asset path is strictly configurable.
- If asset path is missing or invalid, raises a clear error (never silently substitutes).
"""

from typing import Any, Optional
from simulation.adapters.isaac_sim import is_isaac_sim_available
from simulation.scenarios.tabletop_pick_place import (
    RobotConfiguration,
    TabletopPickPlaceScenario,
    create_default_scenario,
)


class RobotSceneConfig:
    """
    Configuration parameters for robot asset loading in Isaac Sim.
    """

    def __init__(
        self,
        robot_id: str = "tabletop_manipulator",
        robot_asset_path: Optional[str] = None,
        usd_prim_path: str = "/World/Robots/tabletop_manipulator",
    ):
        self.robot_id = robot_id
        self.robot_asset_path = robot_asset_path
        self.usd_prim_path = usd_prim_path

    def __repr__(self) -> str:
        return (
            f"RobotSceneConfig(robot_id={self.robot_id!r}, "
            f"robot_asset_path={self.robot_asset_path!r}, "
            f"usd_prim_path={self.usd_prim_path!r})"
        )


class SarthiRobotPrim:
    """
    Manages the robot manipulator prim and articulation in Isaac Sim.
    """

    def __init__(
        self,
        scenario: Optional[TabletopPickPlaceScenario] = None,
        config: Optional[RobotSceneConfig] = None,
        spec: Optional[RobotConfiguration] = None,
    ):
        scen = scenario or create_default_scenario()
        self.spec = spec or scen.robot
        self.config = config or RobotSceneConfig(robot_id=self.spec.robot_id)
        self.robot_id = self.config.robot_id
        self.robot_asset_path = self.config.robot_asset_path
        self.base_pose = self.spec.base_pose
        self.usd_prim_path = self.config.usd_prim_path
        self._articulation: Optional[Any] = None

    def create(self, world_or_stage: Optional[Any] = None) -> Any:
        """
        Loads and instantiates the configured robot articulation in Isaac Sim.

        Raises:
            IsaacSimRuntimeError: If Isaac Sim is unavailable or if the configured
                                  robot asset path is missing.
        """
        if not is_isaac_sim_available():
            from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
            raise IsaacSimRuntimeError(
                "Isaac Sim runtime is required to instantiate robot assets."
            )

        if not self.robot_asset_path:
            from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
            raise IsaacSimRuntimeError(
                f"Missing robot asset path for '{self.robot_id}'. "
                "Specify a valid USD robot asset path via RobotSceneConfig. "
                "Silent substitution of robot models is strictly disallowed."
            )

        from omni.isaac.core.robots import Robot
        from omni.isaac.core.utils.stage import add_reference_to_stage

        # Reference external robot USD into stage
        add_reference_to_stage(
            usd_path=self.robot_asset_path,
            prim_path=self.usd_prim_path,
        )

        self._articulation = Robot(
            prim_path=self.usd_prim_path,
            name=self.robot_id,
            position=[self.base_pose.x, self.base_pose.y, self.base_pose.z],
        )

        if world_or_stage and hasattr(world_or_stage, "scene"):
            world_or_stage.scene.add(self._articulation)

        return self._articulation
