"""
SĀRTHI Isaac Sim Runtime — Physical Scene Object Prims.
Defines the tabletop, red manipulable object, and blue target zone prims
constructed in NVIDIA Isaac Sim's USD stage.

All physical dimensions, poses, and masses are read directly from
simulation.scenarios.tabletop_pick_place.TabletopPickPlaceScenario.
"""

from typing import Any, Optional
from simulation.adapters.isaac_sim import is_isaac_sim_available
from simulation.scenarios.tabletop_pick_place import (
    ObjectSpecification,
    TabletopPickPlaceScenario,
    TargetZoneSpecification,
    create_default_scenario,
)


class RedObjectPrim:
    """
    Movable, graspable red cylinder manipulated in the tabletop scenario.
    Constructed as a dynamic rigid body with PhysX collision enabled.
    """

    def __init__(
        self,
        scenario: Optional[TabletopPickPlaceScenario] = None,
        spec: Optional[ObjectSpecification] = None,
    ):
        scen = scenario or create_default_scenario()
        self.spec = spec or scen.red_object
        self.object_id = self.spec.object_id
        self.position = self.spec.initial_pose
        self.dimensions = self.spec.dimensions_m
        self.mass_kg = self.spec.mass_kg
        self.usd_prim_path = f"/World/Objects/{self.object_id}"
        self._prim: Optional[Any] = None

    def create(self, world_or_stage: Optional[Any] = None) -> Any:
        """
        Creates the dynamic rigid body cylinder prim in Isaac Sim.

        Raises:
            RuntimeError: If called outside active Isaac Sim runtime.
        """
        if not is_isaac_sim_available():
            from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
            raise IsaacSimRuntimeError(
                "Isaac Sim runtime is required to create red object prim."
            )

        from omni.isaac.core.objects import DynamicCylinder

        radius = self.dimensions.x / 2.0
        height = self.dimensions.z

        self._prim = DynamicCylinder(
            prim_path=self.usd_prim_path,
            name=self.object_id,
            position=[self.position.x, self.position.y, self.position.z],
            radius=radius,
            height=height,
            color=[0.85, 0.10, 0.10],  # Canonical high-contrast red
            mass=self.mass_kg,
        )

        if world_or_stage and hasattr(world_or_stage, "scene"):
            world_or_stage.scene.add(self._prim)

        return self._prim


class BlueTargetPrim:
    """
    Static destination zone marking acceptable placement tolerance in the workspace.
    Visual cylinder disc at tabletop level with no blocking collision.
    """

    def __init__(
        self,
        scenario: Optional[TabletopPickPlaceScenario] = None,
        spec: Optional[TargetZoneSpecification] = None,
    ):
        scen = scenario or create_default_scenario()
        self.spec = spec or scen.blue_target
        self.target_id = self.spec.target_id
        self.position = self.spec.target_pose
        self.tolerance_radius_m = self.spec.tolerance_radius_m
        self.usd_prim_path = f"/World/Targets/{self.target_id}"
        self._prim: Optional[Any] = None

    def create(self, world_or_stage: Optional[Any] = None) -> Any:
        """
        Creates the visual target zone disk in Isaac Sim.

        Raises:
            RuntimeError: If called outside active Isaac Sim runtime.
        """
        if not is_isaac_sim_available():
            from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
            raise IsaacSimRuntimeError(
                "Isaac Sim runtime is required to create blue target prim."
            )

        from omni.isaac.core.objects import FixedCylinder

        disc_height = 0.005  # Thin disc flush on table surface

        self._prim = FixedCylinder(
            prim_path=self.usd_prim_path,
            name=self.target_id,
            position=[self.position.x, self.position.y, self.position.z - (disc_height / 2.0)],
            radius=self.tolerance_radius_m,
            height=disc_height,
            color=[0.10, 0.25, 0.85],  # Canonical blue destination
        )

        if world_or_stage and hasattr(world_or_stage, "scene"):
            world_or_stage.scene.add(self._prim)

        return self._prim


class TablePrim:
    """
    Supporting tabletop surface with static collision geometry.
    Dimensions (1.4m x 1.2m x 0.05m) ensure the entire workspace,
    robot mount, objects, and obstacles remain stably supported.
    """

    def __init__(
        self,
        length_x: float = 1.40,
        width_y: float = 1.20,
        height_z: float = 0.05,
        surface_z: float = 0.20,
        usd_prim_path: str = "/World/Table",
    ):
        self.length_x = length_x
        self.width_y = width_y
        self.height_z = height_z
        self.surface_z = surface_z
        # Center of table is placed such that the upper face aligns with surface_z
        self.center_z = surface_z - (height_z / 2.0)
        self.center_x = 0.25
        self.center_y = 0.00
        self.usd_prim_path = usd_prim_path
        self._prim: Optional[Any] = None

    def create(self, world_or_stage: Optional[Any] = None) -> Any:
        """
        Creates the static tabletop cuboid in Isaac Sim.

        Raises:
            RuntimeError: If called outside active Isaac Sim runtime.
        """
        if not is_isaac_sim_available():
            from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
            raise IsaacSimRuntimeError(
                "Isaac Sim runtime is required to create table prim."
            )

        from omni.isaac.core.objects import FixedCuboid

        self._prim = FixedCuboid(
            prim_path=self.usd_prim_path,
            name="tabletop",
            position=[self.center_x, self.center_y, self.center_z],
            scale=[self.length_x, self.width_y, self.height_z],
            size=1.0,
            color=[0.75, 0.75, 0.78],  # Industrial tabletop matte gray
        )

        if world_or_stage and hasattr(world_or_stage, "scene"):
            world_or_stage.scene.add(self._prim)

        return self._prim
