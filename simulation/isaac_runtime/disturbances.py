"""
SĀRTHI Isaac Sim Runtime — Dynamic Disturbance Management.
Handles spawning and physical activation of the dynamic PATH_BLOCKED obstacle
inside NVIDIA Isaac Sim's USD stage.

Strict Architectural Separation:
- The disturbance changes physical reality (spawns rigid-body collision obstacle).
- The disturbance NEVER commands or hardcodes 'REPOSITION'.
- The Decision Engine observes the altered WorldState and plans independently.
"""

from typing import Any, Optional
from simulation.adapters.isaac_sim import is_isaac_sim_available
from simulation.scenarios.tabletop_pick_place import (
    ObstacleSpecification,
    TabletopPickPlaceScenario,
    create_default_scenario,
)


class PathBlockedDisturbance:
    """
    Manages the dynamic PATH_BLOCKED obstacle inside NVIDIA Isaac Sim.

    Lifecycle:
    1. Initial state: absent or deactivated (invisible, no collision).
    2. Injected state: spawned at the locked scenario midpoint with rigid-body collision.
    """

    def __init__(
        self,
        scenario: Optional[TabletopPickPlaceScenario] = None,
        obstacle_spec: Optional[ObstacleSpecification] = None,
    ):
        scen = scenario or create_default_scenario()
        self.spec = obstacle_spec or scen.obstacle
        self.obstacle_id = self.spec.obstacle_id
        self.position = self.spec.position
        self.dimensions = self.spec.dimensions_m
        self.mass_kg = self.spec.mass_kg
        self.usd_prim_path = f"/World/Obstacles/{self.obstacle_id}"
        self._is_active: bool = False
        self._prim: Optional[Any] = None

    @property
    def is_active(self) -> bool:
        """True if the obstacle is actively present in the physics stage."""
        return self._is_active

    def spawn(self, world_or_stage: Optional[Any] = None) -> Any:
        """
        Dynamically spawns the obstacle in Isaac Sim.

        Raises:
            RuntimeError: If called outside an active Isaac Sim environment.
        """
        if not is_isaac_sim_available():
            from simulation.isaac_runtime.runtime import IsaacSimRuntimeError
            raise IsaacSimRuntimeError(
                "Isaac Sim runtime is required to spawn dynamic obstacle prims."
            )

        # Isaac Sim USD stage imports (executed only inside active Isaac Sim environment)
        from omni.isaac.core.objects import DynamicCuboid

        self._prim = DynamicCuboid(
            prim_path=self.usd_prim_path,
            name=self.obstacle_id,
            position=[self.position.x, self.position.y, self.position.z],
            scale=[self.dimensions.x, self.dimensions.y, self.dimensions.z],
            size=1.0,
            color=[0.8, 0.4, 0.1],  # Distinct warning amber
            mass=self.mass_kg,
        )

        if world_or_stage and hasattr(world_or_stage, "scene"):
            world_or_stage.scene.add(self._prim)

        self._is_active = True
        return self._prim

    def deactivate(self) -> None:
        """Deactivates the obstacle in the simulation."""
        self._is_active = False
