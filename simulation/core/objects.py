"""
SĀRTHI Simulation Core — Physical Workspace Entities.
Definitions for simulated manipulable objects, stationary obstacles, and target zones.
"""

from typing import Optional
from pydantic import BaseModel, Field
from simulation.core.geometry import SimDimensions3D, SimPoint3D


class SimulatedObject(BaseModel):
    """
    Physical object located in the simulation environment.
    Can be picked, carried, and placed by the simulated robot.
    """
    id: str = Field(..., description="Unique entity identifier")
    name: str = Field(..., description="Human-readable object name")
    position: SimPoint3D = Field(..., description="3D center-of-mass coordinates (m)")
    dimensions: SimDimensions3D = Field(
        default_factory=SimDimensions3D,
        description="Physical bounding dimensions"
    )
    mass: float = Field(default=0.5, gt=0.0, description="Mass of object in kilograms")
    is_graspable: bool = Field(default=True, description="True if object can be grasped by gripper")
    current_holder: Optional[str] = Field(default=None, description="Entity currently holding object, or None")
    is_target: bool = Field(default=False, description="True if this object is the manipulation target")

    @property
    def bounding_radius(self) -> float:
        """Enclosing collision sphere radius."""
        return self.dimensions.bounding_radius

    def move_to(self, new_position: SimPoint3D) -> None:
        """Update object 3D position."""
        self.position = new_position


class SimulatedObstacle(BaseModel):
    """
    Obstacle or barrier obstructing robot trajectory.
    Can be dynamically activated, spawned, or deactivated as disturbances.
    """
    id: str = Field(..., description="Unique obstacle identifier")
    name: str = Field(default="Obstacle", description="Human-readable name")
    position: SimPoint3D = Field(..., description="3D center coordinates of obstacle (m)")
    dimensions: SimDimensions3D = Field(
        default_factory=lambda: SimDimensions3D(length_x=0.08, width_y=0.08, height_z=0.15),
        description="Bounding dimensions"
    )
    is_active: bool = Field(default=True, description="True if obstacle actively blocks path")

    @property
    def bounding_radius(self) -> float:
        """Effective collision radius."""
        return self.dimensions.bounding_radius


class SimulatedTargetZone(BaseModel):
    """
    Designated target area for object placement or task completion.
    """
    id: str = Field(..., description="Unique zone identifier")
    name: str = Field(default="Target Zone", description="Target zone label")
    position: SimPoint3D = Field(..., description="Center coordinates of target zone (m)")
    tolerance_radius: float = Field(default=0.05, gt=0.0, description="Acceptable placement radius (m)")

    def contains(self, point: SimPoint3D) -> bool:
        """Check if 3D point is within horizontal tolerance radius of target zone."""
        horizontal_dist = ((point.x - self.position.x) ** 2 + (point.y - self.position.y) ** 2) ** 0.5
        return horizontal_dist <= self.tolerance_radius
