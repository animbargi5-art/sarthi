"""
SĀRTHI Simulation Core — Simulated Robotic Manipulator.
Represents robot kinematic state, end-effector pose, gripper status, and payload link.
"""

from typing import Optional
from pydantic import BaseModel, Field
from simulation.core.geometry import SimDimensions3D, SimOrientation3D, SimPoint3D, WorkspaceBounds


class SimulatedRobot(BaseModel):
    """
    Deterministic simulated robotic manipulator end-effector and kinematic interface.
    Tracks spatial position, gripper jaw state, payload attachment, and kinematic bounds.
    """
    robot_id: str = Field(default="sim_robot_01", description="Unique robot identifier")
    position: SimPoint3D = Field(
        default_factory=lambda: SimPoint3D(x=0.0, y=0.0, z=0.2),
        description="Current end-effector 3D position (m)"
    )
    orientation: SimOrientation3D = Field(
        default_factory=SimOrientation3D,
        description="Current end-effector orientation (rad)"
    )
    gripper_open: bool = Field(default=True, description="True if gripper is open, False if closed/holding")
    carrying_object_id: Optional[str] = Field(default=None, description="ID of grasped object, or None")
    max_reach: float = Field(default=0.85, gt=0.0, description="Maximum reach from base origin (m)")
    max_payload_kg: float = Field(default=3.0, gt=0.0, description="Maximum arm payload capacity (kg)")
    workspace_limits: WorkspaceBounds = Field(
        default_factory=WorkspaceBounds,
        description="Physical boundary box of robot operational envelope"
    )

    def can_reach(self, target: SimPoint3D) -> bool:
        """Verify target position is within maximum spherical kinematic reach from origin."""
        origin = SimPoint3D(x=0.0, y=0.0, z=0.0)
        return origin.distance_to(target) <= self.max_reach

    def is_in_workspace(self, target: SimPoint3D) -> bool:
        """Verify target position lies inside designated workspace boundaries."""
        return self.workspace_limits.contains(target)

    def move_to(self, new_position: SimPoint3D) -> None:
        """Update end-effector position."""
        self.position = new_position

    def set_gripper(self, open_state: bool) -> None:
        """Set gripper open or closed."""
        self.gripper_open = open_state

    def attach_object(self, object_id: str) -> None:
        """Attach an object to the end-effector."""
        self.carrying_object_id = object_id
        self.gripper_open = False

    def detach_object(self) -> Optional[str]:
        """Release currently held object."""
        released_id = self.carrying_object_id
        self.carrying_object_id = None
        self.gripper_open = True
        return released_id
