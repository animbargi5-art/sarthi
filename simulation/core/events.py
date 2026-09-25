"""
SĀRTHI Simulation Core — Action Results and Disturbance Events.
Structured representations of action execution outcomes and dynamic environmental perturbations.
"""

from enum import Enum
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field
from simulation.core.geometry import SimDimensions3D, SimPoint3D


class DisturbanceType(str, Enum):
    """Supported physical and environmental disturbance primitives."""
    PATH_BLOCKED = "PATH_BLOCKED"


class DisturbanceEvent(BaseModel):
    """
    Environmental disturbance injected into the simulation.
    Used to test adaptive recovery and cognitive replanning under non-ideal conditions.
    """
    event_id: str = Field(..., description="Unique event identifier")
    disturbance_type: DisturbanceType = Field(..., description="Type of disturbance injected")
    timestamp: float = Field(default=0.0, ge=0.0, description="Simulation timestamp when event occurs")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event parameters (e.g. obstacle position, dimensions, obstacle_id)"
    )

    @classmethod
    def create_path_blocked(
        cls,
        event_id: str,
        obstacle_id: str,
        position: SimPoint3D,
        dimensions: Optional[SimDimensions3D] = None,
        timestamp: float = 0.0,
    ) -> "DisturbanceEvent":
        """Factory method to construct a PATH_BLOCKED disturbance event."""
        dims = dimensions or SimDimensions3D(length_x=0.08, width_y=0.08, height_z=0.20)
        return cls(
            event_id=event_id,
            disturbance_type=DisturbanceType.PATH_BLOCKED,
            timestamp=timestamp,
            parameters={
                "obstacle_id": obstacle_id,
                "position": {"x": position.x, "y": position.y, "z": position.z},
                "dimensions": {
                    "length_x": dims.length_x,
                    "width_y": dims.width_y,
                    "height_z": dims.height_z,
                },
                "is_active": True,
            }
        )


class ActionExecutionResult(BaseModel):
    """
    Structured outcome of executing an action in the simulation.
    Provides complete auditability, state version tracking, and deterministic error reporting.
    """
    success: bool = Field(..., description="True if action completed without constraint violation or collision")
    action_type: str = Field(..., description="Action primitive type (APPROACH, MOVE, GRASP, etc.)")
    action_id: str = Field(..., description="Identifier of executed action")
    simulation_time: float = Field(..., description="Simulation time upon completion (seconds)")
    previous_world_state_version: int = Field(..., description="WorldState version before action execution")
    new_world_state_version: int = Field(..., description="WorldState version after action execution")
    failure_reason: Optional[str] = Field(default=None, description="Detailed failure message if unsuccessful")
    details: Dict[str, Any] = Field(default_factory=dict, description="Execution metrics or state snapshot")
