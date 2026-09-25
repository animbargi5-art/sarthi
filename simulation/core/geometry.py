"""
SĀRTHI Simulation Core — Geometric Primitives and Spatial Utilities.
Deterministic 3D vector mathematics, bounding volumes, and collision checks.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Tuple
from pydantic import BaseModel, ConfigDict, Field


class SimPoint3D(BaseModel):
    """Cartesian 3D coordinate in simulation world frame (meters)."""
    model_config = ConfigDict(frozen=True)

    x: float = Field(..., description="X coordinate (m)")
    y: float = Field(..., description="Y coordinate (m)")
    z: float = Field(..., description="Z coordinate (m)")

    def distance_to(self, other: SimPoint3D) -> float:
        """Euclidean distance to another 3D point."""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )

    def offset(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> SimPoint3D:
        """Return a new SimPoint3D offset by delta values."""
        return SimPoint3D(x=self.x + dx, y=self.y + dy, z=self.z + dz)


class SimOrientation3D(BaseModel):
    """Euler orientation angles in simulation world frame (radians)."""
    model_config = ConfigDict(frozen=True)

    roll: float = Field(default=0.0, description="Roll angle around X-axis (rad)")
    pitch: float = Field(default=0.0, description="Pitch angle around Y-axis (rad)")
    yaw: float = Field(default=0.0, description="Yaw angle around Z-axis (rad)")


class SimDimensions3D(BaseModel):
    """Bounding dimensions of a physical simulation entity (meters)."""
    model_config = ConfigDict(frozen=True)

    length_x: float = Field(default=0.05, gt=0.0, description="Dimension along X axis (m)")
    width_y: float = Field(default=0.05, gt=0.0, description="Dimension along Y axis (m)")
    height_z: float = Field(default=0.05, gt=0.0, description="Dimension along Z axis (m)")

    @property
    def bounding_radius(self) -> float:
        """Half-diagonal sphere radius enclosing the box."""
        return math.sqrt(
            (self.length_x / 2.0) ** 2 +
            (self.width_y / 2.0) ** 2 +
            (self.height_z / 2.0) ** 2
        )


class WorkspaceBounds(BaseModel):
    """Axis-aligned workspace boundaries."""
    model_config = ConfigDict(frozen=True)

    min_x: float = Field(default=-0.8, description="Minimum X coordinate (m)")
    max_x: float = Field(default=0.8, description="Maximum X coordinate (m)")
    min_y: float = Field(default=-0.8, description="Minimum Y coordinate (m)")
    max_y: float = Field(default=0.8, description="Maximum Y coordinate (m)")
    min_z: float = Field(default=0.0, description="Minimum Z coordinate (m)")
    max_z: float = Field(default=1.2, description="Maximum Z coordinate (m)")

    def contains(self, point: SimPoint3D) -> bool:
        """Check if point lies strictly within workspace boundaries."""
        return (
            self.min_x <= point.x <= self.max_x and
            self.min_y <= point.y <= self.max_y and
            self.min_z <= point.z <= self.max_z
        )


def distance_point_to_line_segment(p: SimPoint3D, a: SimPoint3D, b: SimPoint3D) -> float:
    """
    Computes shortest Euclidean distance from point p to 3D line segment ab.
    Used for trajectory collision and obstacle proximity checking.
    """
    ab_x = b.x - a.x
    ab_y = b.y - a.y
    ab_z = b.z - a.z
    ab_len_sq = ab_x ** 2 + ab_y ** 2 + ab_z ** 2

    if ab_len_sq < 1e-9:
        return p.distance_to(a)

    ap_x = p.x - a.x
    ap_y = p.y - a.y
    ap_z = p.z - a.z

    t = (ap_x * ab_x + ap_y * ab_y + ap_z * ab_z) / ab_len_sq
    t = max(0.0, min(1.0, t))

    closest_x = a.x + t * ab_x
    closest_y = a.y + t * ab_y
    closest_z = a.z + t * ab_z

    return math.sqrt(
        (p.x - closest_x) ** 2 +
        (p.y - closest_y) ** 2 +
        (p.z - closest_z) ** 2
    )
