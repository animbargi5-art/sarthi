"""
SĀRTHI Simulation — MuJoCo Runtime Backend.
Provides lightweight, deterministic, physics-accurate simulation for local development.
"""

from typing import Optional


def is_mujoco_available() -> bool:
    """Checks if the MuJoCo Python package is installed and importable."""
    try:
        import mujoco  # noqa: F401
        return True
    except ImportError:
        return False


def require_mujoco() -> None:
    """Raises RuntimeError if MuJoCo is not available in the active environment."""
    if not is_mujoco_available():
        raise RuntimeError(
            "MuJoCo is not installed in this environment. "
            "Install it via 'pip install mujoco' to use SarthiMuJoCoRuntime."
        )


from simulation.mujoco_runtime.scene_builder import SarthiMuJoCoSceneBuilder
from simulation.mujoco_runtime.articulation import (
    SarthiMuJoCoArticulation,
    MuJoCoIKConfig,
)
from simulation.mujoco_runtime.state_reader import SarthiMuJoCoStateReader
from simulation.mujoco_runtime.verifier import SarthiMuJoCoVerifier
from simulation.mujoco_runtime.runtime import SarthiMuJoCoRuntime

__all__ = [
    "is_mujoco_available",
    "require_mujoco",
    "SarthiMuJoCoSceneBuilder",
    "SarthiMuJoCoArticulation",
    "MuJoCoIKConfig",
    "SarthiMuJoCoStateReader",
    "SarthiMuJoCoVerifier",
    "SarthiMuJoCoRuntime",
]
