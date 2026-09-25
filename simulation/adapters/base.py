"""
SĀRTHI Simulation Adapters — Abstract Simulation Interface and Local Adapter.
Defines the standard bridge protocol for connecting any physics simulator
(local deterministic core, NVIDIA Isaac Sim, or mock testbeds) to the SĀRTHI decision engine.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

from backend.app.decision_engine.models import CandidateAction, WorldState
from simulation.core.events import ActionExecutionResult, DisturbanceEvent
from simulation.core.world import SimulationWorld


class SimulationAdapter(ABC):
    """
    Abstract base interface for simulation environments.
    Standardizes interaction across local simulation and future NVIDIA Isaac Sim bridges.
    """

    @abstractmethod
    def get_world_state(self) -> WorldState:
        """
        Fetches the current physical state of the simulation as a structured WorldState.
        """
        pass

    @abstractmethod
    def execute_action(self, action: Union[CandidateAction, Dict[str, Any]]) -> ActionExecutionResult:
        """
        Dispatches a candidate action primitive to the simulation and waits for execution.
        """
        pass

    @abstractmethod
    def inject_disturbance(self, disturbance: DisturbanceEvent) -> bool:
        """
        Injects a physical or environmental disturbance into the active simulation.
        """
        pass

    @abstractmethod
    def verify_action_result(
        self,
        action: Union[CandidateAction, Dict[str, Any]],
        expected_outcome: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Verifies that the executed action achieved the expected physical state change.
        """
        pass


class LocalSimulationAdapter(SimulationAdapter):
    """
    Concrete adapter bridging the deterministic local SimulationWorld
    with the SĀRTHI Decision Engine.
    """

    def __init__(self, world: Optional[SimulationWorld] = None):
        self.world = world or SimulationWorld()

    def get_world_state(self) -> WorldState:
        """Returns WorldState model for decision engine consumption."""
        return self.world.to_decision_world_state()

    def execute_action(self, action: Union[CandidateAction, Dict[str, Any]]) -> ActionExecutionResult:
        """Executes candidate action on local simulation world."""
        return self.world.execute_action(action)

    def inject_disturbance(self, disturbance: DisturbanceEvent) -> bool:
        """Injects dynamic disturbance into the local world."""
        return self.world.inject_disturbance(disturbance)

    def verify_action_result(
        self,
        action: Union[CandidateAction, Dict[str, Any]],
        expected_outcome: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Validates that the last action executed successfully and met physical expectations.
        """
        if not self.world.execution_history:
            return False

        last_res = self.world.execution_history[-1]
        if not last_res.success:
            return False

        if expected_outcome:
            # Check expected carrying object
            if "carrying_object_id" in expected_outcome:
                expected_id = expected_outcome["carrying_object_id"]
                if self.world.robot.carrying_object_id != expected_id:
                    return False

            # Check expected gripper state
            if "gripper_open" in expected_outcome:
                if self.world.robot.gripper_open != expected_outcome["gripper_open"]:
                    return False

            # Check expected robot position proximity
            if "position" in expected_outcome:
                expected_p = expected_outcome["position"]
                dist = self.world.robot.position.distance_to(expected_p)
                tol = expected_outcome.get("tolerance", 0.02)
                if dist > tol:
                    return False

        return True
