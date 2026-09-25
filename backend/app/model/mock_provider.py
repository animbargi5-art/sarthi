"""
SĀRTHI Model Integration — Mock Model Provider.
Provides deterministic, offline task understanding for local testing and CI/CD pipelines
without requiring active cloud credentials or network connectivity.
"""

import hashlib
import re
from typing import Any, Dict, Optional
from backend.app.model.models import TaskUnderstanding
from backend.app.model.provider import ModelProvider


class MockModelProvider(ModelProvider):
    """
    Deterministic mock provider simulating NVIDIA Nemotron task understanding.
    Used for local development and unit tests when Nebius API access is unavailable.
    """

    def understand_task(
        self,
        instruction: str,
        world_context: Optional[Dict[str, Any]] = None,
    ) -> TaskUnderstanding:
        """
        Parses instruction and returns deterministic TaskUnderstanding.
        Specifically maps canonical instruction:
        'Move the red object to the blue target.'
        to target_object='red_object', target_location='blue_target',
        and required_actions=['APPROACH', 'GRASP', 'MOVE', 'RELEASE'].
        """
        norm_inst = instruction.strip()
        lower_inst = norm_inst.lower()

        # Deterministic task ID from instruction hash
        task_hash = hashlib.sha256(norm_inst.encode("utf-8")).hexdigest()[:8]
        task_id = f"task_{task_hash}"

        # Standard canonical test instruction: "Move the red object to the blue target."
        if "red object" in lower_inst and "blue target" in lower_inst:
            return TaskUnderstanding(
                task_id=task_id,
                objective="Move the red object to the blue target.",
                target_object="red_object",
                target_location="blue_target",
                required_actions=[
                    "APPROACH",
                    "GRASP",
                    "MOVE",
                    "RELEASE",
                ],
                constraints=[
                    "maintain_gripper_clearance",
                    "avoid_workspace_boundary_breach",
                    "respect_payload_limits",
                ],
                success_conditions=[
                    "red_object_at_blue_target",
                    "gripper_released",
                ],
                confidence=0.98,
                reasoning_summary=(
                    "Identified pick-and-place task: ground entity 'red_object' as manipulation target, "
                    "ground location 'blue_target' as destination zone. Planned standard 4-phase sequence."
                ),
            )

        # Generalized pattern: "move the <object> to the <target>"
        match = re.search(r"move the ([\w\s]+) to the ([\w\s]+)", lower_inst)
        if match:
            obj_name = match.group(1).replace(" ", "_").strip()
            loc_name = match.group(2).replace(" ", "_").strip()
            return TaskUnderstanding(
                task_id=task_id,
                objective=norm_inst,
                target_object=obj_name,
                target_location=loc_name,
                required_actions=[
                    "APPROACH",
                    "GRASP",
                    "MOVE",
                    "RELEASE",
                ],
                constraints=["avoid_collisions", "maintain_stability"],
                success_conditions=[f"{obj_name}_at_{loc_name}", "gripper_released"],
                confidence=0.90,
                reasoning_summary=f"Extracted object '{obj_name}' and target location '{loc_name}' from instruction.",
            )

        # Fallback generic task understanding
        return TaskUnderstanding(
            task_id=task_id,
            objective=norm_inst,
            target_object="unspecified_object",
            target_location="unspecified_location",
            required_actions=["APPROACH", "MOVE"],
            constraints=["general_safety_hold"],
            success_conditions=["task_completed"],
            confidence=0.75,
            reasoning_summary="Generic instruction parsed with fallback heuristics.",
        )
