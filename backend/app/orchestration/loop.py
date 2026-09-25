"""
SĀRTHI Orchestration Loop — Closed-Loop Step Execution and Event Logging.
Implements the canonical observe → decide → execute → verify → update control cycle.
"""

import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from backend.app.decision_engine.engine import SarthiDecisionEngine
from backend.app.decision_engine.models import CandidateAction, Decision, WorldState
from simulation.adapters.base import SimulationAdapter
from simulation.core.events import ActionExecutionResult


class TaskEventType(str, Enum):
    """Structured orchestration event categories."""
    WORLD_OBSERVED = "WORLD_OBSERVED"
    DECISION_MADE = "DECISION_MADE"
    ACTION_EXECUTED = "ACTION_EXECUTED"
    ACTION_VERIFIED = "ACTION_VERIFIED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_FAILED = "TASK_FAILED"


class TaskRunStatus(str, Enum):
    """High-level terminal status of a task execution run."""
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    MAX_STEPS_REACHED = "MAX_STEPS_REACHED"
    STOPPED_SAFELY = "STOPPED_SAFELY"


class TaskEvent(BaseModel):
    """Structured audit log entry emitted during orchestration."""
    model_config = ConfigDict(frozen=True)

    event_type: TaskEventType = Field(..., description="Category of orchestration event")
    step_number: int = Field(..., description="Active step counter")
    timestamp_ns: int = Field(..., description="Timestamp in nanoseconds")
    message: str = Field(..., description="Summary message describing the event")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Structured contextual metadata")


class TaskStepResult(BaseModel):
    """Detailed audit record of a single closed-loop control step."""
    model_config = ConfigDict(frozen=True)

    step_number: int = Field(..., description="1-indexed step number")
    world_state_version_before: Union[int, str] = Field(..., description="World state version prior to action")
    world_state_version_after: Union[int, str] = Field(..., description="World state version following action")
    selected_action: CandidateAction = Field(..., description="Action primitive chosen by the decision engine")
    action_result: ActionExecutionResult = Field(..., description="Result returned by the simulation adapter")
    verification_result: bool = Field(..., description="Verification result confirming physical expectation")
    decision: Optional[Decision] = Field(default=None, description="Complete decision artifact from engine")


class TaskRunResult(BaseModel):
    """Comprehensive summary of an entire end-to-end task run."""
    model_config = ConfigDict(frozen=True)

    task_id: str = Field(..., description="Identifier of the executed task")
    initial_world_state_version: Union[int, str] = Field(..., description="Initial world state version at start")
    final_world_state_version: Union[int, str] = Field(..., description="Final world state version at end")
    status: TaskRunStatus = Field(..., description="Terminal task status")
    ordered_step_results: List[TaskStepResult] = Field(
        default_factory=list,
        description="Chronological step execution results"
    )
    failure_reason: Optional[str] = Field(default=None, description="Failure diagnostic message if task did not complete")
    events: List[TaskEvent] = Field(default_factory=list, description="Ordered audit log of orchestration events")


class ClosedLoopCycle:
    """
    Executes a single atomic closed-loop step:
    observe → decide → execute → verify → update
    Guarantees no direct motor command access.
    """

    @staticmethod
    def execute_step(
        step_number: int,
        adapter: SimulationAdapter,
        engine: SarthiDecisionEngine,
        events: List[TaskEvent],
    ) -> TaskStepResult:
        """
        Executes one full closed-loop step and emits structured audit events.
        """
        # 1. OBSERVE
        world_state_before = adapter.get_world_state()
        version_before = world_state_before.version
        now_ns = int(time.time() * 1e9)
        events.append(TaskEvent(
            event_type=TaskEventType.WORLD_OBSERVED,
            step_number=step_number,
            timestamp_ns=now_ns,
            message=f"Step {step_number}: World state observed (v{version_before})",
            payload={
                "version": version_before,
                "robot_position": {"x": world_state_before.robot.position.x, "y": world_state_before.robot.position.y, "z": world_state_before.robot.position.z},
                "gripper_open": world_state_before.robot.gripper_open,
                "holding_object_id": world_state_before.robot.holding_object_id,
            }
        ))

        # 2. DECIDE
        decision = engine.decide(world_state_before, decision_id=f"dec_step_{step_number}")
        selected_action = decision.selected_action
        events.append(TaskEvent(
            event_type=TaskEventType.DECISION_MADE,
            step_number=step_number,
            timestamp_ns=int(time.time() * 1e9),
            message=f"Step {step_number}: Selected action {selected_action.action_type.value} ({selected_action.action_id})",
            payload={
                "action_type": selected_action.action_type.value,
                "action_id": selected_action.action_id,
                "overall_score": decision.decision_factors.get("overall_score", 0.0),
                "rejection_reasons": decision.rejection_reasons,
            }
        ))

        # 3. EXECUTE
        action_res = adapter.execute_action(selected_action)
        events.append(TaskEvent(
            event_type=TaskEventType.ACTION_EXECUTED,
            step_number=step_number,
            timestamp_ns=int(time.time() * 1e9),
            message=f"Step {step_number}: Action {selected_action.action_type.value} executed (success={action_res.success})",
            payload={
                "success": action_res.success,
                "failure_reason": action_res.failure_reason,
                "previous_version": action_res.previous_world_state_version,
                "new_version": action_res.new_world_state_version,
            }
        ))

        # 4. VERIFY
        # Build expectation based on action type
        expected_outcome: Dict[str, Any] = {}
        if selected_action.action_type.value == "GRASP":
            expected_outcome["gripper_open"] = False
            if selected_action.target_object_id:
                expected_outcome["carrying_object_id"] = selected_action.target_object_id
        elif selected_action.action_type.value == "RELEASE":
            expected_outcome["gripper_open"] = True
            expected_outcome["carrying_object_id"] = None

        verification_passed = adapter.verify_action_result(selected_action, expected_outcome if expected_outcome else None)
        events.append(TaskEvent(
            event_type=TaskEventType.ACTION_VERIFIED,
            step_number=step_number,
            timestamp_ns=int(time.time() * 1e9),
            message=f"Step {step_number}: Action verification passed: {verification_passed}",
            payload={"verified": verification_passed}
        ))

        # 5. UPDATE
        world_state_after = adapter.get_world_state()
        version_after = world_state_after.version

        return TaskStepResult(
            step_number=step_number,
            world_state_version_before=version_before,
            world_state_version_after=version_after,
            selected_action=selected_action,
            action_result=action_res,
            verification_result=verification_passed,
            decision=decision,
        )
