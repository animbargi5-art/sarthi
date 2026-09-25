"""
SĀRTHI Orchestration Package.
Coordinates closed-loop execution between the Decision Engine and Simulation Adapters.
Phase 4B: natural-language entry point via run_instruction(), TaskInterpreter,
and structured TaskExecutionResult.
"""

from backend.app.orchestration.loop import (
    ClosedLoopCycle,
    TaskEvent,
    TaskEventType,
    TaskRunResult,
    TaskRunStatus,
    TaskStepResult,
)
from backend.app.orchestration.task_runner import SarthiTaskRunner
from backend.app.orchestration.task_interpreter import TaskInterpreter
from backend.app.orchestration.execution_result import TaskExecutionResult, VerificationSummary

__all__ = [
    "ClosedLoopCycle",
    "TaskEvent",
    "TaskEventType",
    "TaskRunResult",
    "TaskRunStatus",
    "TaskStepResult",
    "SarthiTaskRunner",
    "TaskInterpreter",
    "TaskExecutionResult",
    "VerificationSummary",
]
