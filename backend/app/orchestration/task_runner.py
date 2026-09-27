"""
SĀRTHI Orchestration — Task Runner Coordinator.
Executes autonomous task workflows across Decision Engine and Simulation Adapter interfaces.
Never directly controls or commands robot motors.

Phase 4B extension:
  - run_instruction() accepts a natural-language instruction and a ModelProvider,
    interprets it through the model layer, resolves targets against the actual
    WorldState, and delegates execution to the existing closed-loop machinery.
  - Target resolution returns structured failure codes rather than executing
    physical actions if targets are missing.
  - The Decision Engine remains the sole physical-action authority.
"""

import time
from typing import Callable, List, Optional

from backend.app.decision_engine.engine import SarthiDecisionEngine
from backend.app.decision_engine.models import ActionType, WorldState
from backend.app.model.mock_provider import MockModelProvider
from backend.app.model.models import TaskUnderstanding
from backend.app.model.nebius_provider import NebiusNemotronProvider
from backend.app.model.provider import ModelProvider
from backend.app.model.task_understanding import TaskUnderstandingService
from backend.app.orchestration.execution_result import TaskExecutionResult, VerificationSummary
from backend.app.orchestration.loop import (
    ClosedLoopCycle,
    TaskEvent,
    TaskEventType,
    TaskRunResult,
    TaskRunStatus,
    TaskStepResult,
)
from backend.app.orchestration.task_interpreter import TaskInterpreter
from simulation.adapters.base import SimulationAdapter


class SarthiTaskRunner:
    """
    Autonomous closed-loop task runner.
    Coordinates between SĀRTHI Decision Engine and the Simulation Adapter without
    accessing low-level motor primitives or introducing stochastic behaviors.

    Phase 4B additions:
      run_instruction() -- full pipeline entry point from natural language.
    """

    def __init__(
        self,
        adapter: SimulationAdapter,
        engine: Optional[SarthiDecisionEngine] = None,
        max_steps: int = 10,
    ):
        self.adapter = adapter
        self.engine = engine or SarthiDecisionEngine()
        self.max_steps = max_steps
        self.events: List[TaskEvent] = []

    # ------------------------------------------------------------------
    # Completion predicate
    # ------------------------------------------------------------------

    def is_task_completed(self, world_state: WorldState) -> bool:
        """
        Determines whether the primary task has been fully achieved.
        Conditions:
          1. Target object exists and is not currently held.
          2. Target object is situated within target zone tolerance radius.
          3. Robot gripper is open.
          4. Last executed action was RELEASE.
        """
        robot = world_state.robot
        target_zone = world_state.target

        if robot.holding_object_id is not None or not robot.gripper_open:
            return False

        target_obj = None
        for obj in world_state.objects:
            if obj.is_target:
                target_obj = obj
                break

        if target_obj is None:
            return False

        horizontal_dist = (
            (target_obj.position.x - target_zone.position.x) ** 2 +
            (target_obj.position.y - target_zone.position.y) ** 2
        ) ** 0.5

        if horizontal_dist > target_zone.tolerance_radius_m:
            return False

        last_outcome = world_state.last_action_outcome
        if last_outcome.action_type != ActionType.RELEASE or last_outcome.status.value != "SUCCESS":
            return False

        return True

    # ------------------------------------------------------------------
    # Core closed-loop runner
    # ------------------------------------------------------------------

    def run_until_complete(
        self,
        task_id: str = "task_move_red_object",
        step_callback: Optional[Callable[[int, WorldState], None]] = None,
    ) -> TaskRunResult:
        """
        Executes closed-loop control:
        observe -> decide -> execute -> verify -> update
        Repeats until task completed, STOP selected, or max steps reached.
        """
        self.events.clear()
        initial_world_state = self.adapter.get_world_state()
        initial_version = initial_world_state.version
        step_results: List[TaskStepResult] = []

        step_counter = 0

        while step_counter < self.max_steps:
            step_counter += 1

            step_res = ClosedLoopCycle.execute_step(
                step_number=step_counter,
                adapter=self.adapter,
                engine=self.engine,
                events=self.events,
            )
            step_results.append(step_res)

            current_world_state = self.adapter.get_world_state()
            if step_callback:
                step_callback(step_counter, current_world_state)
                current_world_state = self.adapter.get_world_state()

            if step_res.selected_action.action_type == ActionType.STOP:
                self.events.append(TaskEvent(
                    event_type=TaskEventType.TASK_FAILED,
                    step_number=step_counter,
                    timestamp_ns=int(time.time() * 1e9),
                    message="Task halted: safety STOP action engaged",
                    payload={"action_id": step_res.selected_action.action_id}
                ))
                return TaskRunResult(
                    task_id=task_id,
                    initial_world_state_version=initial_version,
                    final_world_state_version=current_world_state.version,
                    status=TaskRunStatus.STOPPED_SAFELY,
                    ordered_step_results=step_results,
                    failure_reason="Safety STOP was selected and executed",
                    events=self.events,
                )

            if self.is_task_completed(current_world_state):
                self.events.append(TaskEvent(
                    event_type=TaskEventType.TASK_COMPLETED,
                    step_number=step_counter,
                    timestamp_ns=int(time.time() * 1e9),
                    message=f"Task '{task_id}' successfully completed in {step_counter} steps",
                    payload={"total_steps": step_counter, "final_version": current_world_state.version}
                ))
                return TaskRunResult(
                    task_id=task_id,
                    initial_world_state_version=initial_version,
                    final_world_state_version=current_world_state.version,
                    status=TaskRunStatus.COMPLETED,
                    ordered_step_results=step_results,
                    failure_reason=None,
                    events=self.events,
                )

        final_state = self.adapter.get_world_state()
        self.events.append(TaskEvent(
            event_type=TaskEventType.TASK_FAILED,
            step_number=step_counter,
            timestamp_ns=int(time.time() * 1e9),
            message=f"Task terminated: maximum step limit ({self.max_steps}) reached without completion",
            payload={"max_steps": self.max_steps}
        ))
        return TaskRunResult(
            task_id=task_id,
            initial_world_state_version=initial_version,
            final_world_state_version=final_state.version,
            status=TaskRunStatus.MAX_STEPS_REACHED,
            ordered_step_results=step_results,
            failure_reason=f"Exceeded maximum step count ({self.max_steps})",
            events=self.events,
        )

    # ------------------------------------------------------------------
    # Phase 4B: natural-language entry point
    # ------------------------------------------------------------------

    def run_instruction(
        self,
        instruction: str,
        model_provider: Optional[ModelProvider] = None,
        step_callback: Optional[Callable[[int, WorldState], None]] = None,
    ) -> TaskExecutionResult:
        """
        Full pipeline entry point from natural language.

        Flow:
            instruction
              |
              v
            TaskUnderstandingService  (model layer -- cognitive only)
              |
              v
            TaskUnderstanding         (validated, frozen)
              |
              v
            Target resolution         (against actual WorldState)
              | (structured failure if targets missing -- NO physical action)
              v
            run_until_complete()      (existing orchestration loop)
              |
              v
            Decision Engine           (sole physical-action authority)
              |
              v
            SimulationAdapter.execute_action()
              |
              v
            Verification
              |
              v
            TaskExecutionResult       (serializable, complete audit)
        """
        provider = model_provider or NebiusNemotronProvider()
        service = TaskUnderstandingService(provider)
        interpreter = TaskInterpreter(service)

        # Step 1: Semantic parsing -- model layer only, no physical action
        understanding = interpreter.interpret(instruction)
        task_id = understanding.task_id

        # Step 2: Target resolution against actual WorldState
        world_state = self.adapter.get_world_state()
        resolution_failure = self._resolve_targets(understanding, world_state)

        if resolution_failure is not None:
            # Structured failure -- ZERO physical actions executed
            current_state = self.adapter.get_world_state()
            return TaskExecutionResult(
                task_id=task_id,
                instruction=instruction,
                task_understanding=understanding,
                completed=False,
                final_world_state_version=current_state.version,
                executed_actions=[],
                failed_actions=[],
                recovery_count=0,
                verification_results=[],
                failure_reason=resolution_failure,
                run_status="FAILED",
            )

        # Step 3: Delegate to existing closed-loop orchestration
        run_result = self.run_until_complete(
            task_id=task_id,
            step_callback=step_callback,
        )

        # Step 4: Build structured TaskExecutionResult from TaskRunResult
        return self._build_execution_result(
            instruction=instruction,
            understanding=understanding,
            run_result=run_result,
        )

    # ------------------------------------------------------------------
    # Target resolution (Phase 4B section 5)
    # ------------------------------------------------------------------

    def _resolve_targets(
        self,
        understanding: TaskUnderstanding,
        world_state: WorldState,
    ) -> Optional[str]:
        """
        Validates that named entities from TaskUnderstanding exist in the WorldState.

        Returns:
            None if all required targets are present.
            Structured failure code string if any target is missing.
            No physical action is ever executed after a non-None return value.
        """
        target_object_label = understanding.target_object
        target_location_label = understanding.target_location

        # Resolve target object -- must have at least one is_target object
        if target_object_label and target_object_label not in ("unspecified_object",):
            has_target_obj = any(obj.is_target for obj in world_state.objects)
            if not has_target_obj:
                return (
                    f"TARGET_OBJECT_NOT_FOUND: '{target_object_label}' "
                    f"not found in current WorldState. No physical action executed."
                )

        # Resolve target location
        if target_location_label and target_location_label not in ("unspecified_location",):
            zone = world_state.target
            zone_id_lower = zone.id.lower().replace("_", "").replace(" ", "")
            label_lower = target_location_label.lower().replace("_", "").replace(" ", "")
            # Accept if zone id contains the label (e.g. 'bluetargetzone' contains 'bluetarget')
            if label_lower not in zone_id_lower and zone_id_lower not in label_lower:
                found_location = any(
                    target_location_label.lower() in obj.id.lower()
                    for obj in world_state.objects
                )
                if not found_location:
                    return (
                        f"TARGET_LOCATION_NOT_FOUND: '{target_location_label}' "
                        f"not found in current WorldState. No physical action executed."
                    )

        return None  # All targets resolved

    # ------------------------------------------------------------------
    # Result builder (Phase 4B section 8)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_execution_result(
        instruction: str,
        understanding: TaskUnderstanding,
        run_result: TaskRunResult,
    ) -> TaskExecutionResult:
        """Converts a TaskRunResult into the richer TaskExecutionResult schema."""
        executed_actions: List[str] = []
        failed_actions: List[str] = []
        recovery_count: int = 0
        verification_results: List[VerificationSummary] = []

        for step in run_result.ordered_step_results:
            action_type_str = step.selected_action.action_type.value
            if step.action_result.success:
                executed_actions.append(action_type_str)
            else:
                failed_actions.append(action_type_str)

            # Count adaptive recovery steps
            if action_type_str == ActionType.REPOSITION.value:
                recovery_count += 1

            verification_results.append(VerificationSummary(
                step_number=step.step_number,
                action_type=action_type_str,
                verified=step.verification_result,
            ))

        completed = (run_result.status == TaskRunStatus.COMPLETED)

        return TaskExecutionResult(
            task_id=understanding.task_id,
            instruction=instruction,
            task_understanding=understanding,
            completed=completed,
            final_world_state_version=run_result.final_world_state_version,
            executed_actions=executed_actions,
            failed_actions=failed_actions,
            recovery_count=recovery_count,
            verification_results=verification_results,
            failure_reason=run_result.failure_reason,
            run_status=run_result.status.value,
        )
