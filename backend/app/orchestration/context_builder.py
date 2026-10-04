"""
SĀRTHI V3 — Deterministic Decision Context Builder.
Constructs compact, bounded DecisionContext payloads for fast decision models (Jev).
Enforces context minimization, deterministic serialization, and strict secret isolation.

Non-Negotiable Architecture Invariants:
1. AI models propose or select bounded semantic candidates; deterministic software validates
   physical feasibility; only validated actions reach the robot controller.
2. Jev must never bypass deterministic safety validation.
3. DecisionContext contains only minimal decision-critical fields; never exposes raw simulator
   internals, meshes, or credentials.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

from backend.app.decision_engine.models import (
    ActionType,
    ActiveConstraint,
    CandidateAction,
    LastActionOutcome,
    LastActionStatus,
    WorldObject,
    WorldState,
)
from backend.app.model.models import TaskUnderstanding
from backend.app.model.v3_models import (
    DecisionContext,
    HumanInstruction,
    PhysicalSituation,
)


FORBIDDEN_SECRET_PATTERNS = ["sk-", "nbf_", "Bearer ", "API_KEY", "SECRET", "TOKEN", "PASSWORD"]


def _sanitize_string(text: Optional[str]) -> Optional[str]:
    """Redacts any secret or credential substring found within a string."""
    if not text:
        return text
    sanitized = text
    for pattern in FORBIDDEN_SECRET_PATTERNS:
        if pattern.lower() in sanitized.lower():
            sanitized = re.sub(rf"{re.escape(pattern)}[A-Za-z0-9_\-\.]*", "[REDACTED]", sanitized, flags=re.IGNORECASE)
    # Also check generic key-value credential expressions (e.g. API_KEY=xyz, token: abc)
    sanitized = re.sub(
        r"(?i)\b(API_KEY|SECRET|TOKEN|PASSWORD|AUTH)\b\s*[:=]\s*\S+",
        r"\1=[REDACTED]",
        sanitized,
    )
    return sanitized


def _sanitize_dict(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Recursively redacts any secret keys or token values."""
    if not data:
        return data
    sanitized: Dict[str, Any] = {}
    for k, v in data.items():
        k_upper = str(k).upper()
        if any(f in k_upper for f in ["KEY", "SECRET", "TOKEN", "PASSWORD", "AUTH"]):
            sanitized[k] = "[REDACTED]"
            continue
        if isinstance(v, dict):
            sanitized[k] = _sanitize_dict(v)
        elif isinstance(v, str):
            if any(f.lower() in v.lower() for f in FORBIDDEN_SECRET_PATTERNS):
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = _sanitize_string(v)
        elif isinstance(v, list):
            sanitized[k] = [
                _sanitize_dict(item) if isinstance(item, dict)
                else ("[REDACTED]" if isinstance(item, str) and any(f.lower() in item.lower() for f in FORBIDDEN_SECRET_PATTERNS) else item)
                for item in v
            ]
        else:
            sanitized[k] = v
    return sanitized


class DecisionContextBuilder:
    """
    Deterministic builder converting canonical SĀRTHI state into compact DecisionContexts.
    Guarantees strict context minimization and zero leakage of simulation internals or secrets.
    """

    def __init__(self, coordinate_precision: int = 3) -> None:
        """
        Args:
            coordinate_precision: Decimal places for rounding Cartesian 3D coordinates.
        """
        self._precision = coordinate_precision

    def _round_point(self, x: float, y: float, z: float) -> List[float]:
        """Rounds 3D coordinates to configured decimal precision."""
        return [
            round(float(x), self._precision),
            round(float(y), self._precision),
            round(float(z), self._precision),
        ]

    def build_context(
        self,
        world_state: Optional[WorldState] = None,
        task_objective: Optional[str] = None,
        candidate_actions: Optional[List[Any]] = None,
        context_id: Optional[str] = None,
        *args,
        **kwargs,
    ) -> DecisionContext:
        """Convenience alias for build()."""
        import time
        cid = context_id or kwargs.pop("context_id", None) or f"ctx_{int(time.time() * 1000) % 1000000}"
        ws = world_state if world_state is not None else kwargs.pop("world_state", None)
        obj = task_objective if task_objective is not None else kwargs.pop("task_objective", None)
        cands = candidate_actions if candidate_actions is not None else kwargs.pop("candidate_actions", None)
        return self.build(
            context_id=cid,
            world_state=ws,
            task_objective=obj,
            candidate_actions=cands,
            **kwargs,
        )

    def build(
        self,
        *,
        context_id: str,
        world_state: Optional[WorldState] = None,
        situation: Optional[PhysicalSituation] = None,
        instruction: Optional[Union[HumanInstruction, str]] = None,
        task_understanding: Optional[TaskUnderstanding] = None,
        candidate_actions: Optional[List[Union[CandidateAction, ActionType, str]]] = None,
        disturbance_info: Optional[Dict[str, Any]] = None,
        task_objective: Optional[str] = None,
        active_constraints: Optional[List[ActiveConstraint]] = None,
        last_action_outcome: Optional[LastActionOutcome] = None,
        timestamp_ns: Optional[int] = None,
    ) -> DecisionContext:
        """
        Deterministically synthesizes a DecisionContext from provided state components.

        Resolves state from either:
        - PhysicalSituation (canonical wrapper), OR
        - Direct WorldState + auxiliary parameters.
        """
        # 1. Resolve canonical WorldState
        ws: Optional[WorldState] = None
        if situation is not None:
            ws = situation.world_state
        elif world_state is not None:
            ws = world_state
        else:
            raise ValueError("Either 'situation' or 'world_state' must be provided to DecisionContextBuilder.")

        # 2. Resolve Task Objective
        resolved_objective: str = "PICK_AND_PLACE"
        if task_objective is not None and task_objective.strip():
            resolved_objective = task_objective.strip()
        elif task_understanding is not None and task_understanding.objective and task_understanding.objective.strip():
            resolved_objective = task_understanding.objective.strip()
        elif instruction is not None:
            if isinstance(instruction, HumanInstruction) and instruction.instruction.strip():
                resolved_objective = instruction.instruction.strip()
            elif isinstance(instruction, str) and instruction.strip():
                resolved_objective = instruction.strip()
        elif hasattr(ws, "task_objective") and ws.task_objective:
            val = ws.task_objective
            resolved_objective = val.value if hasattr(val, "value") else str(val)

        resolved_objective = _sanitize_string(resolved_objective) or "PICK_AND_PLACE"

        # 3. Extract Robot State Summary (Decision-Critical Only)
        robot_summary: Dict[str, Any] = {
            "position": self._round_point(ws.robot.position.x, ws.robot.position.y, ws.robot.position.z),
            "gripper_open": bool(ws.robot.gripper_open),
            "holding_object_id": _sanitize_string(ws.robot.holding_object_id),
            "payload_mass_kg": round(float(ws.robot.payload_mass_kg), 3),
        }

        # 4. Extract Target Object Summary
        target_entity_id = (
            task_understanding.target_object if task_understanding and task_understanding.target_object else None
        )
        target_obj: Optional[WorldObject] = None

        if target_entity_id:
            target_obj = next((obj for obj in ws.objects if obj.id == target_entity_id or obj.name == target_entity_id), None)
        if target_obj is None:
            target_obj = next((obj for obj in ws.objects if obj.is_target), None)

        target_summary: Optional[Dict[str, Any]] = None
        if target_obj:
            state_val = target_obj.state.value if hasattr(target_obj.state, "value") else str(target_obj.state)
            target_summary = {
                "id": _sanitize_string(target_obj.id),
                "name": _sanitize_string(target_obj.name),
                "position": self._round_point(target_obj.position.x, target_obj.position.y, target_obj.position.z),
                "state": state_val,
            }

        # 5. Extract Destination Zone Summary
        dest_summary: Optional[Dict[str, Any]] = None
        if ws.target is not None:
            dest_summary = {
                "id": _sanitize_string(ws.target.id),
                "position": self._round_point(ws.target.position.x, ws.target.position.y, ws.target.position.z),
                "tolerance_radius_m": round(float(ws.target.tolerance_radius_m), 4),
            }

        # 6. Extract & Sort Active Constraints (Deterministic Order & Deduplication)
        raw_constraints: List[ActiveConstraint] = []
        if active_constraints is not None:
            raw_constraints = active_constraints
        elif situation is not None and situation.active_constraints:
            raw_constraints = situation.active_constraints
        elif ws.active_constraints:
            raw_constraints = ws.active_constraints

        # Sort and deduplicate constraint descriptions deterministically with secret sanitization
        unique_descriptions = {
            _sanitize_string(c.description)
            for c in raw_constraints
            if c.description and _sanitize_string(c.description)
        }
        constraint_summaries: List[str] = sorted(list(unique_descriptions))

        # 7. Extract Previous Action Outcome
        outcome_obj: Optional[LastActionOutcome] = None
        if last_action_outcome is not None:
            outcome_obj = last_action_outcome
        elif situation is not None:
            outcome_obj = situation.last_action_outcome
        elif hasattr(ws, "last_action_outcome"):
            outcome_obj = ws.last_action_outcome

        prev_outcome_summary: Optional[str] = None
        if outcome_obj and outcome_obj.status != LastActionStatus.NONE:
            act_str = ""
            if outcome_obj.action_type:
                act_str = outcome_obj.action_type.value if hasattr(outcome_obj.action_type, "value") else str(outcome_obj.action_type)
            status_str = outcome_obj.status.value if hasattr(outcome_obj.status, "value") else str(outcome_obj.status)

            if outcome_obj.error_message:
                clean_err = _sanitize_string(outcome_obj.error_message)
                prev_outcome_summary = f"{act_str}:{status_str} ({clean_err})" if act_str else f"{status_str} ({clean_err})"
            else:
                prev_outcome_summary = f"{act_str}:{status_str}" if act_str else status_str

        # 8. Normalize Candidate Actions
        normalized_candidates: List[str] = []
        if candidate_actions:
            for item in candidate_actions:
                if isinstance(item, CandidateAction):
                    normalized_candidates.append(item.action_type.value)
                elif isinstance(item, ActionType):
                    normalized_candidates.append(item.value)
                elif isinstance(item, str):
                    normalized_candidates.append(item.strip().upper())
                else:
                    normalized_candidates.append(str(item).upper())
        # Deduplicate while preserving order, and sanitize
        seen = set()
        deduped_candidates = [
            _sanitize_string(x)
            for x in normalized_candidates
            if not (x in seen or seen.add(x)) and _sanitize_string(x)
        ]

        # 9. Sanitize Disturbance Info & Timestamp
        clean_disturbance = _sanitize_dict(disturbance_info)

        resolved_timestamp = 0
        if timestamp_ns is not None:
            resolved_timestamp = int(timestamp_ns)
        elif situation is not None and situation.timestamp_ns:
            resolved_timestamp = int(situation.timestamp_ns)
        elif hasattr(ws, "timestamp_ns") and ws.timestamp_ns:
            resolved_timestamp = int(ws.timestamp_ns)

        return DecisionContext(
            context_id=_sanitize_string(context_id) or "ctx_unknown",
            task_objective=resolved_objective,
            robot_state_summary=robot_summary,
            target_object_summary=target_summary,
            destination_summary=dest_summary,
            active_constraints=constraint_summaries,
            previous_action_outcome=prev_outcome_summary,
            candidate_actions=deduped_candidates,
            disturbance_info=clean_disturbance,
            timestamp_ns=resolved_timestamp,
        )
