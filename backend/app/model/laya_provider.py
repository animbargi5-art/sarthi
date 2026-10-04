"""
SĀRTHI V3-4B — Local Laya Fast Decision Provider Adapter.
Implements the JevDecisionProvider interface against a local Laya server (v1/systemone).
Enforces strict bounded context evaluation, response validation, secret sanitization,
and guarantees that decision recommendations remain strictly non-authoritative.

Non-Negotiable Architecture Invariant:
Laya proposes bounded candidate decisions; SĀRTHI Decision Engine remains the sole physical authority.
Laya must never directly command robot motors or bypass safety validation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import socket
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

from backend.app.model.jev_provider import JevDecisionProvider
from backend.app.model.laya_config import LayaConfig
from backend.app.model.real_jev_provider import (
    JevConfigurationError,
    JevConnectionError,
    JevProviderError,
    JevResponseParsingError,
    JevTimeoutError,
    JevValidationError,
)
from backend.app.model.v3_models import DecisionQuestion, JevDecision

logger = logging.getLogger(__name__)

# Standard criteria descriptions for bounded robotics recovery and decision options
DEFAULT_CRITERIA_MAP: Dict[str, str] = {
    "REPOSITION": "Move to a safer position before attempting the task again.",
    "STOP": "Stop the task because safe progress cannot be established.",
    "REPLAN": "Recompute an alternative transit trajectory avoiding obstacles.",
    "REGRASP": "Adjust gripper grasp position or orientation on the object.",
    "APPROACH": "Navigate end-effector toward target object.",
    "GRASP": "Close gripper to secure target object.",
    "MOVE": "Transit payload toward target destination.",
    "RELEASE": "Open gripper to place object at destination.",
}


class LayaDecisionProvider(JevDecisionProvider):
    """
    Adapter interfacing SĀRTHI with a local Laya decision engine (v1/systemone).
    Builds the bounded typed-decisions request, sends it to local Laya,
    and maps the response into the canonical JevDecision schema.
    """

    def __init__(self, config: Optional[LayaConfig] = None) -> None:
        self._config = config or LayaConfig.from_env()

    @property
    def config(self) -> LayaConfig:
        return self._config

    def is_available(self) -> bool:
        """Returns True if the provider has a configured base URL."""
        return bool(self._config.base_url and self._config.base_url.strip())

    def _sanitize(self, message: str) -> str:
        """Sanitizes text by redacting API keys, Bearer tokens, and secrets."""
        if not message:
            return message
        sanitized = message
        sanitized = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer [REDACTED_TOKEN]", sanitized)
        sanitized = re.sub(r"sk-[A-Za-z0-9_\-\.]+", "[REDACTED_TOKEN]", sanitized)
        sanitized = re.sub(r"nbf_[A-Za-z0-9_\-\.]+", "[REDACTED_TOKEN]", sanitized)
        sanitized = re.sub(
            r"(?i)\b(API_KEY|SECRET|TOKEN|PASSWORD|AUTH)\b\s*[:=]\s*\S+",
            r"\1=[REDACTED]",
            sanitized,
        )
        return sanitized

    def _build_bounded_payload(self, question: DecisionQuestion) -> Dict[str, Any]:
        """
        Constructs the bounded Laya /v1/systemone request body.
        Formats compact state context and discrete options as structured criteria.
        """
        ctx = question.context

        # Assemble compact state description
        state_parts: List[str] = [f"Task Objective: {ctx.task_objective}"]
        if ctx.robot_state_summary:
            holding = ctx.robot_state_summary.get("holding_object_id")
            if holding:
                state_parts.append(f"Robot is holding the object {holding}")
            else:
                state_parts.append("Robot gripper is empty")
        if ctx.active_constraints:
            state_parts.append(f"Active constraints: {'; '.join(ctx.active_constraints)}")
        if ctx.previous_action_outcome:
            state_parts.append(f"Previous action outcome: {ctx.previous_action_outcome}")
        if ctx.disturbance_info:
            anomaly = ctx.disturbance_info.get("type") or ctx.disturbance_info.get("anomaly")
            if anomaly:
                state_parts.append(f"Disturbance detected: {anomaly}")

        state_body = ". ".join(state_parts)

        # Assemble criteria for available options
        criteria: Dict[str, str] = {
            opt: DEFAULT_CRITERIA_MAP.get(opt, f"Consider {opt} as the recovery candidate.")
            for opt in question.available_options
        }

        return {
            "state": {"body": state_body},
            "questions": {
                question.question_id: {
                    "type": "choice",
                    "instructions": question.question_text,
                    "criteria": criteria,
                }
            },
            "model": self._config.model,
        }

    def _parse_response_payload(
        self, response_data: Dict[str, Any], question: DecisionQuestion, latency_ms: float
    ) -> JevDecision:
        """
        Parses and validates local Laya response data against DecisionQuestion and JevDecision schemas.
        """
        q_id = question.question_id

        if "answers" not in response_data or not isinstance(response_data["answers"], dict):
            raise JevResponseParsingError(
                f"Laya response missing 'answers' dict. Available keys: {list(response_data.keys())}"
            )
        answers = response_data["answers"]

        q_data = answers.get(q_id)
        if not q_data or not isinstance(q_data, dict):
            # Fallback for single question scenario
            if len(answers) == 1:
                q_data = next(iter(answers.values()))
            else:
                raise JevResponseParsingError(
                    f"Response missing question result for question_id '{q_id}'. "
                    f"Available answers: {list(answers.keys())}"
                )

        # 1. Extract choice / selected option
        chosen_option = q_data.get("choice") or q_data.get("selected_option")
        if not chosen_option:
            raise JevResponseParsingError(
                f"Laya response for '{q_id}' missing 'choice' field."
            )

        # 2. Bounded Option Validation
        if chosen_option not in question.available_options:
            raise JevValidationError(
                f"Laya selected option '{chosen_option}' which is not in question's "
                f"available options: {question.available_options}"
            )

        # 3. Extract and validate confidence
        confidence = q_data.get("confidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
                if not (0.0 <= confidence <= 1.0):
                    raise JevValidationError(f"Confidence score {confidence} outside [0.0, 1.0]")
            except (ValueError, TypeError) as e:
                raise JevValidationError(f"Invalid confidence value: {confidence}") from e

        # 4. Extract and validate answer_confidence
        answer_confidence = q_data.get("answer_confidence")
        if answer_confidence is not None:
            try:
                answer_confidence = float(answer_confidence)
                if not (0.0 <= answer_confidence <= 1.0):
                    raise JevValidationError(f"Answer confidence score {answer_confidence} outside [0.0, 1.0]")
            except (ValueError, TypeError) as e:
                raise JevValidationError(f"Invalid answer_confidence value: {answer_confidence}") from e

        # 5. Extract and validate probabilities
        raw_probs = q_data.get("probabilities") or q_data.get("option_probabilities") or {}
        option_probs: Dict[str, float] = {}
        if isinstance(raw_probs, dict):
            for opt, val in raw_probs.items():
                try:
                    f_val = float(val)
                    if not (0.0 <= f_val <= 1.0):
                        raise JevValidationError(f"Probability {f_val} for option '{opt}' outside [0.0, 1.0]")
                    option_probs[str(opt)] = round(f_val, 4)
                except (ValueError, TypeError) as e:
                    raise JevValidationError(f"Invalid probability value for option '{opt}': {val}") from e

        model_name = response_data.get("model") or self._config.model

        return JevDecision(
            selected_option=str(chosen_option),
            option_probabilities=option_probs,
            confidence=confidence,
            answer_confidence=answer_confidence,
            rationale=None,
            provider="laya",
            model_version=str(model_name),
            latency_ms=round(latency_ms, 2),
            timestamp_ns=time.time_ns(),
            request_id=q_id,
        )

    def _execute_http_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes HTTP POST to local Laya endpoint with finite timeout and sanitized error handling.
        Does NOT send an Authorization header.
        """
        self._config.validate()

        url = f"{self._config.base_url}/v1/systemone"
        json_bytes = json.dumps(payload).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "SARTHI-Robotics/3.0",
        }

        req = urllib.request.Request(url, data=json_bytes, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self._config.timeout) as resp:
                status_code = resp.status
                body = resp.read().decode("utf-8")

                if status_code != 200:
                    raise JevConnectionError(
                        self._sanitize(f"Laya endpoint returned non-200 HTTP status: {status_code}")
                    )

                try:
                    return json.loads(body)
                except json.JSONDecodeError as jde:
                    raise JevResponseParsingError(
                        self._sanitize(f"Failed to decode Laya JSON response: {jde}. Raw body: {body[:200]}")
                    ) from jde

        except urllib.error.HTTPError as he:
            err_body = ""
            try:
                err_body = he.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            msg = f"HTTP error {he.code}: {he.reason}. Details: {err_body[:200]}"
            raise JevConnectionError(self._sanitize(msg)) from he

        except (urllib.error.URLError, TimeoutError, socket.timeout) as ue:
            msg = str(ue)
            if "timed out" in msg.lower():
                raise JevTimeoutError(
                    self._sanitize(f"Laya request to {url} timed out after {self._config.timeout}s")
                ) from ue
            raise JevConnectionError(
                self._sanitize(f"Failed to connect to Laya endpoint {url}: {msg}")
            ) from ue

        except JevProviderError:
            raise

        except Exception as ex:
            raise JevProviderError(self._sanitize(f"Unexpected Laya communication error: {ex}")) from ex

    def ask(self, question: DecisionQuestion) -> JevDecision:
        """
        Synchronously queries local Laya for a bounded decision.
        """
        if not self.is_available():
            raise JevConfigurationError("Laya provider base_url is not configured.")

        payload = self._build_bounded_payload(question)

        start_time = time.perf_counter()
        raw_response = self._execute_http_request(payload)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return self._parse_response_payload(raw_response, question, elapsed_ms)

    async def ask_async(self, question: DecisionQuestion) -> JevDecision:
        """
        Asynchronously queries local Laya for a bounded decision using thread offload.
        """
        return await asyncio.to_thread(self.ask, question)
