"""
SĀRTHI V3 — Real Jev Fast Decision Provider Adapter.
Implements the JevDecisionProvider interface against the TypeSafe AI / Jev endpoint.
Enforces strict bounded context evaluation, response validation, secret sanitization,
and guarantees that Jev recommendations remain strictly non-authoritative.

Non-Negotiable Architecture Invariant:
Jev proposes bounded candidate decisions; SĀRTHI Decision Engine remains the sole physical authority.
Jev must never directly command robot motors or bypass safety validation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

from backend.app.model.jev_config import JevConfig
from backend.app.model.jev_provider import JevDecisionProvider
from backend.app.model.v3_models import DecisionQuestion, JevDecision

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured Exception Hierarchy
# ---------------------------------------------------------------------------

class JevProviderError(Exception):
    """Base exception for all Jev decision provider errors."""
    pass


class JevConfigurationError(JevProviderError):
    """Raised when Jev configuration (API key, endpoint URL) is missing or invalid."""
    pass


class JevConnectionError(JevProviderError):
    """Raised when network connectivity to Jev endpoint fails."""
    pass


class JevTimeoutError(JevProviderError):
    """Raised when Jev query exceeds the configured latency timeout."""
    pass


class JevResponseParsingError(JevProviderError):
    """Raised when Jev returns a non-JSON or malformed payload."""
    pass


class JevValidationError(JevProviderError):
    """Raised when Jev response fails structural or bounded option validation."""
    pass


# ---------------------------------------------------------------------------
# Real Jev Decision Provider Adapter
# ---------------------------------------------------------------------------

class RealJevDecisionProvider(JevDecisionProvider):
    """
    Production adapter interfacing SĀRTHI with TypeSafe AI's Jev model.
    Encapsulates REST communication, request formatting, timeout enforcement,
    response validation, and secret sanitization.
    """

    def __init__(self, config: Optional[JevConfig] = None) -> None:
        self._config = config or JevConfig.from_env()

    @property
    def config(self) -> JevConfig:
        return self._config

    def is_available(self) -> bool:
        """Returns True if the provider has configured API credentials."""
        return bool(self._config.api_key and self._config.api_key.strip())

    def _sanitize(self, message: str) -> str:
        """Sanitizes text by redacting API keys, Bearer tokens, and secrets."""
        if not message:
            return message
        sanitized = message
        if self._config.api_key:
            sanitized = sanitized.replace(self._config.api_key, "[REDACTED_JEV_KEY]")
        sanitized = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer [REDACTED_TOKEN]", sanitized)
        sanitized = re.sub(r"sk-[A-Za-z0-9_\-\.]+", "[REDACTED_TOKEN]", sanitized)
        sanitized = re.sub(r"nbf_[A-Za-z0-9_\-\.]+", "[REDACTED_TOKEN]", sanitized)
        return sanitized

    def _build_bounded_payload(self, question: DecisionQuestion) -> Dict[str, Any]:
        """
        Extracts only the minimum required bounded decision context to send to Jev.
        Never transmits raw simulation state, meshes, or uncurated arrays.
        """
        ctx = question.context
        compact_state = {
            "task_objective": ctx.task_objective,
            "robot_state": ctx.robot_state_summary,
            "target_object": ctx.target_object_summary,
            "destination": ctx.destination_summary,
            "active_constraints": ctx.active_constraints,
            "previous_action_outcome": ctx.previous_action_outcome,
            "disturbance_info": ctx.disturbance_info,
        }

        # Filter out None values for compact wire footprint
        compact_state = {k: v for k, v in compact_state.items() if v is not None}

        return {
            "state": compact_state,
            "questions": {
                question.question_id: {
                    "type": "choice",
                    "instructions": question.question_text,
                    "options": question.available_options,
                }
            },
        }

    def _parse_response_payload(
        self, response_data: Dict[str, Any], question: DecisionQuestion, latency_ms: float
    ) -> JevDecision:
        """
        Parses and validates Jev response data against DecisionQuestion and JevDecision schemas.
        Supports both canonical TypeSafe AI nested structure and flattened gateway outputs.
        """
        q_id = question.question_id
        q_data: Optional[Dict[str, Any]] = None

        # Check canonical TypeSafe structure: data["questions"][q_id]
        if "questions" in response_data and isinstance(response_data["questions"], dict):
            q_data = response_data["questions"].get(q_id)
        # Check alternative single-question or flat structure
        elif "choice" in response_data or "selected_option" in response_data:
            q_data = response_data
        elif q_id in response_data and isinstance(response_data[q_id], dict):
            q_data = response_data[q_id]

        if not q_data or not isinstance(q_data, dict):
            raise JevResponseParsingError(
                f"Response missing question result for question_id '{q_id}'. "
                f"Available keys: {list(response_data.keys())}"
            )

        # Extract selected option
        chosen_option = q_data.get("choice") or q_data.get("selected_option")
        if not chosen_option:
            raise JevResponseParsingError(
                f"Response for '{q_id}' missing 'choice' or 'selected_option' field."
            )

        # Bounded Option Validation
        if chosen_option not in question.available_options:
            raise JevValidationError(
                f"Jev selected option '{chosen_option}' which is not in question's "
                f"available options: {question.available_options}"
            )

        # Extract confidence
        confidence = q_data.get("confidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
                if not (0.0 <= confidence <= 1.0):
                    raise JevValidationError(f"Confidence score {confidence} outside [0.0, 1.0]")
            except (ValueError, TypeError) as e:
                raise JevValidationError(f"Invalid confidence value: {confidence}") from e

        # Extract and validate probabilities
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

        rationale = q_data.get("rationale") or q_data.get("explanation") or None

        return JevDecision(
            selected_option=str(chosen_option),
            option_probabilities=option_probs,
            confidence=confidence,
            rationale=str(rationale) if rationale else None,
            provider="jev",
            model_version=self._config.model,
            latency_ms=round(latency_ms, 2),
            timestamp_ns=time.time_ns(),
            request_id=q_id,
        )

    def _execute_http_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes HTTP POST to Jev endpoint with finite timeout and sanitized error handling.
        """
        self._config.validate()

        url = f"{self._config.base_url}/systemone"
        json_bytes = json.dumps(payload).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._config.api_key}",
            "User-Agent": "SARTHI-Robotics/3.0",
        }

        req = urllib.request.Request(url, data=json_bytes, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self._config.timeout) as resp:
                status_code = resp.status
                body = resp.read().decode("utf-8")

                if status_code != 200:
                    raise JevConnectionError(
                        self._sanitize(f"Jev endpoint returned non-200 HTTP status: {status_code}")
                    )

                try:
                    return json.loads(body)
                except json.JSONDecodeError as jde:
                    raise JevResponseParsingError(
                        self._sanitize(f"Failed to decode Jev JSON response: {jde}. Raw body: {body[:200]}")
                    ) from jde

        except urllib.error.HTTPError as he:
            err_body = ""
            try:
                err_body = he.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            msg = f"HTTP error {he.code}: {he.reason}. Details: {err_body[:200]}"
            raise JevConnectionError(self._sanitize(msg)) from he

        except (urllib.error.URLError, TimeoutError, socket_timeout) as ue:
            msg = str(ue)
            if "timed out" in msg.lower():
                raise JevTimeoutError(
                    self._sanitize(f"Jev request to {url} timed out after {self._config.timeout}s")
                ) from ue
            raise JevConnectionError(
                self._sanitize(f"Failed to connect to Jev endpoint {url}: {msg}")
            ) from ue

        except JevProviderError:
            raise

        except Exception as ex:
            raise JevProviderError(self._sanitize(f"Unexpected Jev communication error: {ex}")) from ex

    def ask(self, question: DecisionQuestion) -> JevDecision:
        """
        Synchronously queries Jev for a bounded decision.
        Enforces timeout, bounded option validation, and secret sanitization.
        """
        if not self.is_available():
            raise JevConfigurationError(
                "Jev provider is not configured or API key is missing. "
                "Check JEV_API_KEY environment variable."
            )

        payload = self._build_bounded_payload(question)

        start_time = time.perf_counter()
        raw_response = self._execute_http_request(payload)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return self._parse_response_payload(raw_response, question, elapsed_ms)

    async def ask_async(self, question: DecisionQuestion) -> JevDecision:
        """
        Asynchronously queries Jev for a bounded decision using thread offload.
        """
        return await asyncio.to_thread(self.ask, question)


# Handle socket timeout import safely across environments
try:
    from socket import timeout as socket_timeout
except ImportError:
    socket_timeout = TimeoutError
