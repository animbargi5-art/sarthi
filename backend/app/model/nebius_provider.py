"""
SĀRTHI Model Integration — Nebius Token Factory Nemotron Provider Adapter.
Provides production-ready, structured task understanding via NVIDIA Nemotron
hosted on Nebius Token Factory (OpenAI-compatible inference endpoint).

Strict cognitive boundary:
- Never executes robot actions
- Never modifies WorldState
- Never calls SimulationAdapter
- Never bypasses Decision Engine
- Never selects physically executable actions
- Never bypasses safety constraints
"""

import json
import logging
import re
from typing import Any, Dict, Optional

import openai
from pydantic import ValidationError

from backend.app.model.config import (
    DEFAULT_NEBIUS_BASE_URL,
    DEFAULT_NEBIUS_TIMEOUT,
    NebiusConfig,
)
from backend.app.model.models import TaskUnderstanding
from backend.app.model.provider import ModelProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured Provider Exceptions
# ---------------------------------------------------------------------------

class NebiusProviderError(Exception):
    """Base exception for all Nebius Nemotron provider errors."""
    pass


class NebiusConfigurationError(NebiusProviderError):
    """Raised when provider configuration (API key, model, URL) is missing or invalid."""
    pass


class NebiusAPIError(NebiusProviderError):
    """Raised when an API error or non-200 HTTP status is returned by Nebius."""
    pass


class NebiusNetworkError(NebiusProviderError):
    """Raised when a network transport failure occurs communicating with Nebius."""
    pass


class NebiusTimeoutError(NebiusProviderError):
    """Raised when an API request to Nebius times out."""
    pass


class NebiusParsingError(NebiusProviderError):
    """Raised when the model response cannot be parsed into valid JSON."""
    pass


class NebiusValidationError(NebiusProviderError):
    """Raised when parsed JSON fails TaskUnderstanding Pydantic schema validation."""
    pass


# ---------------------------------------------------------------------------
# Security & Sanitization Helpers
# ---------------------------------------------------------------------------

def _sanitize_error_message(message: str, api_key: Optional[str] = None) -> str:
    """
    Ensures that API keys and authentication tokens are never exposed in error
    messages, traces, or log outputs.
    """
    if not message:
        return ""
    sanitized = message
    if api_key and len(api_key.strip()) > 3:
        sanitized = sanitized.replace(api_key.strip(), "[REDACTED_API_KEY]")
    # Mask standard bearer tokens and key patterns
    sanitized = re.sub(r"Bearer\s+([A-Za-z0-9_\-\.]+)", "Bearer [REDACTED]", sanitized)
    sanitized = re.sub(
        r"(api[_-]?key[\"'\s:=]+)([A-Za-z0-9_\-\.]{6,})",
        r"\1[REDACTED]",
        sanitized,
        flags=re.IGNORECASE,
    )
    return sanitized


# ---------------------------------------------------------------------------
# Structured System Instruction & Prompt Formulation
# ---------------------------------------------------------------------------

NEMOTRON_SYSTEM_PROMPT = """You are SĀRTHI's cognitive task understanding engine powered by NVIDIA Nemotron.
Your role is purely cognitive: you interpret human natural language instructions and output a structured, schema-compliant JSON specification.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. You do NOT execute robot actions.
2. You do NOT modify world state.
3. You do NOT control simulation or physical motors.
4. Your 'required_actions' are high-level semantic intentions only (e.g. 'APPROACH', 'GRASP', 'MOVE', 'RELEASE'), NOT executable physical commands.
5. All physical action selection, kinematics, obstacle avoidance, and safety constraints are strictly evaluated and governed by the SĀRTHI Decision Engine.

You must respond with ONLY valid JSON (no markdown explanation or text outside the JSON object).
The JSON object must strictly adhere to the following schema:
{
  "task_id": "string (unique task identifier, e.g. task_<hash_or_id>)",
  "objective": "string (clear normalized high-level mission objective)",
  "target_object": "string or null (identified manipulable target entity ID or label, e.g. 'red_object')",
  "target_location": "string or null (identified destination zone ID or label, e.g. 'blue_target')",
  "required_actions": ["string"] (symbolic high-level semantic actions, e.g. ["APPROACH", "GRASP", "MOVE", "RELEASE"]),
  "constraints": ["string"] (operational or safety constraints, e.g. ["avoid_collisions", "respect_workspace_limits"]),
  "success_conditions": ["string"] (verifiable completion conditions, e.g. ["red_object_at_blue_target", "gripper_released"]),
  "confidence": float between 0.0 and 1.0 (self-assessed confidence score),
  "reasoning_summary": "string (concise semantic explanation of entity grounding and intent)"
}
"""


def _format_user_prompt(
    instruction: str,
    world_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Formats the human instruction and optional contextual metadata for Nemotron."""
    prompt = f"Human Instruction: \"{instruction}\""
    if world_context:
        prompt += f"\nWorld Context: {json.dumps(world_context, default=str)}"
    prompt += "\n\nProvide the TaskUnderstanding JSON object now."
    return prompt


def _extract_json(raw_text: str) -> Dict[str, Any]:
    """
    Extracts and parses JSON from raw model output.
    Handles:
      1. Pure JSON strings.
      2. Markdown fenced JSON: ```json ... ``` or ``` ... ```.
    Rejects malformed JSON and non-dict JSON output with NebiusParsingError.
    """
    if not raw_text or not raw_text.strip():
        raise NebiusParsingError("Model returned an empty response.")

    text = raw_text.strip()

    # Detect and unwrap markdown code fences if present
    fence_pattern = r"^```(?:json)?\s*([\s\S]*?)\s*```$"
    fence_match = re.match(fence_pattern, text, flags=re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()
    elif text.startswith("```"):
        fence_search = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
        if fence_search:
            text = fence_search.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise NebiusParsingError(
            f"Failed to parse model response as JSON: {exc.msg} (line {exc.lineno}, col {exc.colno})"
        ) from None

    if not isinstance(data, dict):
        raise NebiusParsingError(
            f"Expected JSON object (dict) from model, got {type(data).__name__}."
        )

    return data


# ---------------------------------------------------------------------------
# Nebius Nemotron Provider Implementation
# ---------------------------------------------------------------------------

class NebiusNemotronProvider(ModelProvider):
    """
    Production-ready Nebius Token Factory model provider adapter for NVIDIA Nemotron.
    Implements the SĀRTHI ModelProvider interface.

    Strict cognitive boundary:
    - Never executes robot actions
    - Never modifies WorldState
    - Never calls SimulationAdapter
    - Never bypasses Decision Engine
    - Never selects physically executable actions
    - Never bypasses safety constraints
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        config: Optional[NebiusConfig] = None,
        client: Optional[Any] = None,
    ):
        """
        Initializes the Nebius Nemotron provider.

        Args:
            api_key: Nebius Token Factory API key (defaults to NEBIUS_API_KEY env var).
            base_url: Nebius endpoint URL (defaults to NEBIUS_BASE_URL env var or default).
            model: NVIDIA Nemotron model ID (defaults to NEBIUS_MODEL env var).
            timeout: Request timeout in seconds (defaults to NEBIUS_TIMEOUT or 30.0).
            config: Optional pre-constructed NebiusConfig instance.
            client: Optional pre-configured OpenAI-compatible client (for mocking/testing).
        """
        if config is not None:
            self._config = config
        else:
            self._config = NebiusConfig.from_env(
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout=timeout,
            )

        # Enforce strict validation: key and model must be present
        self._config.validate()

        if client is not None:
            self._client = client
        else:
            self._client = openai.OpenAI(
                api_key=self._config.api_key,
                base_url=self._config.base_url,
                timeout=self._config.timeout,
            )

    @property
    def model_name(self) -> Optional[str]:
        """Configured NVIDIA Nemotron model ID."""
        return self._config.model

    @property
    def base_url(self) -> str:
        """Configured Nebius Token Factory base URL."""
        return self._config.base_url

    @property
    def config(self) -> NebiusConfig:
        """Read-only access to provider configuration."""
        return self._config

    def understand_task(
        self,
        instruction: str,
        world_context: Optional[Dict[str, Any]] = None,
    ) -> TaskUnderstanding:
        """
        Queries NVIDIA Nemotron via Nebius Token Factory to extract structured TaskUnderstanding.

        Args:
            instruction: Human natural language instruction.
            world_context: Optional dictionary of world state perception / entity context.

        Returns:
            Validated, immutable TaskUnderstanding instance.

        Raises:
            ValueError: If instruction is empty.
            NebiusTimeoutError: On request timeout.
            NebiusNetworkError: On connection/network transport failure.
            NebiusAPIError: On non-200 / API error from Nebius.
            NebiusParsingError: If model response is not valid JSON.
            NebiusValidationError: If model output does not conform to TaskUnderstanding schema.
            NebiusProviderError: For any other provider-level failures.
        """
        if not instruction or not instruction.strip():
            raise ValueError("Instruction string must be non-empty.")

        messages = [
            {"role": "system", "content": NEMOTRON_SYSTEM_PROMPT},
            {"role": "user", "content": _format_user_prompt(instruction.strip(), world_context)},
        ]

        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                temperature=0.0,
            )
        except openai.APITimeoutError as exc:
            msg = _sanitize_error_message(str(exc), self._config.api_key)
            raise NebiusTimeoutError(f"Nebius request timed out: {msg}") from None
        except openai.APIConnectionError as exc:
            msg = _sanitize_error_message(str(exc), self._config.api_key)
            raise NebiusNetworkError(f"Nebius network connection failed: {msg}") from None
        except openai.APIStatusError as exc:
            msg = _sanitize_error_message(str(exc), self._config.api_key)
            raise NebiusAPIError(f"Nebius API error (status {exc.status_code}): {msg}") from None
        except openai.APIError as exc:
            msg = _sanitize_error_message(str(exc), self._config.api_key)
            raise NebiusAPIError(f"Nebius API error: {msg}") from None
        except Exception as exc:
            msg = _sanitize_error_message(str(exc), self._config.api_key)
            raise NebiusProviderError(f"Nebius provider invocation failed: {msg}") from None

        if not response.choices:
            raise NebiusProviderError("Nebius response contained no choices.")

        choice = response.choices[0]
        raw_content = choice.message.content if choice.message else None
        if not raw_content:
            raise NebiusProviderError("Nebius response contained empty message content.")

        parsed_dict = _extract_json(raw_content)

        try:
            return TaskUnderstanding.model_validate(parsed_dict)
        except ValidationError as exc:
            msg = _sanitize_error_message(str(exc), self._config.api_key)
            raise NebiusValidationError(
                f"Model response failed TaskUnderstanding schema validation: {msg}"
            ) from None

    def __repr__(self) -> str:
        return f"NebiusNemotronProvider(model={self.model_name!r}, base_url={self.base_url!r})"

    def __str__(self) -> str:
        return self.__repr__()
