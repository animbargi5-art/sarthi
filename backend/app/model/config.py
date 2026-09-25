"""
SĀRTHI Model Integration — Nebius Token Factory Configuration.
Provides validated, immutable configuration for Nebius Token Factory / Nemotron endpoints.
Safeguards against credential leakage and missing parameters.
"""

from dataclasses import dataclass
import os
from typing import Optional


DEFAULT_NEBIUS_BASE_URL: str = "https://api.tokenfactory.nebius.com/v1"
DEFAULT_NEBIUS_TIMEOUT: float = 30.0


@dataclass(frozen=True)
class NebiusConfig:
    """
    Configuration parameters for Nebius Token Factory provider.
    Strictly prevents empty API keys or unspecified model IDs.
    """
    api_key: Optional[str] = None
    base_url: str = DEFAULT_NEBIUS_BASE_URL
    model: Optional[str] = None
    timeout: float = DEFAULT_NEBIUS_TIMEOUT
    max_retries: int = 2

    @classmethod
    def from_env(
        cls,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> "NebiusConfig":
        """
        Loads configuration from environment variables, allowing explicit parameter overrides.
        """
        env_key = api_key or os.environ.get("NEBIUS_API_KEY") or None
        env_base = base_url or os.environ.get("NEBIUS_BASE_URL") or DEFAULT_NEBIUS_BASE_URL
        env_model = model or os.environ.get("NEBIUS_MODEL") or None

        env_timeout_str = os.environ.get("NEBIUS_TIMEOUT")
        if timeout is not None:
            resolved_timeout = float(timeout)
        elif env_timeout_str:
            resolved_timeout = float(env_timeout_str)
        else:
            resolved_timeout = DEFAULT_NEBIUS_TIMEOUT

        return cls(
            api_key=env_key,
            base_url=env_base,
            model=env_model,
            timeout=resolved_timeout,
            max_retries=max_retries if max_retries is not None else 2,
        )

    def validate(self) -> None:
        """
        Validates that required configuration parameters are non-empty.
        Raises NebiusConfigurationError if API key or model is missing.
        """
        from backend.app.model.nebius_provider import NebiusConfigurationError

        if not self.api_key or not self.api_key.strip():
            raise NebiusConfigurationError(
                "NEBIUS_API_KEY is missing or empty. "
                "Set NEBIUS_API_KEY in the environment or provide it explicitly."
            )
        if not self.model or not self.model.strip():
            raise NebiusConfigurationError(
                "NEBIUS_MODEL is missing or empty. "
                "Set NEBIUS_MODEL in the environment or provide it explicitly. "
                "NVIDIA Nemotron model must be explicitly selected."
            )
        if not self.base_url or not self.base_url.strip():
            raise NebiusConfigurationError(
                "NEBIUS_BASE_URL cannot be empty."
            )
        if self.timeout <= 0:
            raise NebiusConfigurationError(
                f"NEBIUS_TIMEOUT must be positive, got {self.timeout}."
            )

    def __repr__(self) -> str:
        masked_key = "[REDACTED]" if self.api_key else "None"
        return (
            f"NebiusConfig(base_url={self.base_url!r}, model={self.model!r}, "
            f"api_key={masked_key}, timeout={self.timeout}, max_retries={self.max_retries})"
        )

    def __str__(self) -> str:
        return self.__repr__()
