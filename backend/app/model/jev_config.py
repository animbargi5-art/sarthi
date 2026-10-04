"""
SĀRTHI V3 — Jev Fast Decision Provider Configuration.
Provides validated, immutable configuration for TypeSafe AI / Jev endpoints.
Enforces strict credential isolation and safeguards against secret leakage.
"""

from dataclasses import dataclass
import os
from typing import Optional


DEFAULT_JEV_BASE_URL: str = "https://api.typesafe.ai/v1"
DEFAULT_JEV_TIMEOUT: float = 5.0
DEFAULT_JEV_MODEL: str = "jev-1"


@dataclass(frozen=True)
class JevConfig:
    """
    Configuration parameters for Jev fast decision provider.
    Strictly prevents hard-coded secrets and sanitizes runtime parameters.
    """
    api_key: Optional[str] = None
    base_url: str = DEFAULT_JEV_BASE_URL
    model: str = DEFAULT_JEV_MODEL
    timeout: float = DEFAULT_JEV_TIMEOUT
    max_retries: int = 1

    @classmethod
    def from_env(
        cls,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> "JevConfig":
        """
        Loads Jev configuration from environment variables with optional parameter overrides.
        Checks both JEV_API_KEY and TYPESAFE_API_KEY. Automatically sources .env or configs/.env if present.
        """
        try:
            from dotenv import load_dotenv
            load_dotenv()
            configs_env = os.path.join(os.getcwd(), "configs", ".env")
            if os.path.exists(configs_env):
                load_dotenv(configs_env)
        except ImportError:
            pass

        env_key = (
            api_key
            or os.environ.get("JEV_API_KEY")
            or os.environ.get("TYPESAFE_API_KEY")
            or None
        )
        env_base = (
            base_url
            or os.environ.get("JEV_BASE_URL")
            or os.environ.get("TYPESAFE_BASE_URL")
            or DEFAULT_JEV_BASE_URL
        )
        env_model = (
            model
            or os.environ.get("JEV_MODEL")
            or os.environ.get("TYPESAFE_MODEL")
            or DEFAULT_JEV_MODEL
        )

        env_timeout_str = os.environ.get("JEV_TIMEOUT")
        if timeout is not None:
            resolved_timeout = float(timeout)
        elif env_timeout_str:
            resolved_timeout = float(env_timeout_str)
        else:
            resolved_timeout = DEFAULT_JEV_TIMEOUT

        return cls(
            api_key=env_key,
            base_url=env_base.rstrip("/"),
            model=env_model,
            timeout=resolved_timeout,
            max_retries=max_retries if max_retries is not None else 1,
        )

    def validate(self) -> None:
        """
        Validates that required configuration parameters are non-empty.
        Raises JevConfigurationError if API key or base_url is missing.
        """
        from backend.app.model.real_jev_provider import JevConfigurationError

        if not self.api_key or not self.api_key.strip():
            raise JevConfigurationError(
                "JEV_API_KEY is missing or empty. "
                "Set JEV_API_KEY or TYPESAFE_API_KEY in the environment or provide it explicitly."
            )
        if not self.base_url or not self.base_url.strip():
            raise JevConfigurationError("JEV base_url cannot be empty.")
        if self.timeout <= 0.0:
            raise JevConfigurationError(f"JEV timeout must be positive, got {self.timeout}")
