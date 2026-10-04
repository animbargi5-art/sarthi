"""
SĀRTHI V3-4B — Local Laya Fast Decision Provider Configuration.
Provides validated, immutable configuration for local Laya server endpoints.
Does NOT require cloud API keys or external secrets.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional


DEFAULT_LAYA_BASE_URL: str = "http://127.0.0.1:8000"
DEFAULT_LAYA_TIMEOUT: float = 5.0
DEFAULT_LAYA_MODEL: str = "typed-decisions"


@dataclass(frozen=True)
class LayaConfig:
    """
    Configuration parameters for local Laya decision provider.
    Connects to local Laya server without requiring external API keys.
    """
    base_url: str = DEFAULT_LAYA_BASE_URL
    model: str = DEFAULT_LAYA_MODEL
    timeout: float = DEFAULT_LAYA_TIMEOUT
    max_retries: int = 1

    @classmethod
    def from_env(
        cls,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> "LayaConfig":
        """
        Loads Laya configuration from environment variables with optional parameter overrides.
        Checks LAYA_BASE_URL, or synthesizes from LAYA_HOST and LAYA_PORT.
        """
        try:
            from dotenv import load_dotenv
            load_dotenv()
            configs_env = os.path.join(os.getcwd(), "configs", ".env")
            if os.path.exists(configs_env):
                load_dotenv(configs_env)
        except ImportError:
            pass

        host = os.environ.get("LAYA_HOST", "127.0.0.1")
        port = os.environ.get("LAYA_PORT", "8000")
        fallback_base = f"http://{host}:{port}"

        env_base = (
            base_url
            or os.environ.get("LAYA_BASE_URL")
            or fallback_base
        )
        env_model = (
            model
            or os.environ.get("LAYA_MODEL")
            or DEFAULT_LAYA_MODEL
        )

        env_timeout_str = os.environ.get("LAYA_TIMEOUT")
        if timeout is not None:
            resolved_timeout = float(timeout)
        elif env_timeout_str:
            resolved_timeout = float(env_timeout_str)
        else:
            resolved_timeout = DEFAULT_LAYA_TIMEOUT

        return cls(
            base_url=env_base.rstrip("/"),
            model=env_model,
            timeout=resolved_timeout,
            max_retries=max_retries if max_retries is not None else 1,
        )

    def validate(self) -> None:
        """Validates configuration parameters."""
        if not self.base_url or not self.base_url.strip():
            from backend.app.model.real_jev_provider import JevConfigurationError
            raise JevConfigurationError("LAYA base_url cannot be empty.")
        if self.timeout <= 0.0:
            from backend.app.model.real_jev_provider import JevConfigurationError
            raise JevConfigurationError(f"LAYA timeout must be positive, got {self.timeout}")
