"""
SĀRTHI V3-11 — Local Demonstration HTTP Server.
Serves the judge-ready visual dashboard and provides telemetry replay/live endpoints.

Strict Architectural Guarantees:
1. The web server and UI layer cannot directly actuate robot joints or MuJoCo.
2. All execution passes through SarthiDemoService, CandidateActionInjector, and SarthiDecisionEngine.
3. Replay mode is explicitly tagged and never claims to be live.
4. No API keys or secrets are served in payloads.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.demo.models import DemoMode, DemoSessionTelemetry, SystemStatus
from backend.app.demo.service import SarthiDemoService

logger = logging.getLogger(__name__)

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
_FRONTEND_DIR = _PROJECT_ROOT / "frontend"
_INDEX_HTML_PATH = _FRONTEND_DIR / "index.html"
_DEFAULT_TELEMETRY_PATH = _PROJECT_ROOT / "records" / "v3_11_demo_telemetry.json"


def create_demo_app() -> FastAPI:
    """Creates the FastAPI demonstration application."""
    app = FastAPI(
        title="SĀRTHI V3 Visual Demonstration Server",
        description="Judge-ready visual demonstration layer and telemetry event visualizer.",
        version="3.11.0",
    )

    demo_service = SarthiDemoService()

    @app.get("/", response_class=HTMLResponse)
    def get_dashboard() -> HTMLResponse:
        """Serves the primary demonstration dashboard."""
        if not _INDEX_HTML_PATH.exists():
            raise HTTPException(status_code=404, detail="Dashboard UI index.html not found")
        with open(_INDEX_HTML_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)

    @app.get("/api/status")
    def get_status() -> Dict[str, Any]:
        """Returns subsystem connectivity and readiness state."""
        return SystemStatus().model_dump()

    @app.get("/api/telemetry")
    def get_telemetry() -> Dict[str, Any]:
        """Returns current saved demo telemetry record."""
        if not _DEFAULT_TELEMETRY_PATH.exists():
            # If not yet generated, run live baseline
            session = demo_service.run_live_demo(telemetry_output_path=str(_DEFAULT_TELEMETRY_PATH))
            return session.to_sanitized_dict()

        with open(_DEFAULT_TELEMETRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    @app.post("/api/run-live")
    def trigger_live() -> Dict[str, Any]:
        """Executes a real live demonstration across Nemotron, Laya, and MuJoCo."""
        session = demo_service.run_live_demo(telemetry_output_path=str(_DEFAULT_TELEMETRY_PATH))
        return session.to_sanitized_dict()

    @app.post("/api/run-replay")
    def trigger_replay() -> Dict[str, Any]:
        """Loads and returns the offline replay record."""
        session = demo_service.run_replay_demo(telemetry_path=str(_DEFAULT_TELEMETRY_PATH))
        return session.to_sanitized_dict()

    @app.post("/api/run-failure")
    def trigger_failure() -> Dict[str, Any]:
        """Demonstrates the V3-10 failure handling and resiliency fallback."""
        return demo_service.run_failure_demo()

    return app


# Export ASGI application instance
app = create_demo_app()
