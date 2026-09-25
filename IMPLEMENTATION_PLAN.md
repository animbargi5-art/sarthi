# SĀRTHI — Phased Implementation Plan

**Target Event:** Nebius × NVIDIA Global AI Hackathon 2026  
**Document Version:** 1.0 (Phase 0 Foundation)  
**Execution Strategy:** Milestone-Driven, Test-First, Zero-Fake Implementation

---

## Roadmap Overview

```text
[Phase 0: Foundation] (Current)
        │
        ▼
[Phase 1: Core Backend & Nebius Nemotron Engine]
        │
        ▼
[Phase 2: NVIDIA Isaac Sim Environment & Bridge]
        │
        ▼
[Phase 3: Adaptive Disturbance Detection & Recovery Loop]
        │
        ▼
[Phase 4: Operator Dashboard & Telemetry Console]
        │
        ▼
[Phase 5: Benchmark Sweeps, Hardening & Submission]
```

---

## Phase Details

### Phase 0: Repository Foundation & Interface Architecture *(Status: COMPLETED)*
- [x] Establish official repository structure (`backend/app`, `frontend`, `simulation`, `configs`, `scripts`, `tests`, `docs`).
- [x] Author core architectural and requirement specifications (`ARCHITECTURE.md`, `PRD.md`, `TRD.md`, `APP_FLOW.md`, `UI_UX.md`, `BACKEND_SCHEMA.md`).
- [x] Define MIT License and production `.gitignore`.
- [x] Create project evaluation framework (`FEEDBACK.md`).

---

### Phase 1: Core Backend & Nebius Token Factory Integration *(Status: PENDING - Next Phase)*
**Goal:** Build the production asynchronous backend service and connect to Nebius Token Factory for NVIDIA Nemotron inference.
- **Tasks:**
  1. Configure Python environment with `pyproject.toml`, FastAPI, Uvicorn, Pydantic v2, and HTTPX.
  2. Implement `NebiusClient`:
     - Asynchronous HTTP/2 client for `https://api.tokenfactory.nebius.ai/v1`.
     - Strict $450\text{ ms}$ timeout handling and fail-safe fallbacks.
     - Structured JSON schema enforcement with `nvidia/nemotron-4-340b-instruct`.
  3. Implement `DeterministicSafetyValidator`:
     - Kinematic joint limit and velocity limit verification.
     - Cartesian bounding volume enclosure checks.
  4. Write unit tests for schema validation, safety filtering, and client error handling.
- **Deliverables:** Verified FastAPI service capable of ingesting disturbance anomalies and generating validated recovery strategies via Nebius.

---

### Phase 2: NVIDIA Isaac Sim Simulation Environment & Bridge *(Status: PENDING)*
**Goal:** Construct the physical simulation stage, sensor pipeline, and bi-directional communication bridge.
- **Tasks:**
  1. Assemble Omniverse USD stage:
     - Articulated 7-DOF manipulator (Franka Emika Panda).
     - Parallel-jaw gripper with contact reporting meshes.
     - Work table, bin, and manipulable test objects.
  2. Configure synthetic sensor primitives:
     - 6-axis Force/Torque sensor at wrist flange.
     - Joint encoder publisher at 100 Hz.
  3. Develop `DisturbanceInjector`:
     - Python Omni extension to dynamically trigger slip, force impulses, mass shifts, and dynamic obstacles.
  4. Build WebSocket bridge client in Python for low-latency streaming between Isaac Sim and SĀRTHI backend.
- **Deliverables:** Operable Isaac Sim scene publishing real-time telemetry and responding to disturbance injection triggers.

---

### Phase 3: Adaptive Disturbance Detection & Recovery Loop *(Status: PENDING)*
**Goal:** Close the physical AI loop between detection, Nemotron reasoning, and robotic recovery.
- **Tasks:**
  1. Implement `StateBuffer` ring buffer (300 frames) in backend runtime.
  2. Implement `DisturbanceDetector` with configurable thresholding ($F/T$ delta, slip detection, tracking deviation).
  3. Implement `RecoveryCoordinator` state machine:
     - Immediate local freeze hold trigger ($< 5\text{ ms}$).
     - Asynchronous Nemotron deliberation dispatch via Nebius.
     - Motion primitive interpolation and command dispatch to Isaac Sim.
  4. Benchmark closed-loop stability and sensor convergence verification.
- **Deliverables:** Fully autonomous end-to-end recovery loop executing inside Isaac Sim.

---

### Phase 4: Operator Dashboard & Telemetry Console *(Status: PENDING)*
**Goal:** Provide full operator visibility and interactive disturbance demonstration capability.
- **Tasks:**
  1. Initialize frontend client in `frontend/`.
  2. Implement live telemetry visualizer (joint sliders, 6-axis F/T sparklines).
  3. Integrate live Isaac Sim viewport stream.
  4. Build NVIDIA Nemotron reasoning log panel (live diagnosis, tokens/sec, inference latency).
  5. Implement disturbance injection control panel.
- **Deliverables:** Modern, responsive mission control console connecting to backend WebSockets.

---

### Phase 5: Benchmark Sweeps, Hardening & Submission *(Status: PENDING)*
**Goal:** Stress-test system, collect empirical metrics, and assemble submission assets.
- **Tasks:**
  1. Run 50+ automated disturbance trials across varying force magnitudes and slip friction levels.
  2. Quantify key performance indicators:
     - Autonomous recovery success rate ($\ge 90\%$).
     - Average deliberation latency ($\le 350\text{ ms}$).
     - Safety boundary violation count ($0$).
  3. Capture high-fidelity screen recordings of Isaac Sim dynamic recovery.
  4. Finalize submission documentation and README for the Nebius × NVIDIA Global AI Hackathon.
- **Deliverables:** Production codebase, benchmark report, and demonstration video.

---

## Risk Matrix & Mitigation Strategies

| Risk | Impact | Probability | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Inference Latency Spikes (> 500ms)** | High | Medium | Strict client timeout ($450\text{ ms}$) with deterministic safe descent fallback. |
| **LLM Output Kinematic Infeasibility** | Critical | Low | 100% deterministic safety filtering before any motor actuation. |
| **Isaac Sim Bridge Packet Jitter** | Medium | Medium | Local ring-buffer smoothing and timestamp interpolation in backend. |
| **Grasp Slip Recovery Stall** | Medium | Low | Dynamic compliance adjustment and multi-stage regrasp primitives. |
