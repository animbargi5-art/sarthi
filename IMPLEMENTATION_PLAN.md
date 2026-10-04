# SĀRTHI — Phased Implementation Plan

**Project Name:** SĀRTHI  
**Tagline:** An adaptive Physical AI architecture for context-aware, responsibility-driven robotic decision-making under changing physical conditions.  
**Target Event:** Nebius × NVIDIA Global AI Hackathon 2026  
**Document Version:** Version: 3.0  
**Status:** Draft — V3 Architecture Update  

---

## 1. Safety Architecture & Core Invariant

The following non-negotiable architectural invariant governs all implementation phases across SĀRTHI V3:

> [!IMPORTANT]
> **Core System Invariant:**  
> **"AI models propose or select bounded semantic candidates; deterministic software validates physical feasibility; only validated actions reach the robot controller."**  
> **"Jev must never bypass deterministic safety validation."**

No AI model—including NVIDIA Nemotron and the planned Jev fast decision layer—commands robot motors directly or alters low-level control parameters without passing through the deterministic SĀRTHI Decision Engine.

---

## 2. Validated Baseline & Roadmap Overview

### 2.1 Validated Baseline (Phases A–H Completed)
SĀRTHI V3 directly builds upon the validated baseline established in Phases A through H:
- **Test Suite:** 186 unit and integration tests passing ($100\%$, 0 failed, 0 errors).
- **Physical Environment:** MuJoCo (`mujoco >= 3.1.0`) with 7-DOF Franka Emika Panda arm and parallel gripper.
- **Cognitive Model:** `nvidia/Nemotron-3-Ultra-550b-a55b` via Nebius Token Factory (`https://api.tokenfactory.nebius.com/v1`).
- **Physical IK & Control:** Damped-Least-Squares (DLS) differential IK with joint-limit clamping and state extraction.
- **Physical Disturbance Recovery:** Dynamic obstacle (`PATH_BLOCKED`) injected post-grasp; candidate `MOVE` rejected (clearance $0.066\text{ m} < 0.082\text{ m}$); `REPOSITION` recovery elevated payload to $z \approx 0.35\text{ m}$; final placement error $0.0535\text{ m} \le 0.0600\text{ m}$.

### 2.2 V3 Phased Engineering Roadmap

```text
[Phase V3-1: Interface Freezing & Schema Formalization]
                     │
                     ▼
[Phase V3-2: Jev Provider Boundary & Abstract Adapter]
                     │
                     ▼
[Phase V3-3: Decision Context Builder]
                     │
                     ▼
[Phase V3-4: First Real Jev Bounded Decision Query]
                     │
                     ▼
[Phase V3-5: Engine Integration with Safe Candidate Injection]
                     │
                     ▼
[Phase V3-6: Human Command + Physical Situation End-to-End Flow]
                     │
                     ▼
[Phase V3-7: Decision Latency Instrumentation & Benchmarking]
                     │
                     ▼
[Phase V3-8: Physics Validation Framework Implementation]
                     │
                     ▼
[Phase V3-9: Repeated-Run & Reproducibility Validation]
                     │
                     ▼
[Phase V3-10: Failure & Resiliency Fallback Suite]
                     │
                     ▼
[Phase V3-11: Visual Demonstration & Telemetry Logging]
                     │
                     ▼
[Phase V3-12: Final V3 Validation & Submission Readiness]
```

---

## 3. Detailed Phase Specifications

### Phase V3-1: Interface Freezing & Schema Formalization
- **Objective:** Codify and freeze Pydantic models for V3 data contracts (`HumanInstruction`, `PhysicalSituation`, `DecisionContext`, `DecisionQuestion`, `JevDecision`, `CandidateAction`, `Decision`, `PhysicsValidationResult`, `DecisionLatencyRecord`).
- **Files Likely Affected:** `backend/app/model/`, `backend/app/schemas/`
- **Tests:** Unit tests verifying schema serialization, roundtrip parsing, and validation constraints.
- **Acceptance Criteria:** Full Pydantic v2 schemas pass strict typing and serialization tests without breaking existing models.
- **Dependencies:** None.
- **Rollback Behavior:** Revert schema files to Phase H baseline commit.

---

### Phase V3-2: Jev Provider Boundary & Abstract Adapter
- **Objective:** Implement the `JevDecisionProvider` abstract base class and a deterministic `MockJevProvider` for offline testing. Prevent any hardcoded vendor coupling.
- **Files Likely Affected:** `backend/app/providers/jev_provider.py`, `backend/app/providers/mock_jev_provider.py`
- **Tests:** Unit tests verifying provider contract, exception handling, and mock response generation.
- **Acceptance Criteria:** Provider interface correctly defined; mock provider returns structured `JevDecision` responses within schema bounds.
- **Dependencies:** Phase V3-1 schemas.
- **Rollback Behavior:** Isolate provider module; fallback to baseline mock providers.

---

### Phase V3-3: Decision Context Builder
- **Objective:** Implement `DecisionContextBuilder` to filter active `WorldState`, target goals, and active constraints into compact, bounded payloads for decision queries.
- **Files Likely Affected:** `backend/app/orchestration/context_builder.py`
- **Tests:** Unit tests verifying context extraction from simulated `WorldState` snapshots across nominal, blocked-path, and post-grasp states.
- **Acceptance Criteria:** Context payload contains only relevant subset of physical variables; zero raw simulation pointers or unstructured memory dumps.
- **Dependencies:** Phase V3-1 schemas, `SarthiMuJoCoStateReader`.
- **Rollback Behavior:** Fall back to full `WorldState` passing in `TaskRunner`.

---

### Phase V3-4: First Real Jev Bounded Decision Query
- **Objective:** Connect `JevDecisionProvider` to live or configured external decision service for bounded recovery question evaluation (`REPOSITION`, `REPLAN`, `REGRASP`, `STOP`).
- **Files Likely Affected:** `backend/app/providers/jev_provider.py`, `configs/.env.example`
- **Tests:** Contract integration tests testing bounded question queries, timeout enforcement, and response parsing.
- **Acceptance Criteria:** Provider successfully evaluates bounded question and returns valid `JevDecision` with option probabilities.
- **Dependencies:** Phase V3-2, Phase V3-3.
- **Rollback Behavior:** Enable offline mock mode automatically if live endpoint is unreachable.

---

### Phase V3-5: Engine Integration with Safe Candidate Injection
- **Objective:** Wire Jev candidate proposals into `SarthiDecisionEngine` as proposed candidate actions, ensuring candidate actions pass through full deterministic constraint validation.
- **Files Likely Affected:** `backend/app/decision/engine.py`, `backend/app/orchestration/task_runner.py`
- **Tests:** Unit tests confirming that invalid Jev proposals are strictly rejected by the Decision Engine (e.g. clearance violations).
- **Acceptance Criteria:** Jev proposal accepted if and only if deterministic constraints are satisfied; 100% rejection of unsafe suggestions.
- **Dependencies:** Phase V3-4, existing SĀRTHI Decision Engine.
- **Rollback Behavior:** Revert `TaskRunner` to baseline heuristic candidate generation.

---

### Phase V3-6: Human Command + Physical Situation End-to-End Flow
- **Objective:** Integrate the complete flow: Natural-language command (Nemotron) $\to$ `WorldState` observation $\to$ `DecisionContext` $\to$ Jev candidate proposal $\to$ deterministic Decision Engine validation $\to$ MuJoCo physical execution.
- **Files Likely Affected:** `backend/app/orchestration/task_runner.py`, `scripts/mujoco/run_sarthi_mujoco_e2e.py`
- **Tests:** End-to-end simulation tests replicating pick-and-place with dynamic obstacle avoidance.
- **Acceptance Criteria:** Robot executes pick-and-place, recovers from `PATH_BLOCKED` via `REPOSITION`, and settles object within $0.0600\text{ m}$ tolerance.
- **Dependencies:** Phase V3-5.
- **Rollback Behavior:** Revert orchestration to Phase H `run_sarthi_mujoco_e2e.py` baseline.

---

### Phase V3-7: Decision Latency Instrumentation & Benchmarking
- **Objective:** Implement granular timestamp tracking across all decision and execution stages. Record and output structured `DecisionLatencyRecord` telemetry.
- **Files Likely Affected:** `backend/app/orchestration/telemetry.py`, `records/`
- **Tests:** Benchmark scripts measuring and aggregating latency across 20+ runs.
- **Acceptance Criteria:** Telemetry captures exact duration for task understanding, Jev decision, validation, IK, and physics stepping.
- **Dependencies:** Phase V3-6.
- **Rollback Behavior:** Disable latency logging wrappers without affecting execution.

---

### Phase V3-8: Physics Validation Framework Implementation
- **Objective:** Implement the 12-point Physics Validation Framework to evaluate simulation fidelity, numerical stability, and kinematic constraints.
- **Files Likely Affected:** `simulation/mujoco_runtime/physics_validation.py`, `tests/test_physics_validation.py`
- **Tests:** Automated test runner executing PV-1 through PV-12 test suites.
- **Acceptance Criteria:** All 12 validation suites execute and produce structured `PhysicsValidationResult` records against configured thresholds.
- **Dependencies:** `SarthiMuJoCoArticulation`, `SarthiMuJoCoStateReader`.
- **Rollback Behavior:** Quarantine physics validation suite as optional diagnostic script.

---

### Phase V3-9: Repeated-Run & Reproducibility Validation
- **Objective:** Execute 50 automated deterministic replays of the benchmark pick-place-recover scenario to quantify placement repeatability and contact stability.
- **Files Likely Affected:** `scripts/mujoco/benchmark_repeatability.py`, `records/`
- **Tests:** Automated batch runner measuring variance in final object coordinates and IK convergence.
- **Acceptance Criteria:** Placement variance is within configured tolerance across all runs; zero simulation crashes or divergence.
- **Dependencies:** Phase V3-8.
- **Rollback Behavior:** None required (read-only benchmarking).

---

### Phase V3-10: Failure & Resiliency Fallback Suite
- **Objective:** Rigorously test all failure modes: Jev timeout, Nemotron timeout, malformed JSON, invalid target entities, and unrecoverable physical blockages.
- **Files Likely Affected:** `tests/test_v3_resiliency.py`
- **Tests:** Chaos and fault-injection test suite simulating dropped packets, slow responses, and boundary violations.
- **Acceptance Criteria:** System gracefully degrades to deterministic candidate ranking or safe controlled stops; zero unhandled crashes or uncommanded actuations.
- **Dependencies:** Phase V3-6, Phase V3-7.
- **Rollback Behavior:** None (test-only suite).

---

### Phase V3-11: Visual Demonstration & Telemetry Logging
- **Objective:** Update interactive 3D viewer runner and telemetry recording to visually and analytically demonstrate the V3 pipeline (human command $\to$ situation $\to$ fast decision $\to$ robot action).
- **Files Likely Affected:** `scripts/mujoco/run_sarthi_mujoco_e2e.py`
- **Tests:** Manual viewer run and automated headless record generation.
- **Acceptance Criteria:** Live 3D viewer displays robot motion, obstacle disturbance, and recovery path while streaming decision telemetry.
- **Dependencies:** Phase V3-6.
- **Rollback Behavior:** Maintain existing `--headless` and `--viewer` flags from Phase H.

---

### Phase V3-12: Final V3 Validation & Submission Readiness
- **Objective:** Final repository audit, end-to-end regression validation (all unit tests + V3 suites), documentation alignment, and artifact archiving.
- **Files Likely Affected:** `README.md`, `FEEDBACK.md`, `records/`
- **Tests:** Full test suite execution across all test files.
- **Acceptance Criteria:** 100% tests passing; complete telemetry records generated; documentation fully aligned with implemented code.
- **Dependencies:** Phases V3-1 through V3-11.
- **Rollback Behavior:** Full checkpoint rollback to previous stable commit.
