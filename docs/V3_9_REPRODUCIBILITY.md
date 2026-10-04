# SĀRTHI V3-9 — Repeated-Run & Reproducibility Validation Report

## 1. Validation Purpose & Scope

The purpose of Phase V3-9 is to systematically validate the repeatability and reproducibility of the **complete integrated SĀRTHI V3 decision-to-physics pipeline** across controlled repeated runs.

While Phase V3-8 strictly validated the physical simulation subsystem (kinematics, joint limits, stability, collision boundaries, and contact mechanics), Phase V3-9 validates the complete cyber-physical loop:

$$\text{Human Instruction} \longrightarrow \text{Nemotron} \longrightarrow \text{DecisionContext} \longrightarrow \text{Laya} \longrightarrow \text{CandidateAction} \longrightarrow \text{SarthiDecisionEngine} \longrightarrow \text{MuJoCo} \longrightarrow \text{Verification} \longrightarrow \text{Recovery}$$

A central objective of V3-9 is to empirically determine which parts of the pipeline are deterministic and which parts are subject to live model inference variability, establishing an architectural boundary between cognitive proposal variability and deterministic physical execution safety.

---

## 2. Core Architectural Invariants

Throughout all repeated executions, the following architectural invariants are strictly enforced:

1. **Sole Physical Authority**: `SarthiDecisionEngine` remains the sole physical authority. No model output commands joints, end-effectors, or simulation states directly.
2. **Bounded Non-Authoritative Laya**: `Laya` (via `JevDecisionProvider` / `LocalLayaDecisionProvider`) operates strictly as an advisory policy proposing bounded decisions (`REPOSITION`, `STOP`) over discrete option sets.
3. **Task-Understanding Only**: `Nemotron-3-Ultra-550b-a55b` via the Nebius Token Factory operates solely on natural language task comprehension and intent extraction.
4. **Canonical Boundary**: `CandidateActionInjector` is the only bridge translating model proposals into canonical `CandidateAction` representations. Out-of-bounds proposals are rejected with `JevValidationError`.
5. **No AI Direct Control**: No AI model may directly command MuJoCo actuators or bypass kinematic constraints.
6. **No Fabricated Determinism**: Prompts are not artificially constrained to force identical text, nor are model probabilities mocked or faked during live evaluations.
7. **Physics Parameter Invariance**: All physics parameters, timesteps ($\Delta t = 0.002$ s), and Franka Panda joint/velocity limits remain identical to V3-8.

---

## 3. The Three Reproducibility Levels

Validation is structured across three distinct tiers:

### Level A — Physics-Only Replay (5 Runs)
- **Objective**: Re-verify that headless MuJoCo physics, contact dynamics, and the Franka Panda controller are bitwise or sub-millimeter deterministic when executing the canonical recovery action sequence from identical keyframe resets.
- **Components Active**: MuJoCo 7-DOF Panda, DLS-IK solver, Semi-implicit Euler integrator, fixed action trajectory (`APPROACH`, `GRASP`, `REPOSITION`, `MOVE`, `RELEASE`).
- **Target**: Positional identity within numerical threshold ($\le 0.0001$ m), identical world-state transitions, identical placement error.

### Level B — Decision-Pipeline Replay (5 Runs)
- **Objective**: Evaluate the repeatability of the decision path under fixed cognitive inputs (`TaskUnderstanding` and `PhysicalSituation`).
- **Components Active**: `DecisionContextBuilder` $\to$ `DecisionQuestion` $\to$ Local Laya (`/v1/systemone`, model `typed-decisions`) $\to$ `JevDecision` $\to$ `CandidateActionInjector` $\to$ `SarthiDecisionEngine` deterministic evaluation.
- **Distinction**: Distinguishes between logical decision consistency (selected choice), probability consistency (floating-point distribution), and physical validation consistency (safety constraint evaluation).

### Level C — Full Live V3 Repeatability (5 Complete Cycles)
- **Objective**: Run the end-to-end cyber-physical pipeline 5 consecutive times in real time under dynamic obstacle disturbance (`PATH_BLOCKED`).
- **Components Active**:
  1. Real hosted `Nemotron-3-Ultra-550b-a55b` via Nebius Token Factory API.
  2. Real local Laya (`0.3.25`) server running at `http://127.0.0.1:8000`.
  3. Real headless MuJoCo Franka Panda simulation (`tabletop_pick_and_place_mvp`).
  4. Real physical obstacle injection triggering autonomous `PATH_BLOCKED` detection and recovery.
- **Telemetry Recorded**: Task understanding, per-step Laya decisions, candidate actions, deterministic constraint decisions, step-by-step action execution, recovery events, final object pose, final placement error, and per-step latency profiles.

---

## 4. Empirical Results Summary

The empirical findings from all 5 runs across Levels A, B, and C are archived in [`records/v3_9_reproducibility.json`](file:///d:/Projects/SĀRTHI/records/v3_9_reproducibility.json).

### Level A: Physics Replay Results (5 Runs)

| Metric | Measured Value | Requirement / Bound | Status |
|---|---|---|:---:|
| Total Runs Attempted | $5$ | $5$ | PASS |
| Successful Completions | $5 / 5$ ($100\%$) | $100\%$ | PASS |
| Action Sequence Identity | Identical across all runs | Identical | PASS |
| World-State Transition Match | $100\%$ | $100\%$ | PASS |
| Mean Placement Error | $0.01756$ m | $\le 0.060$ m | PASS |
| Max Placement Error | $0.01756$ m | $\le 0.060$ m | PASS |
| Placement Error Std Dev | $0.000000$ m | $\le 0.001$ m | PASS |
| Max Euclidean Pose Spread | $0.000000$ m | $\le 0.0001$ m | PASS |

*Conclusion*: Headless CPU MuJoCo simulation of the 7-DOF Franka Panda arm achieves exact sub-millimeter determinism across resets.

---

### Level B: Decision-Pipeline Replay Results (5 Queries)

| Metric | Measured Value | Requirement / Bound | Status |
|---|---|---|:---:|
| Total Bounded Queries | $5$ | $5$ | PASS |
| Selected Option Agreement | $100\%$ (`REPOSITION` in 5/5) | Bounded valid | PASS |
| Candidate Injection Validity | $100\%$ ($5/5$ valid `CandidateAction`) | $100\%$ valid | PASS |
| Deterministic Engine Agreement | $100\%$ (`REPOSITION` accepted in 5/5) | Deterministic | PASS |
| Mean Decision Confidence | $0.5619$ | Bounded $[0.0, 1.0]$ | PASS |
| Rejection Reasons | $0$ rejections ($5/5$ approved) | Safe & collision-free | PASS |

*Conclusion*: Fixed context queries against local Laya produced logically identical bounded choices (`REPOSITION`), which were uniformly validated and accepted by `SarthiDecisionEngine`.

---

### Level C: Full Live V3 Repeatability Results (5 Real Cycles)

| Run ID | Nemotron Target | Laya Recovery Decision | Recovery Status | Final Placement Error | Task Status | Failure Class | Total Cycle Time |
|---|---|---|---|---|---|---|---|
| `run_live_1` | `blue_target_zone` | `REPOSITION` ($p=0.5619$) | SUCCESS | $0.04412$ m | COMPLETED | `NONE` | $3878.8$ ms |
| `run_live_2` | `blue_target_zone` | `REPOSITION` ($p=0.5619$) | SUCCESS | $0.04412$ m | COMPLETED | `NONE` | $3844.7$ ms |
| `run_live_3` | `blue_target_zone` | `REPOSITION` ($p=0.5619$) | SUCCESS | $0.04412$ m | COMPLETED | `NONE` | $3998.0$ ms |
| `run_live_4` | `blue_target_zone` | `REPOSITION` ($p=0.5619$) | SUCCESS | $0.04412$ m | COMPLETED | `NONE` | $3928.3$ ms |
| `run_live_5` | `blue_target_zone` | `REPOSITION` ($p=0.5619$) | SUCCESS | $0.04412$ m | COMPLETED | `NONE` | $3747.3$ ms |

---

## 5. Aggregate Reproducibility Metrics

| Metric | Value | Interpretation |
|---|:---:|---|
| **Total Live Runs Attempted** | $5$ | Complete cycles executed |
| **Total Live Runs Completed** | $5$ | $100\%$ completion rate |
| **Task Success Rate** | **$100.0\%$** ($1.00$) | All tasks achieved final goal pose |
| **Recovery Success Rate** | **$100.0\%$** ($1.00$) | Obstacle detected and bypassed cleanly |
| **Action Sequence Match Rate** | **$100.0\%$** ($1.00$) | All 5 runs executed the same 5-step sequence |
| **Laya Decision Agreement Rate** | **$100.0\%$** ($1.00$) | All 5 queries selected `REPOSITION` |
| **Deterministic Validation Agreement** | **$100.0\%$** ($1.00$) | `SarthiDecisionEngine` validated identical candidates |
| **Final Placement Error Mean** | **$0.04412$ m** | Within destination zone tolerance ($0.060$ m) |
| **Final Placement Error Max** | **$0.04412$ m** | Strict compliance across all runs |
| **Final Placement Error Std Dev** | **$0.000000$ m** | Zero physical dispersion across runs |
| **Final Object Pose Spread** | **$0.000000$ m** | Identical final 3D coordinates |
| **Total Failure Count** | **$0$** | No physical, execution, or validation drops |
| **Unexpected Behavior Count** | **$0$** | Zero unbounded or anomalous transitions |

---

## 6. Failure Classification Taxonomy

All runs and potential divergent steps are evaluated against the standard SĀRTHI failure taxonomy:

| Classification | Meaning & Criteria | Level C Frequency |
|---|---|:---:|
| `NONE` | Run completed nominal and recovery goals within tolerance | **5** |
| `MODEL_VARIABILITY` | LLM generated differing semantic text/tokens, but task intent was preserved safely | 0 |
| `NETWORK_LATENCY` | Remote provider API exceeded deadline timeout ($\tau > \tau_{\text{timeout}}$) | 0 |
| `CONTEXT_VARIATION` | Physical context attributes diverged between runs | 0 |
| `BOUNDED_DECISION_VARIATION` | Laya selected an alternative bounded option (e.g., `STOP` instead of `REPOSITION`) | 0 |
| `DETERMINISTIC_VALIDATION_FAILURE` | Proposed candidate action violated kinematic or safety constraints | 0 |
| `PHYSICS_FAILURE` | Simulation exploded, collided unexpectedly, or produced NaN/Inf states | 0 |
| `EXECUTION_FAILURE` | Controller or actuator failed to achieve target setpoint | 0 |
| `VERIFICATION_FAILURE` | Final object position exceeded target zone threshold ($> 0.060$ m) | 0 |
| `UNKNOWN` | Unclassified anomalous condition | 0 |

---

## 7. Safety Architecture & Non-Bypass Verification

A critical verification test was executed to confirm that no model output can bypass physical validation:

1. **Controlled Mutation**: A synthetic unviable candidate action (`MOVE` directly through the obstacle) was injected.
2. **Deterministic Evaluation**: `SarthiDecisionEngine` and `ConstraintValidator` evaluated the candidate against workspace bounds and collision geometries.
3. **Enforcement**: The candidate was deterministically rejected (`PATH_BLOCKED`, clearance $< 0.030$ m), proving that even if an AI model produces an invalid or varied output, zero unauthorized physical actions can reach MuJoCo.

---

## 8. Secret-Free Telemetry Verification

In accordance with strict security protocols:
- `records/v3_9_reproducibility.json` and this document were scanned for API credentials, tokens, and authorization headers (`Bearer`, `sk-`, `NEBIUS_API_KEY`, etc.).
- All authorization tokens are managed strictly via system environment variables and are excluded from serialized telemetry objects.

---

## 9. Known Limitations

The following limitations are explicitly and honestly documented:
1. **Model Stochasticity**: While local Laya selected `REPOSITION` across all 5 Level B and Level C queries in this validation session, probability distributions and option choices may vary if sampling temperature or seeds change. The architecture relies on `SarthiDecisionEngine` for safety rather than assuming model determinism.
2. **Hosted Provider Latency**: Hosted `Nemotron-3-Ultra-550b-a55b` latency on Nebius Cloud varied between $1573.7$ ms and $1761.5$ ms depending on remote load. While response structure was consistent, network latency is non-deterministic.
3. **Simulation vs. Reality**: MuJoCo CPU physics is deterministic under identical floating-point initialization and semi-implicit Euler integration. Real-world robotic deployments introduce non-deterministic motor backlash, sensor noise, variable tabletop friction, and camera jitter.
4. **Sample Size**: Five live runs provide a rigorous empirical demonstration of repeatability for multi-second closed-loop robotic tasks, but do not constitute infinite-sample statistical proof.

---

## 10. Conclusion

Phase V3-9 successfully establishes that:
- The **physics simulation layer is completely deterministic** ($0.000000$ m pose spread across repeated runs).
- The **deterministic safety engine provides an absolute boundary** ensuring that AI model variability cannot compromise physical safety.
- The **complete integrated SĀRTHI V3 pipeline is $100\%$ repeatable** under the tested scenarios, achieving $100\%$ task success, $100\%$ recovery success, and placement error of $0.0441$ m ($\le 0.060$ m tolerance).
