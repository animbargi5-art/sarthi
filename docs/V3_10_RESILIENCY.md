# SĀRTHI V3-10 — Failure & Resiliency Fallback Suite Report

## 1. Executive Summary & Purpose

The purpose of Phase V3-10 is to implement and validate the **SĀRTHI V3 Failure & Resiliency Fallback Suite**.

In cyber-physical systems, autonomous foundation models (Nemotron, Laya) operate at high semantic levels but are inherently susceptible to transient network timeouts, service outages, malformed syntax, confidence drift, and out-of-bounds proposals. Physical controllers, inverse kinematics solvers, and contact mechanics can also experience unreachable target poses, execution slips, or environmental disturbances.

Phase V3-10 systematically proves the central architectural invariant:

$$\text{AI Failure / Physical Disturbance} \longrightarrow \text{Bounded Fallback} \longrightarrow \text{Deterministic Validation} \longrightarrow \text{Safe Action or SAFE STOP} \longrightarrow \text{Verification}$$

**Zero AI output or fallback candidate can ever bypass `SarthiDecisionEngine` or command MuJoCo actuators directly.**

---

## 2. Failure Taxonomy

SĀRTHI V3-10 categorizes system anomalies into five explicitly separated failure domains:

| Category | Domain | Description & System Response |
|---|---|---|
| **Provider Failure** | Cognitive / Remote API | Nemotron or Laya experiences network timeout, connection refusal, or malformed JSON payloads. Triggers safe cognitive abort or local deterministic recovery candidate. |
| **Decision Rejection** | Cyber-Physical Boundary | Laya returns an out-of-bounds option or an action violating physical constraints (e.g. `MOVE` through obstacle). Rejected deterministically by `CandidateActionInjector` or `ConstraintValidator`. |
| **Physical Execution Failure** | Actuation / Joint Control | IK solver fails to converge for unreachable coordinates, or actuator command fails. Trajectory execution is aborted without NaN/Inf generation. |
| **Verification Failure** | Sensor / Outcome Audit | Physical post-action verification detects that expected state conditions (clamped gripper, target coordinates) were not achieved. Prevents blind forward progression. |
| **Environmental Disturbance** | External World | Dynamic obstacles (`PATH_BLOCKED`) or scene modifications occur during execution. Injected into WorldState constraints, prompting validated recovery replanning. |

---

## 3. Formal Fallback Hierarchy

To ensure transparency and explainability, the fallback policy operates on an explicit 3-tier hierarchy:

```
[ Human Instruction + Physical WorldState ]
                     │
                     ▼
       ┌───────────────────────────┐
       │   Tier 1: AI Proposal     │ ──(Valid Proposal)──► [ SarthiDecisionEngine ]
       └───────────────────────────┘                        │ (ConstraintValidator)
                     │                                      │
            (Unavailable / Error)                  (Rejected: Constraint Breach)
                     │                                      │
                     ▼                                      ▼
       ┌────────────────────────────────────────────────────────────┐
       │             Tier 2: Deterministic Recovery Candidate       │
       │   (Context-derived candidate: clearance lift / regrasp)     │
       └────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                         [ SarthiDecisionEngine ]
                          (ConstraintValidator)
                                     │
                     ┌───────────────┴───────────────┐
             (Accepted & Safe)               (No Valid Motion)
                     │                               │
                     ▼                               ▼
       ┌───────────────────────────┐   ┌───────────────────────────┐
       │ Execute Validated Action  │   │     Tier 3: SAFE STOP     │
       │     (MuJoCo Physics)      │   │  (Zero velocity joint-hold)│
       └───────────────────────────┘   └───────────────────────────┘
```

### Invariant Rules
1. **No Shadow AI**: Fallback logic never queries a secondary LLM or stochastic model. Fallbacks are rule-bounded, deterministic, and verifiable.
2. **Authority Primacy**: `SarthiDecisionEngine` remains the sole physical authority. All Tier 1 and Tier 2 candidates undergo identical constraint validation.
3. **Fail-Safe Primitive**: Tier 3 `SAFE STOP` (`ActionType.STOP`) is always present in candidate pools and guarantees immediate zero-velocity holding of joint configurations.

---

## 4. Failure Detection Points

Failure detection is embedded at distinct pipeline boundaries:

```
Human Command
     │
     ▼  [Point 1: Task Understanding] ──► Nemotron timeout / schema failure
TaskUnderstanding
     │
     ▼  [Point 2: Target Resolution]   ──► Target entity missing from WorldState
DecisionContext
     │
     ▼  [Point 3: Decision Proposal]   ──► Laya timeout / connection error / malformed response
JevDecision
     │
     ▼  [Point 4: Candidate Injection] ──► Out-of-bounds options / unknown action types
CandidateAction
     │
     ▼  [Point 5: Deterministic Engine]──► Workspace limits / obstacle collisions / reachability
Decision
     │
     ▼  [Point 6: Kinematics / IK]     ──► DLS-IK convergence failure / singularity / reach
Joint Trajectory
     │
     ▼  [Point 7: Physical Execution]  ──► Actuator torque saturation / joint clamping
MuJoCo Simulation
     │
     ▼  [Point 8: State Verification]  ──► Contact slip / pose tolerance / dropped payload
Updated WorldState
```

---

## 5. Controlled Failure Injection Methodology & Results

All 13 controlled failure scenarios were executed and evaluated using `ResiliencySuite`.

| # | Scenario | Injected Condition | Detection Stage | Strategy Engaged | Final Status | Unsafe Action? | AI Bypass? |
|---|---|---|---|---|---|:---:|:---:|
| **1** | **Laya Timeout** | 500ms API timeout on recovery query | `DECISION_PROPOSAL` | `DETERMINISTIC_RECOVERY` | `COMPLETED` | **None** | **No** |
| **2** | **Laya Connection Failure** | Connection refused (`127.0.0.1:8000` offline) | `DECISION_PROPOSAL` | `DETERMINISTIC_RECOVERY` | `COMPLETED` | **None** | **No** |
| **3** | **Malformed Laya Response** | Non-JSON / missing required fields | `DECISION_PROPOSAL` | `DETERMINISTIC_RECOVERY` | `COMPLETED` | **None** | **No** |
| **4** | **Laya Out-of-Bounds** | Proposes option not in available choices | `CANDIDATE_INJECTION` | `DETERMINISTIC_RECOVERY` | `COMPLETED` | **None** | **No** |
| **5** | **Nemotron Timeout** | Remote Nebius API timeout | `TASK_UNDERSTANDING` | `SAFE_ABORT` | `STOPPED_SAFELY` | **None** | **No** |
| **6** | **Nemotron Malformed** | Incomplete Pydantic schema returned | `TASK_UNDERSTANDING` | `SAFE_ABORT` | `STOPPED_SAFELY` | **None** | **No** |
| **7** | **Deterministic Rejection** | Unsafe `MOVE` through obstacle ($p=0.99$) | `DETERMINISTIC_VALIDATION` | `SAFE_STOP` | `STOPPED_SAFELY` | **None** | **No** |
| **8** | **IK Failure** | Unreachable target waypoint ($z=5.0$m) | `IK_SOLVER` | `SAFE_STOP` | `STOPPED_SAFELY` | **None** | **No** |
| **9** | **Execution Failure** | Invalid grasp parameters / actuator fault | `PHYSICAL_EXECUTION` | `SAFE_STOP` | `STOPPED_SAFELY` | **None** | **No** |
| **10** | **Verification Failure** | Unmet Cartesian tolerance ($< 0.01$mm) | `VERIFICATION` | `SAFE_STOP` | `STOPPED_SAFELY` | **None** | **No** |
| **11** | **Object / Grasp Failure** | Gripper closes in empty air (no contact) | `STATE_OBSERVATION` | `SAFE_STOP` | `STOPPED_SAFELY` | **None** | **No** |
| **12** | **PATH_BLOCKED Disturbance**| Obstacle placed on direct transit line | `DECISION_PROPOSAL` | `PROPOSAL_ACCEPTED` | `COMPLETED` | **None** | **No** |
| **13** | **Multi-Failure Cascade** | Obstacle present + Laya provider outage | `DECISION_PROPOSAL` | `DETERMINISTIC_RECOVERY` | `COMPLETED` | **None** | **No** |

---

## 6. Live Validation Benchmarks (Real Models + MuJoCo Physics)

Following offline verification, four live closed-loop benchmarks were performed against the live Franka Panda MuJoCo environment:

### Live Validation A — Nominal Closed-Loop Run
- **Configuration**: Real hosted `Nemotron-3-Ultra-550b-a55b` + Real local Laya (`0.3.25`) + MuJoCo Panda (clean tabletop, zero obstacles).
- **Execution Path**: `APPROACH` $\to$ `GRASP` $\to$ `REPOSITION` (clearance lift) $\to$ `MOVE` (transit) $\to$ `RELEASE`.
- **Result**: Task `COMPLETED`, placement error $= 0.0534$ m ($\le 0.060$ m tolerance).

### Live Validation B — PATH_BLOCKED Disturbance Recovery
- **Configuration**: Real Nemotron + Real Laya + MuJoCo Panda with dynamic obstacle injection at Step 2.
- **Execution Path**: `APPROACH` $\to$ `GRASP` $\to$ Obstacle injected $\to$ Direct path rejected $\to$ Laya proposes `REPOSITION` $\to$ Elevated clearance transit $\to$ `RELEASE`.
- **Result**: Task `COMPLETED`, recovery occurred $= \text{True}$, placement error $= 0.0527$ m.

### Live Validation C — Live Laya Provider Outage with Fallback
- **Configuration**: Real Nemotron + Simulated Laya Timeout Provider + MuJoCo Panda with obstacle.
- **Execution Path**: Laya query times out across steps; Tier 2 deterministic fallback policy steps in at each cycle, passes `SarthiDecisionEngine` constraints, and drives recovery to goal.
- **Result**: Task `COMPLETED`, fallback engaged on 5 steps, placement error $= 0.0545$ m, **zero AI bypass**.

### Live Validation D — Live Deterministic Rejection
- **Configuration**: Franka holding payload at tabletop height with active obstacle barrier.
- **Test**: Injection of direct ground `MOVE` candidate with artificially high confidence ($p=0.99$).
- **Result**: `ConstraintValidator` rejected candidate (`BlockedPath`, clearance $< 0.082$ m). `SarthiDecisionEngine` selected `STOP`. **Unsafe candidate was never executed.**

---

## 7. Safety Metrics & Invariant Audit

| Safety Metric | Target | Measured Result | Compliance |
|---|:---:|:---:|:---:|
| **Total Scenarios Evaluated** | 13 | 13 | PASS |
| **Total Passed** | 13 | 13 | PASS |
| **Unsafe Physical Executions** | **0** | **0** | **100% PASS** |
| **AI-to-Robot Bypasses** | **0** | **0** | **100% PASS** |
| **Direct Unvalidated Candidates Executed** | **0** | **0** | **100% PASS** |
| **Secret Sanitization Status** | 100% Sanitized | 100% Sanitized | PASS |

---

## 8. Telemetry & Provenance

The complete per-scenario telemetry audit record is serialized at:
[`records/v3_10_resiliency.json`](file:///d:/Projects/SĀRTHI/records/v3_10_resiliency.json)

Every record captures:
- Scenario identifier & failure classification
- Injected condition and exact detection stage
- Candidate validation outcome and rejection reasons
- Physical execution and verification results
- Provenance trail tracking candidate origin and fallback strategy

All authorization credentials, bearer tokens (`Bearer`), and keys (`sk-`, `nbf_`) are automatically scrubbed via `_sanitize_data()`.

---

## 9. Known Limitations

The following technical boundaries are explicitly noted:
1. **Network Variability**: Hosted Nemotron response latency depends on Nebius cloud network status. Timeouts are caught and result in graceful cognitive aborts without physical motion.
2. **Local Laya Concurrency**: Local Laya server performance depends on host CPU/GPU scheduling. When Laya latency exceeds threshold, deterministic fallback seamlessly takes over.
3. **Simulation Boundary**: Validations were conducted within MuJoCo physics. While joint limits, contact friction cones, and inertia matrices reflect the Franka Emika Panda, physical motor backlash, gear compliance, and real-world perception noise require real-world calibration.
4. **Stopping Dynamics**: Tier 3 `SAFE STOP` commands immediate joint velocity nullification and position holding; dynamic momentum dissipation is governed by PD controller gains.
