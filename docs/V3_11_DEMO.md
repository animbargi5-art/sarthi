# SĀRTHI V3-11: Visual Demonstration & Telemetry Logging Architecture

## 1. Executive Summary

SĀRTHI V3-11 establishes a clean, judge-ready visual demonstration and telemetry presentation layer. Building upon the verified cognitive-to-physical closed-loop (V3-6), latency instrumentation (V3-7), 12-dimensional physics validation (V3-8), reproducibility framework (V3-9), and failure resiliency fallback suite (V3-10), V3-11 provides the definitive presentation and evidence interface.

**Strict Scope & Safety Invariants**:
1. **Zero Core Redesign**: `SarthiDecisionEngine` remains the sole physical authority.
2. **Strict Authority Hierarchy**:
   - `AI PROPOSAL — NON-AUTHORITATIVE`: LLM / RL models (Nemotron, Laya) propose candidates only.
   - `DETERMINISTIC PHYSICAL AUTHORITY`: `SarthiDecisionEngine` and `ConstraintValidator` possess sole validation authority.
   - `PHYSICAL EXECUTION — VALIDATED ACTION ONLY`: MuJoCo Franka Panda acts solely upon mathematically validated actions.
3. **Replay Mode Integrity**: Replay mode is explicitly labeled `REPLAY MODE — OFFLINE DEMONSTRATION` and never claims to be live execution.
4. **UI Isolation**: The visual dashboard is strictly a presentation and observation layer. It contains zero endpoints or code paths capable of commanding robot motors or bypassing safety constraints.
5. **Security**: Telemetry and audit trails are 100% sanitized and free of API keys, bearer tokens, or secrets.

---

## 2. Primary Demonstration Sequence

The demonstration traces an end-to-end task requiring physical obstacle avoidance:
**"Move the red object to the blue target."**

The sequence generates a chronological, 19-step judge-readable telemetry stream:

| Step | Event Code | Authority Classification | Description |
| :--- | :--- | :--- | :--- |
| **01** | `[01] TASK_RECEIVED` | `HUMAN OPERATOR` | Operator commands natural language objective. |
| **02** | `[02] NEMOTRON_TASK_UNDERSTANDING` | `AI COGNITIVE UNDERSTANDING (NON-AUTHORITATIVE)` | Nemotron parses entities, target object, and target zone. |
| **03** | `[03] WORLD_STATE_OBSERVED` | `PHYSICAL SENSING (MUJOCO RUNTIME)` | Initial scene observed: robot pose, object poses, target location. |
| **04** | `[04] LAYA_DECISION` | `AI PROPOSAL — NON-AUTHORITATIVE` | Local Laya RL selects candidate option (`APPROACH`) with probabilities. |
| **05** | `[05] CANDIDATE_INJECTED` | `CANDIDATE ACTION INJECTOR` | `CandidateActionInjector` formats bounded action primitive. |
| **06** | `[06] DETERMINISTIC_VALIDATION_ACCEPTED` | `DETERMINISTIC PHYSICAL AUTHORITY` | `SarthiDecisionEngine` scores and accepts candidate. |
| **07** | `[07] ACTION_EXECUTED` | `PHYSICAL EXECUTION — VALIDATED ACTION ONLY` | Panda articulations execute approach trajectory. |
| **08** | `[08] ACTION_VERIFIED` | `PHYSICAL STATE VERIFICATION` | IK/End-effector verifies arrival at standoff position. |
| **09** | `[09] DISTURBANCE_DETECTED` | `ENVIRONMENTAL DISTURBANCE` | Dynamic obstacle `blocking_barrier_01` appears in corridor. |
| **10** | `[10] MOVE_REJECTED_BLOCKED_PATH` | `DETERMINISTIC PHYSICAL AUTHORITY` | Direct horizontal `MOVE` is evaluated and **REJECTED** by `BlockedPath`. |
| **11** | `[11] RECOVERY_OPTIONS_GENERATED` | `DECISION CONTEXT SYNTHESIZER` | Bounded question generated with recovery choices: `["REPOSITION", "STOP"]`. |
| **12** | `[12] LAYA_SELECTED_REPOSITION` | `AI PROPOSAL — NON-AUTHORITATIVE` | Local Laya evaluates context and selects `REPOSITION` (elevate payload). |
| **13** | `[13] REPOSITION_VALIDATED` | `DETERMINISTIC PHYSICAL AUTHORITY` | Engine verifies lift altitude ($z=0.35\text{m}$) safely clears obstacle. |
| **14** | `[14] REPOSITION_EXECUTED` | `PHYSICAL EXECUTION — VALIDATED ACTION ONLY` | Panda elevates payload above obstacle height. |
| **15** | `[15] MOVE_VALIDATED` | `DETERMINISTIC PHYSICAL AUTHORITY` | Elevated transit trajectory verified clear of collision corridor. |
| **16** | `[16] MOVE_EXECUTED` | `PHYSICAL EXECUTION — VALIDATED ACTION ONLY` | Robot transits payload across elevated plane to target zone. |
| **17** | `[17] RELEASE_VERIFIED` | `PHYSICAL STATE VERIFICATION` | Gripper releases; payload settles stably on surface. |
| **18** | `[18] FINAL_PLACEMENT_VERIFIED` | `PHYSICAL VERIFICATION` | Final placement error verified within $0.08\text{m}$ tolerance. |
| **19** | `[19] TASK_COMPLETED` | `SYSTEM SUPERVISOR` | Full physical task declared successfully accomplished. |

---

## 3. Dynamic Disturbance & Recovery Flow

The core narrative demonstrates deterministic authority protecting the robot when a dynamic obstacle is introduced:

```
DYNAMIC DISTURBANCE: Obstacle in corridor
                  │
                  ▼
DIRECT MOVE ──► DETERMINISTIC VALIDATION ──► REJECTED (BlockedPath: corridor obstruction)
                                                   │
                                                   ▼
                                         RECOVERY OPTIONS GENERATED
                                              [REPOSITION, STOP]
                                                   │
                                                   ▼
                                         LAYA BOUNDED PROPOSAL
                                         "REPOSITION" (p = 0.56)
                                      [AI PROPOSAL - NON-AUTHORITATIVE]
                                                   │
                                                   ▼
                                         DETERMINISTIC VALIDATION
                                                 ACCEPTED
                                     [DETERMINISTIC PHYSICAL AUTHORITY]
                                                   │
                                                   ▼
                                         PHYSICAL RECOVERY LIFT
                                            z = 0.35m clearance
                                     [VALIDATED PHYSICAL EXECUTION]
                                                   │
                                                   ▼
                                         TRANSIT & TASK COMPLETED
```

---

## 4. Visual Dashboard Interface

The demonstration interface is implemented as a lightweight, dependency-free local web application (`frontend/index.html`) served via FastAPI (`backend/app/demo/server.py`).

### Key Dashboard Panels:
1. **Header & System Status**:
   - Connection indicators for Nemotron, Local Laya, SarthiDecisionEngine, and MuJoCo Panda.
   - Mode badge dynamically displaying `LIVE DEMONSTRATION` or `REPLAY MODE — OFFLINE DEMONSTRATION`.
2. **Human Command**:
   - Operator prompt display with cognitive extraction indicators.
3. **Current World State**:
   - Live telemetry of end-effector $(x, y, z)$, object coordinates, target zone, and gripper status.
4. **AI Proposal Panel (`AI PROPOSAL — NON-AUTHORITATIVE`)**:
   - Model name, bounded question, option probabilities (rendered with visual progress bars), and answer confidence.
5. **Deterministic Validation Panel (`DETERMINISTIC PHYSICAL AUTHORITY`)**:
   - Candidate action, validation verdict (`ACCEPTED` / `REJECTED`), scoring, active constraints, and safety rejection reasons.
6. **Physical Execution Panel (`PHYSICAL EXECUTION — VALIDATED ACTION ONLY`)**:
   - Verification status, trajectory progression, and execution diagnostics.
7. **Dynamic Disturbance & Recovery Flow**:
   - Dedicated flow diagram visualizing the rejection of direct transit and subsequent elevated recovery.
8. **Chronological Event Stream**:
   - Interactive stream of all 19 judge-readable telemetry events with timestamps and authority tags.
9. **Failure / Fallback Suite**:
   - Dedicated panel showing V3-10 resilient fallback behavior (e.g. Laya timeout fallback).

---

## 5. Demonstration Modes & Execution

### Live Mode
Connects to real local Laya (`http://127.0.0.1:8000`), real Nemotron via Nebius Token Factory, and real MuJoCo Franka Panda 7-DOF physics:
```bash
python scripts/run_v3_11_demo.py --mode live
```

### Replay Mode
Plays back recorded telemetry from `records/v3_11_demo_telemetry.json` without requiring active neural endpoints or simulation spin-up:
```bash
python scripts/run_v3_11_demo.py --mode replay
```
*Note*: Replay mode explicitly tags all records and visual UI elements with `REPLAY MODE — OFFLINE DEMONSTRATION` to ensure transparency.

### Local Visual Dashboard
Launches the web dashboard at `http://127.0.0.1:8080`:
```bash
python scripts/run_v3_11_demo.py --serve --port 8080
```

---

## 6. Security & Audit Verification

- **Sanitization Engine**: The `_sanitize_data` recursive scrubber automatically redacts:
  - `sk-*` API keys
  - `nbf_*` Nebius tokens
  - `Bearer *` authorization headers
  - Secret keys, passwords, and tokens in any dictionary key or value.
- **Audit Verification**:
  - `records/v3_11_demo_telemetry.json` audited: 0 secrets detected.
  - UI code audited: No endpoints exist that accept direct motor commands or bypass validation.
