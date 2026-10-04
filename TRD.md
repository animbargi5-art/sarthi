# SĀRTHI — Technical Requirements Document (TRD)

**Project Name:** SĀRTHI  
**Tagline:** An adaptive Physical AI architecture for context-aware, responsibility-driven robotic decision-making under changing physical conditions.  
**Target Environment:** Nebius × NVIDIA Global AI Hackathon 2026  
**Document Version:** Version: 3.0  
**Status:** Draft — V3 Architecture Update  

---

## 1. Safety Architecture & Core Invariant

The technical implementation of SĀRTHI V3 is anchored on strict separation between cognitive candidate selection and deterministic physical authorization:

> [!IMPORTANT]
> **Core System Invariant:**  
> **"AI models propose or select bounded semantic candidates; deterministic software validates physical feasibility; only validated actions reach the robot controller."**  
> **"Jev must never bypass deterministic safety validation."**

No AI model—neither large language models nor fast decision networks—possesses direct motor actuation authority, raw trajectory access, or low-level simulation controller interfaces.

---

## 2. System Technology Stack

| Subsystem | Component / Framework | Specification / Version | Architectural Role |
| :--- | :--- | :--- | :--- |
| **Cognitive Task Understanding** | NVIDIA Nemotron via Nebius Token Factory | `nvidia/Nemotron-3-Ultra-550b-a55b` | Natural-language instruction parsing, goal/entity extraction, semantic plan generation |
| **Fast Bounded Decision** *(Planned)* | Jev Decision Layer | Abstract Provider Interface (`JevDecisionProvider`) | Rapid candidate action ranking, discrete recovery strategy selection, bounded decision scoring |
| **Deterministic Action Authority** | SĀRTHI Decision Engine | Native Python 3.10+ Architecture | Sole physical action authority; constraint validation; collision clearance checks; final authorization |
| **Physics Simulation Engine** | MuJoCo | `mujoco >= 3.1.0` | Rigid-body dynamics, contact manifold computation, collision geometry, physical actuation |
| **Robotic Manipulator Asset** | Franka Emika Panda | Sourced from MuJoCo Menagerie (Apache 2.0) | 7-DOF articulated arm + parallel-jaw gripper, joint limits, torque actuation |
| **Numerical Articulation** | SarthiMuJoCoArticulation | Damped-Least-Squares (DLS) IK | Differential kinematics, pseudo-inverse calculation, joint-limit clipping, trajectory stepping |
| **Physical State Extraction** | SarthiMuJoCoStateReader | Direct `mjData` extraction | Live coordinate translation to canonical Pydantic `WorldState` |
| **Data Contracts & Schemas** | Pydantic v2 | Python 3.10+, strict typing | Type-safe validation for instructions, telemetry, decisions, and physics verification |
| **Telemetry & Verification** | SĀRTHI Telemetry Logger | JSON Lines / Structured JSON (`records/`) | Audit trail recording, physical state verification, latency instrumentation |

---

## 3. Cognitive Tier: Nebius Token Factory & NVIDIA Nemotron

### 3.1 Provider Specification
SĀRTHI interfaces with **Nebius Token Factory** using an OpenAI-compatible REST API client to query NVIDIA Nemotron:
- **Base URL:** `https://api.tokenfactory.nebius.com/v1` (configurable via `NEBIUS_BASE_URL`)
- **Model Identifier:** `nvidia/Nemotron-3-Ultra-550b-a55b` (configurable via `NEBIUS_MODEL`)
- **Authentication:** Bearer token via `NEBIUS_API_KEY` environment variable.
- **Client Implementation:** `NebiusNemotronProvider` implements the abstract `ModelProvider` boundary.

### 3.2 Cognitive Data Contract
The provider receives natural-language input and outputs a structured `TaskUnderstanding` object:
```json
{
  "task_type": "PICK_AND_PLACE",
  "target_object": "red_object",
  "destination_zone": "blue_target",
  "required_actions": ["APPROACH", "GRASP", "MOVE", "RELEASE"],
  "semantic_constraints": {
    "preserve_orientation": true,
    "max_approach_velocity": 0.15
  },
  "raw_reasoning": "The user requested moving the red object to the blue target."
}
```

### 3.3 Strict Invariants & Non-Execution Guarantees
- Nemotron output is strictly semantic; it cannot command robot joints or modify `WorldState`.
- `required_actions` represents an unvalidated high-level sequence; every action must be independently evaluated by the SĀRTHI Decision Engine prior to dispatch.
- Client timeout is enforced (default $30.0\text{ s}$ for initial parsing; configurable). On failure or timeout, an explicit task-level failure is returned without moving the arm.

---

## 4. Fast Decision Tier: Jev Provider Boundary (Planned)

### 4.1 Architectural Role & Scope
**Jev** is incorporated as a fast structured decision layer designed for bounded, discrete questions where high-throughput scoring is advantageous over full generative re-planning.

Jev does **NOT**:
- Directly command robot motors or joint trajectories.
- Replace or bypass the deterministic SĀRTHI Decision Engine.
- Perform open-ended, unconstrained task synthesis.

### 4.2 Abstract Adapter Interface
To ensure modularity and prevent vendor lock-in, Jev is accessed strictly behind an abstract interface:
```python
from abc import ABC, abstractmethod
from typing import Optional

class JevDecisionProvider(ABC):
    @abstractmethod
    async def evaluate_decision(
        self, question: "DecisionQuestion", context: "DecisionContext"
    ) -> "DecisionResponse":
        """Evaluates a bounded decision question against compact context."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the provider is healthy and authenticated."""
        pass
```

### 4.3 Bounded Decision Formats
Jev decision queries are restricted to small, well-defined questions:

1. **Recovery Strategy Selection:**
   - *Question:* `"Which valid recovery strategy should be considered?"`
   - *Options:* `["REPOSITION", "REPLAN", "REGRASP", "STOP"]`
2. **Candidate Action Selection:**
   - *Question:* `"Which candidate action is most appropriate for the current state?"`
   - *Options:* `["APPROACH", "GRASP", "MOVE", "RELEASE", "REPOSITION", "STOP"]`
3. **Recovery Necessity:**
   - *Question:* `"Does the current situation require recovery?"`
   - *Options:* `["RECOVER", "CONTINUE", "STOP"]`
4. **Target Disambiguation:**
   - *Question:* `"Which candidate target is most relevant?"` (when multiple candidates exist).

### 4.4 Compact Decision Context
Raw simulation data is never dumped into Jev. The `DecisionContextBuilder` constructs a minimal payload containing only:
- Current robot phase and gripper status (`FREE`, `HOLDING`).
- Target entity and destination zone coordinates.
- Active constraint tags (e.g., `PATH_BLOCKED`, `MIN_CLEARANCE_VIOLATED`).
- Previous action outcome and error code.

### 4.5 Resiliency & Fallback Protocol
If the Jev provider encounters a timeout, connection failure, or invalid schema:
1. The request is logged as a decision provider fallback event.
2. The system seamlessly reverts to the **deterministic Decision Engine's native heuristic ranking**.
3. Execution proceeds without halting, ensuring Jev is **never a single point of failure**.

---

## 5. Deterministic Decision Tier: SĀRTHI Decision Engine

The deterministic **SĀRTHI Decision Engine** is the sole authority governing physical action approval:

```
[Candidate Action Proposal] (from TaskRunner, Jev, or Heuristic)
                     │
                     ▼
       [SarthiDecisionEngine.evaluate()]
                     │
       ┌─────────────┴─────────────┐
       ▼                           ▼
[Physical Kinematics]     [Active Constraints]
- Reachability             - Clearance Buffer (d >= 0.082m)
- Workspace limits         - Dynamic Obstacles (PATH_BLOCKED)
- Grasp preconditions     - Keepout Zones
       │                           │
       └─────────────┬─────────────┘
                     ▼
          [Deterministic Gate]
         /                    \
   [Approved]             [Rejected]
       │                       │
       ▼                       ▼
[Action Authorized]    [Trigger Recovery / REPOSITION]
```

### 5.1 Enforced Physical Constraints
- **Clearance Checking:** Evaluates minimum Euclidean distance between interpolated end-effector trajectory and active obstacle bounding volumes. (Validated benchmark: minimum distance `0.066 m` < required `0.082 m` triggers rejection).
- **Workspace Envelope:** Enforces $x, y, z$ bounding boxes ($x \in [0.1, 0.8]$, $y \in [-0.5, 0.5]$, $z \in [0.0, 0.8]$).
- **Grasp State Consistency:** Verifies gripper contact forces and object proximity before authorizing lifting or transit actions.

---

## 6. Execution & Verification Tier: MuJoCo Physics & Articulation

### 6.1 Platform & Model
- **Simulator:** MuJoCo (`mujoco >= 3.1.0`), headless and interactive viewer modes.
- **Robot Model:** Franka Emika Panda (7-DOF arm, 2-finger parallel gripper).
- **Kinematic Solver:** Damped-Least-Squares (DLS) differential inverse kinematics:
  $$\Delta q = J^T (J J^T + \lambda^2 I)^{-1} \Delta x$$
  with dynamic damping parameter $\lambda$ and joint-limit clamping.

### 6.2 State Extraction & Physical Verification
- `SarthiMuJoCoStateReader` performs zero-copy inspection of active `mjData` structures:
  - End-effector position (`data.site_xpos`) and orientation.
  - Joint positions (`data.qpos`) and velocities (`data.qvel`).
  - Contact manifolds (`data.contact`) for gripper-to-object and object-to-table contacts.
- Verification confirms post-action physical reality:
  - `GRASP_VERIFIED`: Confirms persistent inward normal force on target object.
  - `REPOSITION_VERIFIED`: Confirms elevation exceeds obstacle clearance altitude ($z \ge 0.33\text{ m}$).
  - `FINAL_PLACEMENT_VERIFIED`: Confirms final object position error $\le \text{tolerance}$ (validated: $0.0535\text{ m} \le 0.0600\text{ m}$).

---

## 7. Physics Validation Framework (New V3 Requirement)

The Physics Validation Framework explicitly tests and verifies physical simulation fidelity and repeatability across 12 distinct dimensions:

| ID | Test Dimension | Verification Method | Measured Metric | Validation Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **PV-1** | **Timestep Sensitivity** | Run identical scenario at $dt = 0.001\text{s}$, $0.002\text{s}$, $0.005\text{s}$. | Trajectory deviation ($\Delta x$) | Stable convergence within configurable parameter $\epsilon_{\text{dt}}$ |
| **PV-2** | **Joint-Limit Correctness** | Sweep commanded trajectories to joint extremes. | Commanded vs. physical joint angles | Zero joint-limit penetration across all 7 joints |
| **PV-3** | **Joint Velocity Limits** | Monitor numerical differentiation of $q(t)$ across motion. | $\max |\dot{q}_i(t)|$ | Never exceeds configured joint velocity bounds |
| **PV-4** | **EE Repeatability** | Execute nominal trajectory 20 times from identical initial state. | Final EE position variance ($\sigma_{\text{pos}}$) | Repeatability within configurable tolerance |
| **PV-5** | **Contact Stability** | Monitor normal forces during sustained grasp hold (1000 steps). | Force delta $\Delta F_N$, slippage $\Delta p_{\text{obj}}$ | Zero uncommanded object release or drift |
| **PV-6** | **Placement Repeatability**| Execute release and settling 20 times. | Distribution of final resting coordinates | Placement variance within configured tolerance |
| **PV-7** | **Collision/Clearance** | Sweep obstacle toward trajectory at varying distances. | Rejection flag vs. measured clearance | $100\%$ rejection when $d < d_{\text{clearance}}$ |
| **PV-8** | **Gravity & Settling** | Release object from $z = 0.25\text{m}$; record acceleration. | Measured descent acceleration | Matches configured MuJoCo gravity ($g = -9.81\text{ m/s}^2$) |
| **PV-9** | **IK Convergence** | Record residual $\|x_{\text{target}} - x_{\text{actual}}\|$ across IK steps. | Convergence iteration count & residual | Residual $\le \epsilon_{\text{IK}}$ within maximum iterations |
| **PV-10**| **Numerical Stability** | Inspect all state vectors during high-acceleration recovery. | Detection of `NaN` or `Inf` values | Zero non-finite floating-point states |
| **PV-11**| **Deterministic Replay** | Replay scenario with identical seed and inputs. | Bitwise or floating-point state delta | Zero divergence across repeated identical runs |
| **PV-12**| **Physics Regression** | Compare current run telemetry against baseline records. | Key landmark deviations | Metrics match or improve upon baseline records |

*Note: Where specific numerical thresholds are not established by current implementation, they are managed as configurable validation parameters.*

---

## 8. Decision Latency Measurement Specification

To rigorously evaluate the performance of the fast decision layer versus general LLM inference, SĀRTHI V3 instruments seven distinct latency markers:

```
[Command Input] ──t1──► [TaskUnderstanding] ──t2──► [DecisionContext]
                                                           │
                                                           t3 (Jev Bounded Decision)
                                                           ▼
[Physical Action] ◄──t5── [Action Approved] ◄──t4── [CandidateAction]
       │
       t6 (MuJoCo Physics Execution)
       ▼
[State Verification] ──t7──► [Outcome Verified / Recovery Cycle]
```

- **$\tau_{\text{task\_understanding}}$ ($t_2 - t_1$):** Human command to structured `TaskUnderstanding`.
- **$\tau_{\text{fast\_decision}}$ ($t_3$):** Bounded decision query to `JevDecision` response.
- **$\tau_{\text{validation}}$ ($t_4$):** Deterministic constraint evaluation and ranking.
- **$\tau_{\text{dispatch}}$ ($t_5$):** Trajectory synthesis and controller handoff.
- **$\tau_{\text{execution}}$ ($t_6$):** Physical execution and simulation stepping.
- **$\tau_{\text{verification}}$ ($t_7$):** `mjData` extraction and predicate verification.
- **$\tau_{\text{recovery\_cycle}}$:** End-to-end duration from disturbance trigger to validated recovery motion dispatch.

All latency metrics are recorded into `DecisionLatencyRecord` objects and persisted in telemetry.

---

## 9. Failure Handling & Resiliency Matrix

| Failure Mode | Detection Mechanism | Primary Mitigation | Fallback Action |
| :--- | :--- | :--- | :--- |
| **Nemotron Network Timeout** | HTTP client timeout ($> 30\text{s}$) | Log error; report failure to operator | Terminate task safely; zero robot actuation |
| **Nemotron Schema Violation** | Pydantic validation failure | Re-raise structured parsing exception | Safe termination without moving robot |
| **Jev Provider Timeout** | Async timeout ($> \text{timeout}_{\text{Jev}}$) | Log provider fallback event | Revert to deterministic heuristic candidate ranking |
| **Jev Malformed Output** | Schema validation failure | Reject candidate ranking | Revert to deterministic heuristic ranking |
| **Path Blockage (`PATH_BLOCKED`)** | Trajectory clearance $< 0.082\text{m}$ | Reject candidate `MOVE` | Select and execute `REPOSITION` recovery |
| **Grasp Slip / Failure** | State reader contact inspection | Detect missing contact normal force | Transition to `REGRASP` or safe hold |
| **IK Divergence** | Residual $> \text{threshold}$ | Clamp velocity; retry with damping $\lambda \uparrow$ | Abort action; safe position hold |
| **Workspace Breach** | Boundary filter check | Reject action before dispatch | Clamp to safe workspace envelope |

---

## 10. Security & Secret Management

- **Zero Secret Persistence:** `NEBIUS_API_KEY` and any future Jev authentication credentials must be read exclusively from runtime environment variables.
- **Sanitized Logging:** All loggers, telemetry serializers, and exception formatters automatically scrub authorization headers, Bearer tokens, and credential substrings.
- **Air-Gapped Simulation Interface:** MuJoCo runtime communicates through direct in-process Python C-bindings (`mujoco`); no open network ports or unauthenticated sockets are exposed.
