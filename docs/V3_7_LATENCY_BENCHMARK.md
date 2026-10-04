# SĀRTHI V3-7 — Decision Latency Instrumentation & Benchmarking Report

## 1. Overview & Objectives
SĀRTHI Phase V3-7 instruments and benchmarks the complete cognitive-to-physical decision pipeline.
The goal is to capture high-resolution, sub-millisecond, monotonic timing across every phase of the V3 architecture:
```
Human Instruction
       ↓ [tau_nemotron]
Nemotron Task Understanding
       ↓ [tau_context]
DecisionContext Synthesis
       ↓ [tau_jev]
Local Laya Bounded Proposal
       ↓ [tau_validation]
SarthiDecisionEngine Deterministic Validation
       ↓ [tau_ik]
Differential Inverse Kinematics
       ↓ [tau_sim]
MuJoCo Physics Stepping & Joint Trajectory Control
       ↓ [tau_verify]
Physical State Outcome Verification
═══════════════════════════════════════════════
       ↓ [tau_total (Measured Independently)]
```

### Strict Architectural Boundaries Enforced
- **Measurement Only**: Zero optimizations were performed during V3-7.
- **Physics Unchanged**: Joint controllers, RK4 numerical integrator, contact dynamics, and scenario geometry remain identical.
- **Decision Logic Unchanged**: `SarthiDecisionEngine` remains the sole physical authority.
- **Secret-Free**: Zero API tokens, authentication headers, or credentials stored or logged.

---

## 2. Latency Marker Definitions

| Marker | Measurement Boundary | Implementation Method |
| :--- | :--- | :--- |
| **`tau_nemotron`** | From submission of human instruction to structured `TaskUnderstanding` reception. | `time.perf_counter()` around `TaskUnderstandingService.understand()` |
| **`tau_context`** | From observed MuJoCo `WorldState` to compact `DecisionContext` & `DecisionQuestion` synthesis. | `time.perf_counter()` around `build_bounded_question()` & `DecisionContextBuilder` |
| **`tau_jev`** | From `DecisionQuestion` serialization to reception and validation of bounded `JevDecision`. | `time.perf_counter()` around `LayaDecisionProvider.ask()` (`/v1/systemone`) |
| **`tau_validation`** | From `CandidateAction` injection through `SarthiDecisionEngine` and `ConstraintValidator` feasibility checks. | `time.perf_counter()` around `CandidateActionInjector` & engine candidate evaluation |
| **`tau_ik`** | Computation time required by damped-least-squares (DLS) Jacobian IK to compute arm joint positions. | `time.perf_counter()` within `SarthiMuJoCoRuntime.execute_action()` around `solve_ik()` |
| **`tau_sim`** | Physical execution of multi-step joint trajectory interpolation and gripper actuation in MuJoCo. | `time.perf_counter()` within `SarthiMuJoCoRuntime.execute_action()` around physics ticks |
| **`tau_verify`** | Post-execution physical verification of contact kinematics, gripper clamping, and spatial tolerance. | `time.perf_counter()` around internal verifier and `verify_action_result()` |
| **`tau_total`** | Complete end-to-end wall-clock duration of the closed-loop decision transaction. | Outer transaction timer measured independently via monotonic clock |

---

## 3. Hardware & Execution Environment
- **Operating System**: Windows 11 Home (Build 10.0.26100)
- **Architecture**: x86_64
- **Runtime Environment**: Python 3.10.11 Virtual Environment (`.venv`)
- **Physics Engine**: MuJoCo 3.14.0 (CPU / Runge-Kutta 4th order integrator)
- **Manipulator**: Franka Emika Panda 7-DOF Robotic Arm + 2-Finger Gripper
- **Scenario**: `tabletop_pick_and_place_mvp.xml` (Tabletop scene with red object, blue target, and dynamic obstacle)
- **Cognitive Model**: NVIDIA Nemotron-3-Ultra-550b-a55b via Nebius Token Factory (`https://api.tokenfactory.nebius.com/v1`)
- **Fast Decision Model**: Local Laya 0.3.25 (`http://127.0.0.1:8000`, `typed-decisions` on CPU)

---

## 4. Benchmark Results

### Benchmark A: Offline Deterministic Overhead (5 Full Cycles)
Measures deterministic orchestration overhead with mock AI providers and real MuJoCo physics.
Eliminates all network variance to isolate pure computational and physical control costs.

| Marker | Count | Min (ms) | Mean (ms) | Median (ms) | Max (ms) | p95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`tau_nemotron`** | 5 | 0.111 | 0.157 | 0.124 | 0.264 | 0.246 |
| **`tau_context`** | 5 | 0.889 | 1.066 | 1.043 | 1.354 | 1.309 |
| **`tau_jev`** | 5 | 0.048 | 0.053 | 0.053 | 0.062 | 0.060 |
| **`tau_validation`** | 5 | 1.217 | 1.319 | 1.237 | 1.537 | 1.504 |
| **`tau_ik`** | 5 | 20.479 | 24.890 | 25.468 | 30.385 | 29.510 |
| **`tau_sim`** | 5 | 140.303 | 143.209 | 143.350 | 146.224 | 145.684 |
| **`tau_verify`** | 5 | 0.383 | 0.489 | 0.508 | 0.584 | 0.575 |
| **`tau_total`** | 5 | 167.119 | 174.286 | 175.010 | 179.314 | 178.472 |

**Key Takeaway**: Deterministic validation overhead (`tau_validation`) is only **~1.24 ms**. Full offline cycle duration is dominated by physical trajectory execution (`tau_sim` ~143 ms) and numerical IK (`tau_ik` ~25 ms).

---

### Benchmark B: Local Laya Fast Bounded Decisions (20 Queries)
Measures latency of 20 discrete decision queries sent directly to local Laya server (`/v1/systemone`) across varying robotics prompts (`APPROACH`, `GRASP`, `REPOSITION`, `MOVE`, `RELEASE`).

| Marker | Count | Min (ms) | Mean (ms) | Median (ms) | Max (ms) | p95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`tau_jev`** | 20 | 263.766 | 304.287 | 302.109 | 361.286 | 356.058 |

**Key Takeaway**: Local Laya exhibits highly predictable, low-latency bounded choices on CPU with a median latency of **302.11 ms** and a 95th percentile under **356.06 ms**.

---

### Benchmark C: Full Live Cognitive-to-Physical V3 Pipeline (5 Cycles)
Measures the complete real system:
`Human Instruction` $\rightarrow$ `Real Nebius Nemotron` $\rightarrow$ `Real Local Laya` $\rightarrow$ `CandidateActionInjector` $\rightarrow$ `SarthiDecisionEngine` $\rightarrow$ `MuJoCo Franka Panda` $\rightarrow$ `PATH_BLOCKED dynamic obstacle recovery` $\rightarrow$ `Verification`.

| Marker | Count | Min (ms) | Mean (ms) | Median (ms) | Max (ms) | p95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`tau_nemotron`** | 5 | 1328.910 | 3530.179 | 2758.214 | 7602.175 | 6955.573 |
| **`tau_context`** | 5 | 0.281 | 0.841 | 0.963 | 1.064 | 1.046 |
| **`tau_jev`** (sum per task) | 5 | 269.064 | 1487.541 | 1719.383 | 1992.195 | 1958.604 |
| **`tau_validation`** (sum per task) | 5 | 0.212 | 1.743 | 2.047 | 2.359 | 2.329 |
| **`tau_ik`** (sum per task) | 5 | 0.000 | 25.871 | 31.865 | 34.171 | 33.791 |
| **`tau_sim`** (sum per task) | 5 | 1.001 | 141.466 | 169.030 | 196.607 | 194.658 |
| **`tau_verify`** (sum per task) | 5 | 0.045 | 0.394 | 0.442 | 0.556 | 0.548 |
| **`tau_total`** | 5 | 3381.951 | 5191.327 | 4641.275 | 9514.630 | 8603.346 |

**Key Takeaways**:
1. **Cognitive Tier Latency**: `tau_nemotron` has a median of **2.76 s** (min 1.33 s, max 7.60 s due to cloud WAN inference).
2. **Fast Bounded Decisions**: `tau_jev` averages **~300 ms per step** (totaling ~1.72 s across 5 steps).
3. **Deterministic Safety Validation**: Deterministic physical validation adds only **~2.05 ms** total per 5-step task.
4. **Physical Actuation**: MuJoCo physics stepping and trajectory control takes **~169 ms** total.

---

## 5. Limitations & Future Work
1. **Network WAN Variance**: Cloud-hosted LLM endpoints (Nebius Token Factory) introduce non-deterministic network latency (ranging between 1.3s and 7.6s) depending on internet congestion and queue depth.
2. **CPU-Bound Inference**: Local Laya runs in CPU fallback mode on the host laptop. GPU inference (CUDA / TensorRT) could reduce `tau_jev` below 50 ms in future phases.
3. **No Optimizations Made**: As strictly mandated by V3-7, zero model pruning, caching, multi-threading, or pipeline pipelining was implemented. Measurement was strictly non-intrusive.

---

## 6. Verification & Audit Trail
- Structured benchmark data file: [`records/v3_7_latency_benchmark.json`](file:///d:/Projects/SĀRTHI/records/v3_7_latency_benchmark.json)
- Full test suite: **298/298 tests passing** (`test_*.py`)
- Zero secrets detected: Validated against regex patterns for API keys, tokens, and authorization credentials.
