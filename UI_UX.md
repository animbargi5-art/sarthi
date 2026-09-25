# SĀRTHI — Operator Dashboard & Telemetry Console UI/UX Specification

## 1. Design Vision & Principles

The **SĀRTHI Operator Dashboard** is an enterprise-grade mission control console for Physical AI operations. It is designed to provide robotics operators, automation engineers, and researchers with real-time situational awareness, sub-second disturbance insights, transparent cognitive reasoning visibility, and interactive disturbance injection controls.

The UI emphasizes:
- **Low Cognitive Overhead:** High-contrast data hierarchy for rapid anomaly detection.
- **Physical Grounding:** Synchronized 3D trajectory overlays and force/torque sparklines.
- **Explainable AI (XAI):** Real-time visibility into NVIDIA Nemotron's deliberation stream powered by Nebius Token Factory.

---

## 2. Layout & Information Architecture

```text
+-------------------------------------------------------------------------------------------------------+
|  SĀRTHI Mission Control  |  Nebius: CONNECTED (284ms)  |  Isaac Sim: STREAMING (100Hz)  | [ E-STOP ] |
+-------------------------------------------------------------------------------------------------------+
|  LEFT PANE (300px)         |  CENTER PANE (Flexible Viewport)        |  RIGHT PANE (360px)            |
|  Telemetry & Kinematics    |  Live Simulation & State Machine        |  Cognitive Reasoner & Controls |
|                            |                                         |                                |
|  - Joint Angles (q1-q7)    |  +-----------------------------------+  |  NVIDIA Nemotron Stream:       |
|    [===|====] 42.1 deg     |  |                                   |  |  "Detected lateral slip:      |
|    [==|=====] -12.4 deg    |  |   Isaac Sim 3D Viewport           |  |   Normal force dropped to 4N.  |
|                            |  |   (Live RGB / Trajectory Overlay) |  |   Generating adaptive grip     |
|  - 6-Axis F/T Telemetry    |  |                                   |  |   compensation primitive..."   |
|    Fx: [~/\_/\_] 2.1 N     |  +-----------------------------------+  |                                |
|    Fy: [___/\_] 18.4 N (!) |                                         |  Inference Latency: 284 ms     |
|    Fz: [------] 45.2 N     |  State: [ DISTURBANCE: SLIP DETECTED ]  |  Tokens Generated: 162         |
|                            |                                         |                                |
|  - End-Effector Pose (XYZ) |  Recovery State Machine:                |  Disturbance Injection Panel:  |
|    X: 0.421 m | Y: -0.112 m|  [Nominal] -> (Disturbed) -> [Recover]  |  [ Trigger Slip ] [ Force Bump]|
|    Z: 0.654 m              |                                         |  [ +200% Mass ]   [ Obstacle ] |
+-------------------------------------------------------------------------------------------------------+
|  BOTTOM DOCK (Full Width, Height: 180px) - Incident Audit Timeline & Event Stream                    |
|  19:42:01.120 [NOMINAL] Waypoint 4 reached | 19:42:02.400 [WARN] F/T Y-delta exceeded threshold (18.4N)|
|  19:42:02.415 [COGNITIVE] Nebius query dispatched | 19:42:02.699 [RECOVERY] Regrasp primitive executed|
+-------------------------------------------------------------------------------------------------------+
```

---

## 3. Component Specifications

### 3.1 Header Bar
- **System Branding:** Clean, minimalist typography.
- **Connection Indicators:**
  - `Nebius Token Factory`: Green badge with live ping (e.g., `284 ms`). Turns amber if $> 400\text{ ms}$, red if disconnected.
  - `Isaac Sim Bridge`: Green badge with packet frequency (e.g., `100 Hz`).
- **Emergency Stop Button:** High-prominence, guarded hardware/software kill switch that instantly commands zero velocity and engages mechanical brakes.

### 3.2 Left Pane: Telemetry & Joint Sensor Array
- **Joint State Visualizer:** Real-time dual-slider gauges showing current position versus soft/hard limits for all 7 manipulator joints.
- **6-Axis F/T Sparklines:** High-refresh canvas charts rendering the last 3 seconds of raw and filtered force/torque vectors ($F_x, F_y, F_z, \tau_x, \tau_y, \tau_z$).
- **End-Effector Coordinates:** Numerical XYZ coordinates with 3-axis velocity vectors.

### 3.3 Center Pane: Primary Simulation & State Machine View
- **Simulation Viewport:** Low-latency WebRTC/WebSocket video canvas displaying Isaac Sim camera stream with dynamic bounding box and planned trajectory overlays.
- **State Badge:** Prominent status indicator reflecting the active system state:
  - `NOMINAL` (Green)
  - `DISTURBANCE_DETECTED` (Rose)
  - `COGNITIVE_DELIBERATING` (Purple / Violet)
  - `RECOVERING` (Cyan)
  - `EMERGENCY_STOP` (Amber / Red)
- **State Machine Progress:** Step indicator illustrating transitions between nominal tracking, freeze hold, cognitive planning, safety check, and recovery execution.

### 3.4 Right Pane: NVIDIA Nemotron Cognitive Stream & Testing Controls
- **Cognitive Thought Stream:** Card showing NVIDIA Nemotron's step-by-step diagnostic breakdown:
  - Detected Failure Classification (e.g., `SLIP_ROTATIONAL`, `COLLISION_DYNAMIC`).
  - Physical Rationale (Natural language explanation of the root cause).
  - Selected Recovery Primitive and Parametric Overrides.
  - Performance Metrics: Roundtrip inference latency, prompt token count, completion token count.
- **Disturbance Injection Testing Harness:** Quick-action buttons allowing operators/judges to inject controlled disturbances into Isaac Sim:
  - `Trigger Micro-Slip`: Drops surface friction coefficient by 70%.
  - `Force Impulse (25N)`: Applies an instantaneous lateral impulse.
  - `Mass Surge (+2.0kg)`: Instantly alters payload inertia.
  - `Spawn Obstacle`: Introduces a dynamic obstacle directly into the active trajectory.

### 3.5 Bottom Dock: Incident Timeline & Audit Log
- Chronological, searchable, and exportable log recording every disturbance trigger, LLM deliberation, safety evaluation result, and recovery outcome.

---

## 4. Design Tokens & Styling Guide

| Token Type | Value / Hex | Usage |
| :--- | :--- | :--- |
| **Surface Background** | `#0B0F19` (Deep Charcoal Slate) | Main window background |
| **Card / Panel Surface**| `#111827` (Card Charcoal) | Panes, data cards, toolbars |
| **Panel Border** | `#1F2937` (Subtle Slate Border) | Structural divider lines |
| **Primary Accent** | `#06B6D4` (Cyan) | Active selections, trajectory paths, recovery states |
| **Cognitive / AI Accent**| `#8B5CF6` (Nebius Purple/Violet) | Nemotron thought stream, inference badges |
| **Success State** | `#10B981` (Emerald Green) | Nominal execution, recovered confirmation |
| **Warning State** | `#F59E0B` (Amber) | High torque warning, friction marginality |
| **Error / Disturbance** | `#F43F5E` (Rose Red) | Active physical disturbance, safety threshold breach |
| **Primary Typography** | `Inter`, `Roboto`, `system-ui` | UI headers, metrics, labels |
| **Monospace Typography**| `JetBrains Mono`, `Fira Code` | Coordinates, telemetry readings, log streams |

---

## 5. Accessibility & Performance Considerations

- **Color Contrast:** All text elements adhere to WCAG AAA standards against dark backgrounds.
- **High-Refresh Rendering:** Telemetry canvas charts are decoupled from React/DOM re-renders using `requestAnimationFrame` to ensure a consistent 60 FPS UI experience.
- **Resilient Network Handling:** Automatic UI reconnection fallback if WebSocket telemetry frames drop.
