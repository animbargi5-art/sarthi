# SĀRTHI — Backend Schema & Interface Contracts

This document formalizes the data contracts, Pydantic v2 models, and JSON schemas governing inter-module communication across the **SĀRTHI** Physical AI architecture.

---

## 1. Schema Domain Overview

```text
[ NVIDIA Isaac Sim ]
        │  TelemetrySnapshot (100 Hz)
        ▼
[ Backend State Buffer ]
        │  DisturbanceEvent
        ▼
[ Nebius Token Factory ] ──(NVIDIA Nemotron)──► NemotronRecoveryStrategy
        │
        ▼
[ Deterministic Safety Validator ]
        │  SafetyValidationResult / ValidatedMotionCommand
        ▼
[ NVIDIA Isaac Sim Robot Actuator ]
```

---

## 2. Core Primitive Schemas

### 2.1 Spatial Primitives
```python
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class Vector3D(BaseModel):
    x: float = Field(..., description="X coordinate in meters or Newtons")
    y: float = Field(..., description="Y coordinate in meters or Newtons")
    z: float = Field(..., description="Z coordinate in meters or Newtons")

class Quaternion(BaseModel):
    w: float = Field(..., description="Scalar component")
    x: float = Field(..., description="Vector component X")
    y: float = Field(..., description="Vector component Y")
    z: float = Field(..., description="Vector component Z")

class Pose6D(BaseModel):
    position: Vector3D = Field(..., description="Cartesian 3D position (m)")
    orientation: Quaternion = Field(..., description="Unit quaternion orientation")
```

### 2.2 Sensor & Telemetry Models
```python
class JointTelemetry(BaseModel):
    joint_names: List[str] = Field(..., description="Names of active manipulator joints")
    positions: List[float] = Field(..., description="Joint positions in radians")
    velocities: List[float] = Field(..., description="Joint velocities in rad/s")
    torques: List[float] = Field(..., description="Joint efforts in Newton-meters (Nm)")

class ForceTorqueSensor(BaseModel):
    force: Vector3D = Field(..., description="3-axis contact force (N)")
    torque: Vector3D = Field(..., description="3-axis contact torque (Nm)")

class TelemetrySnapshot(BaseModel):
    timestamp_ns: int = Field(..., description="Simulation epoch timestamp in nanoseconds")
    robot_id: str = Field(default="sarthi_franka_01", description="Identifier of the robotic agent")
    joints: JointTelemetry = Field(..., description="Current joint kinematics and dynamics")
    end_effector_pose: Pose6D = Field(..., description="World-frame 6D pose of the end-effector")
    ft_sensor: ForceTorqueSensor = Field(..., description="Wrist 6-axis F/T readings")
    is_gripping: bool = Field(..., description="Grip engagement boolean")
    payload_mass_kg: float = Field(..., description="Estimated or nominal payload mass (kg)")
    slip_detected: bool = Field(default=False, description="Physical slip flag reported by contact sensors")
```

---

## 3. Disturbance & Anomaly Schemas

```python
class DisturbanceType(str, Enum):
    SLIP_TRANSLATIONAL = "SLIP_TRANSLATIONAL"
    SLIP_ROTATIONAL = "SLIP_ROTATIONAL"
    COLLISION_EXTERNAL = "COLLISION_EXTERNAL"
    EXCESSIVE_TORQUE = "EXCESSIVE_TORQUE"
    PAYLOAD_MASS_SHIFT = "PAYLOAD_MASS_SHIFT"
    TRAJECTORY_DEVIATION = "TRAJECTORY_DEVIATION"

class DisturbanceSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class DisturbanceEvent(BaseModel):
    event_id: str = Field(..., description="Unique UUID for the disturbance occurrence")
    timestamp_ns: int = Field(..., description="Detection timestamp in nanoseconds")
    disturbance_type: DisturbanceType = Field(..., description="Classified nature of the physical anomaly")
    severity: DisturbanceSeverity = Field(..., description="Assessed impact on operational safety")
    trigger_sensor: str = Field(..., description="Sensor origin: 'FT_WRIST', 'JOINT_EFFORT', 'CONTACT_MESH'")
    delta_magnitude: float = Field(..., description="Scalar metric of deviation (e.g., Delta Newtons or meters)")
    telemetry_window: List[TelemetrySnapshot] = Field(..., description="Pre-disturbance temporal buffer (last 200ms)")
```

---

## 4. Cognitive Inference Contracts (Nebius Token Factory & Nemotron)

### 4.1 Prompt Context Payload
```python
class CognitivePromptContext(BaseModel):
    mission_objective: str = Field(..., description="Active task (e.g., 'Transfer cylinder to pallet B')")
    disturbance_summary: DisturbanceEvent = Field(..., description="Detailed anomaly metadata")
    current_ee_pose: Pose6D = Field(..., description="Current end-effector location")
    target_ee_pose: Pose6D = Field(..., description="Nominal target destination")
    workspace_bounds: dict = Field(..., description="Safe bounding limits for validation awareness")
    available_primitives: List[str] = Field(
        default=["REGRASP_ADAPTIVE", "RETRACT_ALONG_NORMAL", "REPLAN_TRAJECTORY", "SAFE_CONTROLLED_DESCENT"]
    )
```

### 4.2 NVIDIA Nemotron Response Schema (Structured Output)
```python
class RecoveryTactic(str, Enum):
    REGRASP_ADAPTIVE = "REGRASP_ADAPTIVE"
    RETRACT_ALONG_NORMAL = "RETRACT_ALONG_NORMAL"
    REPLAN_TRAJECTORY = "REPLAN_TRAJECTORY"
    COMPLIANCE_ADJUST = "COMPLIANCE_ADJUST"
    SAFE_CONTROLLED_DESCENT = "SAFE_CONTROLLED_DESCENT"

class NemotronRecoveryStrategy(BaseModel):
    reasoning_trace: str = Field(..., description="Step-by-step physical diagnosis and cognitive rationale")
    root_cause: str = Field(..., description="Identified physical cause of the disturbance")
    recovery_tactic: RecoveryTactic = Field(..., description="Selected high-level recovery primitive")
    grip_force_adjustment_newtons: float = Field(default=0.0, description="Differential clamping force adjustment (N)")
    speed_scale_factor: float = Field(default=1.0, ge=0.1, le=1.0, description="Velocity scaling for safe recovery")
    intermediate_waypoints: List[Pose6D] = Field(default=[], description="Proposed recovery trajectory waypoints")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Model self-evaluated confidence score")
```

---

## 5. Deterministic Safety & Actuation Schemas

```python
class SafetyValidationStatus(str, Enum):
    APPROVED = "APPROVED"
    CLAMPED = "CLAMPED"
    REJECTED_JOINT_LIMIT = "REJECTED_JOINT_LIMIT"
    REJECTED_VELOCITY_LIMIT = "REJECTED_VELOCITY_LIMIT"
    REJECTED_WORKSPACE_BREACH = "REJECTED_WORKSPACE_BREACH"
    REJECTED_TIMEOUT = "REJECTED_TIMEOUT"

class SafetyValidationResult(BaseModel):
    status: SafetyValidationStatus = Field(..., description="Verification verdict")
    is_safe_to_execute: bool = Field(..., description="Binary execution gate")
    violation_reasons: List[str] = Field(default=[], description="Diagnostic list of safety violations if any")
    execution_timeout_ms: int = Field(default=3000, description="Maximum execution duration allowed for recovery")

class MotionCommand(BaseModel):
    command_id: str = Field(..., description="Unique command execution ID")
    tactic: RecoveryTactic = Field(..., description="Tactic being executed")
    target_joint_positions: Optional[List[List[float]]] = Field(None, description="Interpolated trajectory setpoints")
    gripper_target_effort: float = Field(..., description="Commanded gripper force/effort")
    timestamp_ns: int = Field(..., description="Dispatch timestamp")
```

---

## 6. System State Enum

```python
class SystemOperationalState(str, Enum):
    INITIALIZING = "INITIALIZING"
    NOMINAL = "NOMINAL"
    DISTURBANCE_DETECTED = "DISTURBANCE_DETECTED"
    COGNITIVE_DELIBERATING = "COGNITIVE_DELIBERATING"
    RECOVERING = "RECOVERING"
    RECOVERED_NOMINAL = "RECOVERED_NOMINAL"
    SAFE_CONTROLLED_STOP = "SAFE_CONTROLLED_STOP"
    EMERGENCY_STOP = "EMERGENCY_STOP"
```
