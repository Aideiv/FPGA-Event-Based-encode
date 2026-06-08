# Drone Collision Avoidance System

Event-based camera collision avoidance for drones using the VecKM normal flow estimator. Detects objects approaching the drone via looming-based time-to-collision estimation and generates graded evasion maneuvers.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        DRONE MAIN LOOP                            │
│                                                                   │
│  Event Camera ──► DroneController.step() ──► Flight Controller    │
│                      │                          ▲                 │
│                      ▼                          │                 │
│              ┌───────────────┐        ┌─────────────────┐        │
│              │ ObjectDetector │──────►│ CollisionPredictor│       │
│              │ (VecKM flow +  │       │ (TTC + threats)   │       │
│              │  clustering)   │       └────────┬─────────┘        │
│              └───────────────┘                │                   │
│                                               ▼                   │
│                                    ┌──────────────────┐          │
│                                    │ EvasionController │          │
│                                    │ (velocity setpts) │          │
│                                    └────────┬─────────┘          │
│                                             │                     │
│                                    DroneCommand                   │
└──────────────────────────────────────────────────────────────────┘
```

## Modules

| Module | Purpose |
|--------|---------|
| `object_detector.py` | Detects objects from event-camera streams using normal flow clustering |
| `collision_predictor.py` | Computes TTC, threat levels, and optimal evasion vectors |
| `evasion_controller.py` | Translates threat assessments into graded drone flight commands |
| `drone_controller.py` | Main integration layer combining all modules |
| `main.py` | Entry point with demo simulation, replay, and live modes |

## Evasion Levels

| Level | Danger | Behavior |
|-------|--------|----------|
| **NONE** | < 0.15 | Normal cruise at set speed |
| **CAUTION** | ≥ 0.15 | 20% speed reduction, gentle lateral nudge |
| **WARNING** | ≥ 0.35 | 50% speed reduction, strong lateral + vertical |
| **CRITICAL** | ≥ 0.60 | Full stop forward, aggressive dodge |
| **EMERGENCY** | ≥ 0.85 | Reverse thrust + maximum dodge + hard yaw |

## Quick Start

```bash
# Install the base package first
pip install -e .

# Run the demo simulation with synthetic events
python -m drone.main --demo

# Replay recorded event data
python -m drone.main --replay demo/demo_data/dataset_events_t.npy demo/demo_data/undistorted_events_xy.npy
```

## API Usage

```python
from drone import DroneController
import torch

# Initialize
controller = DroneController(
    training_set="UNION",
    cruise_speed=2.0,           # m/s forward
    safety_time_threshold=1.5,   # TTC threshold for evasion
)
controller.arm()

# Main loop
while flying:
    # Get events from your event camera
    events_t = torch.tensor([...])   # (n,) timestamps in seconds
    events_xy = torch.tensor([...])  # (n, 2) normalized coordinates

    # Process and get command
    command = controller.step(events_t, events_xy)
    
    # Send to your flight controller
    drone.set_velocity(command.velocity_x, command.velocity_y, command.velocity_z)
    drone.set_yaw_rate(command.yaw_rate)

    # Check status
    status = controller.get_status()
    print(f"Danger: {status.overall_danger:.2f}, TTC: {status.min_ttc:.1f}s")

controller.disarm()
```

## Integration with Real Hardware

### Event Camera
The system expects `events_xy` in undistorted normalized coordinates. Convert raw sensor coordinates using your camera's calibration:

```python
import cv2

def undistort_events(raw_xy, K, D):
    """raw_xy: (n, 2) raw pixel coordinates
       K: (3, 3) camera intrinsic matrix
       D: (k,) distortion coefficients"""
    undistorted = cv2.undistortPoints(raw_xy.reshape(-1, 1, 2).astype(np.float32), K, D)
    return undistorted.reshape(-1, 2)
```

### Flight Controller
Connect the `DroneCommand` output to your flight stack:

```python
def send_to_px4(command):
    """Example: PX4 offboard control"""
    setpoint = PositionSetpoint()
    setpoint.velocity[0] = command.velocity_x   # forward
    setpoint.velocity[1] = command.velocity_y   # lateral
    setpoint.velocity[2] = command.velocity_z   # vertical
    setpoint.yawspeed = command.yaw_rate
    offboard.setpoint_send(setpoint)

controller = DroneController(
    send_command_callback=send_to_px4,
)

# Optional: integrate IMU for ego-motion compensation
def get_imu():
    return (accel_xyz, gyro_rpy)

controller = DroneController(imu_callback=get_imu)
```

## Configuration

Create a `config.json` to tune parameters:

```json
{
    "model": {"training_set": "UNION"},
    "detector": {
        "min_events_per_object": 100,
        "spatial_cluster_radius": 0.15,
        "flow_similarity_threshold": 0.6
    },
    "predictor": {
        "safety_time_threshold": 1.5,
        "expansion_smoothing_alpha": 0.3
    },
    "controller": {
        "cruise_speed": 2.0,
        "max_lateral_speed": 3.0,
        "max_vertical_speed": 2.0
    }
}
```

Run with: `python -m drone.main --config config.json --demo`

## Key Design Decisions

1. **Looming-based TTC**: Uses Lee's τ hypothesis (τ = θ/θ̇) for time-to-collision estimation from angular size expansion rate, inspired by biological vision systems.

2. **Graded evasion**: Five discrete response levels with hysteresis prevent oscillatory behavior while maintaining responsiveness.

3. **Spatio-flow clustering**: Objects are segmented by both spatial proximity AND flow coherence, distinguishing moving objects from static background even with ego-motion.

4. **VecKM encoder**: Leverages the pretrained VecKM normal flow estimator for per-event motion vectors without requiring frame-based processing.

## Related Work

This system builds on the VecKM normal flow estimator from:
> Yuan et al., "Learning Normal Flow Directly From Event Neighborhoods", ICCV 2025

See the project root README for details on the underlying model architecture and training.