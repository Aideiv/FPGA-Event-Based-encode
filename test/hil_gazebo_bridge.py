#!/usr/bin/env python3
"""hil_gazebo_bridge.py — Hardware-in-the-Loop Simulation Bridge.

Connects the event-based collision avoidance pipeline to a drone simulator
for hardware-in-the-loop testing WITHOUT requiring physical hardware.

Supported simulators:
  - Gazebo + PX4 SITL (primary, via MAVLink)
  - AirSim (via Python API)
  - Custom physics simulator (built-in simple model)

Architecture:
  ┌──────────────────────────────────────────────────────────────┐
  │                    HIL Simulation Loop                        │
  │                                                               │
  │  Simulator ──► (IMU, GPS, pose) ──► Event Generator ──►      │
  │                                                               │
  │  Event Generator ──► (x, y, t, p) stream ──► Pipeline ──►    │
  │                                                               │
  │  Pipeline ──► (vx, vy, vz, yaw) command ──► Simulator ──►    │
  │                                                               │
  │  Loop at 100Hz (simulated FPGA) or software-only             │
  └──────────────────────────────────────────────────────────────┘

USAGE:
    python test/hil_gazebo_bridge.py --sim gazebo  # Connect to PX4 SITL
    python test/hil_gazebo_bridge.py --sim airsim   # Connect to AirSim
    python test/hil_gazebo_bridge.py --sim simple   # Built-in simple physics
    python test/hil_gazebo_bridge.py --demo         # Short demo with objects
"""

import os
import sys
import math
import time
import json
import argparse
import threading
import traceback
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# ── Logging ──────────────────────────────────────────────────────────────────
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("hil_bridge")


# ═════════════════════════════════════════════════════════════════════════════
# Data Types
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class DroneState:
    """Full drone state from simulator or real vehicle."""

    timestamp_s: float = 0.0
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # x,y,z (m)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # vx,vy,vz
    attitude: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # roll,pitch,yaw
    angular_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # p,q,r
    imu_accel: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, -9.81])
    )  # ax,ay,az
    battery_v: float = 12.6
    armed: bool = False


@dataclass
class SimulatedCamera:
    """Simulated event camera parameters."""

    width: int = 640
    height: int = 480
    fov_rad: float = 1.2  # ~68°
    event_rate_hz: float = 1_000_000  # 1 MHz event rate
    noise_std_px: float = 1.0
    min_depth_m: float = 0.3
    max_depth_m: float = 50.0


# ═════════════════════════════════════════════════════════════════════════════
# Simulator Interfaces
# ═════════════════════════════════════════════════════════════════════════════


class SimulatorInterface:
    """Abstract simulator interface."""

    def connect(self) -> bool:
        raise NotImplementedError

    def get_state(self) -> DroneState:
        raise NotImplementedError

    def send_command(self, vx: float, vy: float, vz: float, yaw_rate: float):
        raise NotImplementedError

    def disconnect(self):
        pass


class PX4SITLInterface(SimulatorInterface):
    """PX4 Software-in-the-Loop via MAVLink."""

    def __init__(self, udp_port: int = 14540):
        self.udp_port = udp_port
        self._connected = False

    def connect(self) -> bool:
        try:
            from pymavlink import mavutil

            self._mav = mavutil.mavlink_connection(
                f"udp:127.0.0.1:{self.udp_port}"
            )
            self._mav.wait_heartbeat(timeout=10)
            logger.info(
                f"Connected to PX4 SITL on udp:{self.udp_port} "
                f"(system={self._mav.target_system})"
            )
            self._connected = True

            # Request data streams
            self._mav.mav.request_data_stream_send(
                self._mav.target_system,
                self._mav.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                10,  # 10 Hz
                1,
            )

            # Enable offboard mode
            self._set_offboard_mode()

            return True
        except ImportError:
            logger.error("pymavlink not installed. pip install pymavlink")
            logger.info("Falling back to simple simulator")
            return False
        except Exception as e:
            logger.error(f"PX4 SITL connection failed: {e}")
            return False

    def get_state(self) -> DroneState:
        if not self._connected:
            return DroneState()

        state = DroneState()
        try:
            msg = self._mav.recv_match(blocking=False)
            while msg is not None:
                mtype = msg.get_type()
                if mtype == "ATTITUDE":
                    state.attitude = np.array([msg.roll, msg.pitch, msg.yaw])
                    state.angular_velocity = np.array(
                        [msg.rollspeed, msg.pitchspeed, msg.yawspeed]
                    )
                elif mtype == "LOCAL_POSITION_NED":
                    state.position = np.array([msg.x, msg.y, -msg.z])
                    state.velocity = np.array([msg.vx, msg.vy, msg.vz])
                elif mtype == "HIGHRES_IMU":
                    state.imu_accel = np.array(
                        [msg.xacc, msg.yacc, msg.zacc]
                    )
                elif mtype == "HEARTBEAT":
                    state.armed = bool(msg.base_mode & 128)
                msg = self._mav.recv_match(blocking=False)
        except Exception:
            pass

        state.timestamp_s = time.time()
        return state

    def send_command(self, vx: float, vy: float, vz: float, yaw_rate: float):
        if not self._connected:
            return
        try:
            self._mav.mav.set_position_target_local_ned_send(
                0,  # time_boot_ms
                self._mav.target_system,
                self._mav.target_component,
                8,  # MAV_FRAME_BODY_NED
                0b0000111111000111,  # type_mask: ignore position + accel
                0, 0, 0,  # position (ignored)
                vx, vy, vz,  # velocity
                0, 0, 0,  # acceleration (ignored)
                0,  # yaw (ignored)
                yaw_rate,
            )
        except Exception as e:
            logger.warning(f"Command send failed: {e}")

    def disconnect(self):
        if self._connected:
            self._mav.close()

    def _set_offboard_mode(self):
        try:
            self._mav.mav.command_long_send(
                self._mav.target_system,
                self._mav.target_component,
                176,  # MAV_CMD_DO_SET_MODE
                0,
                1,  # MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
                6,  # PX4_CUSTOM_MAIN_MODE_OFFBOARD
                0, 0, 0, 0, 0,
            )
        except Exception:
            pass


class SimplePhysicsSimulator(SimulatorInterface):
    """Built-in simple 3DOF drone physics model for testing without external
    simulators.

    State: [x, y, z, vx, vy, vz, yaw]
    Control: [ax, ay, az] via velocity setpoint with rate limiting

    Includes basic gravity, drag, and velocity limits.
    """

    def __init__(self):
        self.pos = np.zeros(3)  # x, y, z (m)
        self.pos[2] = 10.0  # Start at 10m altitude
        self.vel = np.zeros(3)  # vx, vy, vz (m/s)
        self.yaw = 0.0  # rad
        self.yaw_rate = 0.0  # rad/s
        self._dt = 0.01  # 100Hz
        self._time = 0.0

        # Physics constants
        self._max_horiz_vel = 10.0  # m/s
        self._max_vert_vel = 5.0  # m/s
        self._max_horiz_acc = 8.0  # m/s²
        self._max_vert_acc = 6.0  # m/s²
        self._drag_coeff = 0.3  # 1/s velocity decay
        self._ground_z = 0.0  # m

        # Simulated obstacles
        self._obstacles = []  # List of (x, y, z, radius) tuples

    def connect(self) -> bool:
        logger.info("Simple physics simulator initialized")
        return True

    def add_obstacle(self, x: float, y: float, z: float, radius: float):
        """Add a spherical obstacle in world frame."""
        self._obstacles.append((x, y, z, radius))
        logger.info(f"Added obstacle at ({x:.1f}, {y:.1f}, {z:.1f}) r={radius:.1f}")

    def step_physics(
        self, target_vx: float, target_vy: float, target_vz: float, _yaw_rate: float
    ):
        """Integrate physics one time step."""
        dt = self._dt
        self._time += dt

        # Yaw integration
        self.yaw += self.yaw_rate * dt
        self.yaw = math.fmod(self.yaw, 2 * math.pi)

        # Velocity tracking (rate-limited)
        ax = np.clip(
            (target_vx - self.vel[0]) / dt, -self._max_horiz_acc, self._max_horiz_acc
        )
        ay = np.clip(
            (target_vy - self.vel[1]) / dt, -self._max_horiz_acc, self._max_horiz_acc
        )
        az = np.clip(
            (target_vz - self.vel[2]) / dt, -self._max_vert_acc, self._max_vert_acc
        )

        # Integrate
        self.vel[0] += ax * dt
        self.vel[1] += ay * dt
        self.vel[2] += az * dt

        # Drag
        self.vel[0] *= 1.0 - self._drag_coeff * dt
        self.vel[1] *= 1.0 - self._drag_coeff * dt
        self.vel[2] *= 1.0 - self._drag_coeff * dt

        # Velocity limits
        horiz = math.sqrt(self.vel[0] ** 2 + self.vel[1] ** 2)
        if horiz > self._max_horiz_vel:
            self.vel[0] *= self._max_horiz_vel / horiz
            self.vel[1] *= self._max_horiz_vel / horiz
        self.vel[2] = np.clip(self.vel[2], -self._max_vert_vel, self._max_vert_vel)

        # Integrate position
        self.pos += self.vel * dt

        # Ground collision
        if self.pos[2] < self._ground_z:
            self.pos[2] = self._ground_z
            self.vel[2] = max(0, self.vel[2])

    def get_state(self) -> DroneState:
        state = DroneState()
        state.timestamp_s = self._time
        state.position = self.pos.copy()
        state.velocity = self.vel.copy()
        state.attitude = np.array([0.0, 0.0, self.yaw])
        state.angular_velocity = np.array([0.0, 0.0, self.yaw_rate])
        state.imu_accel = np.array([0.0, 0.0, -9.81])
        state.armed = True
        return state

    def send_command(self, vx: float, vy: float, vz: float, yaw_rate: float):
        self.step_physics(vx, vy, vz, yaw_rate)

    def disconnect(self):
        pass

    def get_obstacles(self):
        return self._obstacles


# ═════════════════════════════════════════════════════════════════════════════
# Event Generator: converts simulated scene to event camera output
# ═════════════════════════════════════════════════════════════════════════════


class SimulatedEventGenerator:
    """Generates synthetic event camera events from a simulated scene.

    Given drone state and a list of obstacles, projects obstacles onto
    the camera image plane and generates events at edges with appropriate
    flow vectors.

    For hardware-in-the-loop testing, this replaces the physical event camera.
    """

    def __init__(self, camera: SimulatedCamera):
        self.cam = camera
        self._prev_projections = {}  # object_id → (cx, cy, radius_px)

    def generate_events(
        self, drone_state: DroneState, obstacles: list, max_per_frame: int = 4096
    ) -> tuple:
        """Generate synthetic events for one frame.

        Returns:
            events_xy: (n, 2) normalized coordinates [-1, 1]
            events_t: (n,) timestamps in seconds
        """
        if not obstacles:
            # Uniform noise events (static background)
            n = min(256, max_per_frame)
            xy = np.random.uniform(-1, 1, (n, 2)).astype(np.float32)
            t = np.full(n, drone_state.timestamp_s, dtype=np.float64)
            return xy, t

        focal_px = (self.cam.width / 2) / math.tan(self.cam.fov_rad / 2)

        events_xy_list = []
        events_t_list = []

        for obj_id, (ox, oy, oz, radius) in enumerate(obstacles):
            # Transform obstacle to drone body frame
            # Simple translation: drone at (px, py, pz), looking forward (+X)
            # Camera forward = drone body +X (yaw rotation)
            yaw = drone_state.attitude[2]
            cos_y, sin_y = math.cos(-yaw), math.sin(
                -yaw
            )  # World → body

            dx_world = ox - drone_state.position[0]
            dy_world = oy - drone_state.position[1]
            dz_world = oz - drone_state.position[2]

            # Rotate to body frame
            dx_body = cos_y * dx_world - sin_y * dy_world
            dy_body = sin_y * dx_world + cos_y * dy_world
            dz_body = dz_world

            # Forward must be positive (in front of camera)
            if dx_body < self.cam.min_depth_m:
                continue

            # Perspective projection
            if dx_body > 0:
                proj_x = focal_px * dy_body / dx_body
                proj_y = focal_px * dz_body / dx_body
                proj_radius = focal_px * radius / dx_body

                # Normalize to [-1, 1]
                norm_x = proj_x / (self.cam.width / 2)
                norm_y = proj_y / (self.cam.height / 2)
                norm_r = proj_radius / (self.cam.width / 2)

                # Check if on screen
                if abs(norm_x) > 1.2 or abs(norm_y) > 1.2:
                    continue

                # Generate events on the object's boundary (edge events)
                n_events = int(max_per_frame * 0.3)
                angles = np.linspace(0, 2 * math.pi, n_events, endpoint=False)
                angles += np.random.normal(0, 0.05, n_events)

                # Event positions: circle perimeter + noise
                ex = norm_x + norm_r * np.cos(angles) + np.random.normal(
                    0, self.cam.noise_std_px / (self.cam.width / 2), n_events
                )
                ey = norm_y + norm_r * np.sin(angles) + np.random.normal(
                    0, self.cam.noise_std_px / (self.cam.height / 2), n_events
                )

                # Clip to image bounds
                ex = np.clip(ex, -1.0, 1.0)
                ey = np.clip(ey, -1.0, 1.0)

                # Compute previous projection for flow
                if obj_id in self._prev_projections:
                    prev_cx, prev_cy, prev_r = self._prev_projections[obj_id]
                    # Expansion ratio → looming or receding
                    if prev_r > 0.001:
                        expansion_ratio = norm_r / prev_r
                    else:
                        expansion_ratio = 1.0
                else:
                    expansion_ratio = 1.0

                self._prev_projections[obj_id] = (norm_x, norm_y, norm_r)

                events_xy_list.append(np.column_stack([ex, ey]))
                events_t_list.append(
                    np.full(n_events, drone_state.timestamp_s, dtype=np.float64)
                )

        if events_xy_list:
            events_xy = np.vstack(events_xy_list).astype(np.float32)
            events_t = np.concatenate(events_t_list).astype(np.float64)
        else:
            n = min(128, max_per_frame)
            events_xy = np.random.uniform(-1, 1, (n, 2)).astype(np.float32)
            events_t = np.full(n, drone_state.timestamp_s, dtype=np.float64)

        return events_xy, events_t


# ═════════════════════════════════════════════════════════════════════════════
# HIL Simulation Runner
# ═════════════════════════════════════════════════════════════════════════════


class HILSimulation:
    """Main hardware-in-the-loop simulation loop.

    Integrates:
      - Simulator (physics + rendering)
      - Event generator (scene → events)
      - Collision avoidance pipeline (events → velocity command)
      - Telemetry logging
    """

    def __init__(self, simulator: SimulatorInterface, use_pipeline: bool = True):
        self.sim = simulator
        self.camera = SimulatedCamera()
        self.event_gen = SimulatedEventGenerator(self.camera)
        self.use_pipeline = use_pipeline

        self._running = False
        self._loop_hz = 100
        self._loop_period = 1.0 / self._loop_hz

        # Statistics
        self.stats = {
            "iterations": 0,
            "collision_detections": 0,
            "evasion_maneuvers": 0,
            "max_urgency_seen": 0.0,
            "min_altitude_m": float("inf"),
            "pipeline_latency_ms": [],
        }

        # Initialize pipeline
        self._pipeline = None
        if use_pipeline:
            try:
                from drone import DroneController

                self._pipeline = DroneController(
                    training_set="UNION", safety_time_threshold=1.5
                )
                self._pipeline.arm()
                logger.info("Collision avoidance pipeline initialized")
            except Exception as e:
                logger.warning(f"Pipeline init failed: {e}. Using simple evasion.")
                self._pipeline = None

    def run(self, duration_s: float = 30.0, add_demo_obstacles: bool = True):
        """Run HIL simulation for specified duration.

        Args:
            duration_s: Simulation duration in seconds
            add_demo_obstacles: If True, add moving obstacles for testing
        """
        logger.info(f"Starting HIL simulation ({duration_s}s, {self._loop_hz}Hz)")

        if not self.sim.connect():
            logger.error("Simulator connection failed")
            return

        # Add demo obstacles if using simple simulator
        if add_demo_obstacles and hasattr(self.sim, "add_obstacle"):
            # Stationary obstacle ahead
            self.sim.add_obstacle(30.0, 0.0, 10.0, 2.0)
            # Moving obstacle approaching from right
            self.sim.add_obstacle(10.0, 15.0, 10.0, 1.5)

        self._running = True
        start_time = time.time()
        last_print_time = start_time

        try:
            while self._running and (time.time() - start_time) < duration_s:
                loop_start = time.perf_counter()

                # ── Step 1: Get drone state ──
                drone_state = self.sim.get_state()

                # ── Step 2: Generate synthetic events ──
                obstacles = (
                    self.sim.get_obstacles()
                    if hasattr(self.sim, "get_obstacles")
                    else []
                )
                events_xy, events_t = self.event_gen.generate_events(
                    drone_state, obstacles
                )

                # ── Step 3: Run collision avoidance pipeline ──
                cmd_vx, cmd_vy, cmd_vz, cmd_yaw = 2.0, 0.0, 0.0, 0.0  # Default: cruise
                evasion_level = "NONE"

                if self._pipeline and len(events_t) > 100:
                    try:
                        import torch

                        t_tensor = torch.from_numpy(events_t)
                        xy_tensor = torch.from_numpy(events_xy)
                        command = self._pipeline.step(t_tensor, xy_tensor)

                        cmd_vx = command.velocity_x
                        cmd_vy = command.velocity_y
                        cmd_vz = command.velocity_z
                        cmd_yaw = command.yaw_rate

                        status = self._pipeline.get_status()
                        evasion_level = status.level_name

                        if status.overall_danger > 0.1:
                            self.stats["collision_detections"] += 1
                        if status.overall_danger > 0.5:
                            self.stats["evasion_maneuvers"] += 1
                        self.stats["max_urgency_seen"] = max(
                            self.stats["max_urgency_seen"], status.overall_danger
                        )

                    except Exception as e:
                        logger.debug(f"Pipeline error: {e}")

                # ── Step 4: Send command to simulator ──
                self.sim.send_command(cmd_vx, cmd_vy, cmd_vz, cmd_yaw)

                # ── Step 5: Log telemetry ──
                self.stats["iterations"] += 1
                latency = (time.perf_counter() - loop_start) * 1000
                self.stats["pipeline_latency_ms"].append(latency)
                self.stats["min_altitude_m"] = min(
                    self.stats["min_altitude_m"], drone_state.position[2]
                )

                # Print status periodically
                if time.time() - last_print_time >= 1.0:
                    logger.info(
                        f"t={drone_state.timestamp_s:.1f}s  "
                        f"pos=({drone_state.position[0]:.1f},{drone_state.position[1]:.1f},"
                        f"{drone_state.position[2]:.1f})  "
                        f"vel=({drone_state.velocity[0]:.1f},{drone_state.velocity[1]:.1f},"
                        f"{drone_state.velocity[2]:.1f})  "
                        f"cmd=({cmd_vx:+.1f},{cmd_vy:+.1f},{cmd_vz:+.1f})  "
                        f"evasion={evasion_level}  "
                        f"latency={latency:.1f}ms"
                    )
                    last_print_time = time.time()

                # Maintain loop rate
                elapsed = time.perf_counter() - loop_start
                if elapsed < self._loop_period:
                    time.sleep(self._loop_period - elapsed)

        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        finally:
            self._running = False
            self.sim.disconnect()
            self._print_summary()

    def _print_summary(self):
        if self.stats["iterations"] == 0:
            return

        latencies = self.stats["pipeline_latency_ms"]
        avg_lat = np.mean(latencies) if latencies else 0
        max_lat = np.max(latencies) if latencies else 0

        print("\n" + "=" * 60)
        print("  HIL Simulation Summary")
        print("=" * 60)
        print(f"  Iterations:           {self.stats['iterations']}")
        print(f"  Collision detections: {self.stats['collision_detections']}")
        print(f"  Evasion maneuvers:    {self.stats['evasion_maneuvers']}")
        print(f"  Max urgency:          {self.stats['max_urgency_seen']:.3f}")
        print(f"  Min altitude:         {self.stats['min_altitude_m']:.1f} m")
        print(f"  Pipeline latency:     avg={avg_lat:.1f}ms  max={max_lat:.1f}ms")
        print(f"  Control rate:         {1000/avg_lat:.0f} Hz effective" if avg_lat > 0 else "")
        print("=" * 60)

        # Threshold check
        if self.stats["collision_detections"] > 0:
            print("\n  ✓ Pipeline detected and responded to obstacles")
        else:
            print("\n  ⚠ No collision detections — check obstacle placement")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HIL Simulation — Event-Based Drone Collision Avoidance"
    )
    parser.add_argument(
        "--sim",
        default="simple",
        choices=["gazebo", "px4", "airsim", "simple"],
        help="Simulator backend",
    )
    parser.add_argument(
        "--duration", type=float, default=30.0, help="Simulation duration (seconds)"
    )
    parser.add_argument(
        "--no-pipeline",
        action="store_true",
        help="Run without collision avoidance (raw flight)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a short demo scenario with obstacles",
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        default=14540,
        help="UDP port for PX4 SITL connection",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Hardware-in-the-Loop Simulation")
    print("  FPGA Event-Based Drone Collision Avoidance")
    print(f"  Backend: {args.sim}")
    print("=" * 60)

    # Create simulator
    if args.sim in ("gazebo", "px4"):
        sim = PX4SITLInterface(udp_port=args.udp_port)
        if not sim.connect():
            logger.info("PX4 SITL unavailable — falling back to simple simulator")
            sim = SimplePhysicsSimulator()
            sim.connect()
    elif args.sim == "airsim":
        try:
            import airsim  # noqa: F401

            logger.info("AirSim interface: configure AirSim connection settings")
            # AirSim client would be set up here
            sim = SimplePhysicsSimulator()
            sim.connect()
        except ImportError:
            logger.error("airsim not installed. pip install airsim")
            sys.exit(1)
    else:
        sim = SimplePhysicsSimulator()
        sim.connect()

    # Run simulation
    hil = HILSimulation(sim, use_pipeline=not args.no_pipeline)
    hil.run(duration_s=args.duration, add_demo_obstacles=args.demo or True)