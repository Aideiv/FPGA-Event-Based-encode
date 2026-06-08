"""
Drone Controller - Main Integration Layer

Integrates the event-camera normal flow estimator, object detector,
collision predictor, and evasion controller into a unified real-time
drone collision avoidance system.

Architecture:
  Event Stream → ObjectDetector → CollisionPredictor → EvasionController → DroneCommand
                    (NormalFlow)        (TTC + threats)       (velocity setpoints)
"""

import time
import numpy as np
import torch
from typing import Optional, Callable, List, Tuple, Dict
from dataclasses import dataclass, field
from collections import deque

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.inference import NormalFlowEstimator

from .object_detector import ObjectDetector, DetectedObject
from .collision_predictor import CollisionPredictor, CollisionAssessment, CollisionThreat
from .evasion_controller import (
    EvasionController,
    EvasionLevel,
    DroneCommand,
)


@dataclass
class DroneState:
    """Current drone state estimate."""

    timestamp: float = 0.0
    position_xyz: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # world frame (m)
    velocity_xyz: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # world frame (m/s)
    attitude_rpy: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # roll, pitch, yaw (rad)
    angular_velocity_rpy: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # rad/s
    camera_velocity_xy: np.ndarray = field(
        default_factory=lambda: np.zeros(2)
    )  # normalized coords/s in image plane


@dataclass
class AvoidanceSystemStatus:
    """Diagnostic and status information for the avoidance system."""

    level: EvasionLevel
    num_objects_detected: int
    num_threats: int
    overall_danger: float
    min_ttc: float
    processing_time_ms: float
    event_rate_hz: float
    current_command: Optional[DroneCommand]


class DroneController:
    """
    Main drone controller integrating event-based collision avoidance.

    Usage:
        controller = DroneController()
        controller.arm()

        while flying:
            events_t, events_xy = get_events_from_sensor()
            command = controller.step(events_t, events_xy)
            drone_flight_controller.send(command)
    """

    def __init__(
        self,
        training_set: str = "UNION",
        # Object detector settings
        min_events_per_object: int = 100,
        spatial_cluster_radius: float = 0.15,
        flow_similarity_threshold: float = 0.6,
        # Collision predictor settings
        safety_time_threshold: float = 1.5,
        expansion_smoothing_alpha: float = 0.3,
        # Evasion controller settings
        cruise_speed: float = 2.0,
        max_lateral_speed: float = 3.0,
        max_vertical_speed: float = 2.0,
        # Event buffer settings
        event_buffer_size: int = 100000,
        min_events_per_step: int = 5000,
        # IMU callback
        imu_callback: Optional[Callable[[], Tuple[np.ndarray, np.ndarray]]] = None,
        # Flight controller callback
        send_command_callback: Optional[Callable[[DroneCommand], None]] = None,
    ):
        """
        Args:
            training_set: Which pretrained model to use
            min_events_per_object: Minimum events needed to declare an object
            spatial_cluster_radius: Radius for spatial clustering (normalized coords)
            flow_similarity_threshold: Cosine similarity threshold for flow coherence
            safety_time_threshold: TTC below which evasion is triggered
            expansion_smoothing_alpha: EMA α for expansion rate smoothing
            cruise_speed: Nominal forward flight speed (m/s)
            max_lateral_speed: Max lateral evasion speed (m/s)
            max_vertical_speed: Max climb/descend speed (m/s)
            event_buffer_size: Maximum number of events to buffer
            min_events_per_step: Minimum events before processing a batch
            imu_callback: Optional function returning (accel_xyz, gyro_rpy) in SI units
            send_command_callback: Optional function to send DroneCommand to flight controller
        """
        # Core components
        self.estimator = NormalFlowEstimator(training_set=training_set)
        self.object_detector = ObjectDetector(
            estimator=self.estimator,
            training_set=training_set,
            min_events_per_object=min_events_per_object,
            spatial_cluster_radius=spatial_cluster_radius,
            flow_similarity_threshold=flow_similarity_threshold,
        )
        self.collision_predictor = CollisionPredictor(
            safety_time_threshold=safety_time_threshold,
            expansion_smoothing_alpha=expansion_smoothing_alpha,
        )
        self.evasion_controller = EvasionController(
            cruise_speed=cruise_speed,
            max_lateral_speed=max_lateral_speed,
            max_vertical_speed=max_vertical_speed,
        )

        # Event buffering
        self.event_buffer_size = event_buffer_size
        self.min_events_per_step = min_events_per_step
        self.event_t_buffer: deque = deque(maxlen=event_buffer_size)
        self.event_xy_buffer: deque = deque(maxlen=event_buffer_size)

        # Callbacks
        self.imu_callback = imu_callback
        self.send_command_callback = send_command_callback

        # State
        self.drone_state = DroneState()
        self.last_command: Optional[DroneCommand] = None
        self.is_armed: bool = False
        self.total_events_processed: int = 0
        self.last_step_time: float = 0.0
        self.processing_times: deque = deque(maxlen=100)
        self.event_rate_history: deque = deque(maxlen=100)

    def arm(self):
        """Arm the avoidance system. Must be called before step()."""
        self.is_armed = True
        self._reset_buffers()
        self.collision_predictor.reset()
        self.evasion_controller.reset()
        self.last_step_time = time.time()

    def disarm(self):
        """Disarm the avoidance system."""
        self.is_armed = False
        # Send hover command
        hover_cmd = DroneCommand(
            level=EvasionLevel.NONE,
            velocity_x=0.0,
            velocity_y=0.0,
            velocity_z=0.0,
            yaw_rate=0.0,
            hover=True,
            description="Disarmed - hover",
        )
        if self.send_command_callback:
            self.send_command_callback(hover_cmd)
        self.last_command = hover_cmd

    def step(
        self,
        events_t: torch.Tensor,
        events_xy: torch.Tensor,
        timestamp: Optional[float] = None,
    ) -> DroneCommand:
        """
        Process one batch of events and produce a flight command.

        Args:
            events_t: (n,) sorted event timestamps (seconds)
            events_xy: (n, 2) undistorted normalized event coordinates
            timestamp: Current time; uses clock if None

        Returns:
            DroneCommand to send to the flight controller
        """
        if not self.is_armed:
            raise RuntimeError("DroneController is not armed. Call arm() first.")

        if timestamp is None:
            timestamp = time.time()

        t_start = time.perf_counter()

        # Step 1: Update drone state from IMU if available
        self._update_drone_state(timestamp)

        # Step 2: Buffer incoming events
        self._buffer_events(events_t, events_xy)
        self.total_events_processed += len(events_t)

        # Step 3: Get batched events for processing
        batch_t, batch_xy = self._get_event_batch()

        if batch_t is None or len(batch_t) < self.min_events_per_step:
            # Not enough events - continue with last command or cruise
            command = self._get_safe_default_command()
            self.last_command = command
            return command

        batch_t_tensor = torch.tensor(batch_t, dtype=torch.float64)
        batch_xy_tensor = torch.tensor(batch_xy, dtype=torch.float32)

        # Step 4: Detect objects
        objects = self.object_detector.detect(
            batch_t_tensor, batch_xy_tensor, timestamp
        )

        # Step 5: Assess collision risk
        assessment = self.collision_predictor.assess(
            objects,
            timestamp=timestamp,
            drone_velocity=self.drone_state.camera_velocity_xy,
        )

        # Step 6: Compute evasion command
        dt = timestamp - self.last_step_time if self.last_step_time > 0 else 0.05
        command = self.evasion_controller.compute_command(assessment, dt=max(dt, 0.01))

        self.last_command = command
        self.last_step_time = timestamp

        # Step 7: Send command to flight controller
        if self.send_command_callback:
            self.send_command_callback(command)

        # Step 8: Log performance
        processing_time = (time.perf_counter() - t_start) * 1000  # ms
        self.processing_times.append(processing_time)

        return command

    def process_event_packet(
        self,
        events_t: np.ndarray,
        events_xy: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> Tuple[DroneCommand, AvoidanceSystemStatus]:
        """
        High-level interface: process event packet and return command + status.

        Args:
            events_t: (n,) event timestamps
            events_xy: (n, 2) event coordinates
            timestamp: Optional timestamp

        Returns:
            (DroneCommand, AvoidanceSystemStatus) tuple
        """
        t_tensor = torch.from_numpy(events_t).double()
        xy_tensor = torch.from_numpy(events_xy).float()

        command = self.step(t_tensor, xy_tensor, timestamp)

        # Build status
        assessment = self.collision_predictor.assess(
            self.object_detector.last_objects,
            timestamp=timestamp or time.time(),
        )

        avg_proc_time = (
            np.mean(list(self.processing_times)) if self.processing_times else 0.0
        )

        status = AvoidanceSystemStatus(
            level=command.level,
            num_objects_detected=len(self.object_detector.last_objects),
            num_threats=len(assessment.threats),
            overall_danger=assessment.overall_danger_level,
            min_ttc=(
                assessment.most_dangerous.time_to_collision
                if assessment.most_dangerous
                else float("inf")
            ),
            processing_time_ms=avg_proc_time,
            event_rate_hz=self._estimate_event_rate(),
            current_command=command,
        )

        return command, status

    def get_status(self) -> AvoidanceSystemStatus:
        """Get current system status without processing new events."""
        assessment = self.collision_predictor.assess(
            self.object_detector.last_objects
        )
        avg_proc_time = (
            np.mean(list(self.processing_times)) if self.processing_times else 0.0
        )

        return AvoidanceSystemStatus(
            level=self.evasion_controller.current_level,
            num_objects_detected=len(self.object_detector.last_objects),
            num_threats=len(assessment.threats),
            overall_danger=assessment.overall_danger_level,
            min_ttc=(
                assessment.most_dangerous.time_to_collision
                if assessment.most_dangerous
                else float("inf")
            ),
            processing_time_ms=avg_proc_time,
            event_rate_hz=self._estimate_event_rate(),
            current_command=self.last_command,
        )

    def _buffer_events(self, events_t: torch.Tensor, events_xy: torch.Tensor):
        """Add events to the processing buffer."""
        t_np = events_t.numpy() if isinstance(events_t, torch.Tensor) else events_t
        xy_np = events_xy.numpy() if isinstance(events_xy, torch.Tensor) else events_xy

        for i in range(len(t_np)):
            self.event_t_buffer.append(t_np[i])
            self.event_xy_buffer.append(xy_np[i])

    def _get_event_batch(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get accumulated events as a batch.
        Drains the buffer and returns sorted arrays.
        """
        if len(self.event_t_buffer) == 0:
            return None, None

        t_arr = np.array(list(self.event_t_buffer))
        xy_arr = np.array(list(self.event_xy_buffer))

        self.event_t_buffer.clear()
        self.event_xy_buffer.clear()

        # Ensure sorted by time
        sort_idx = np.argsort(t_arr)
        return t_arr[sort_idx], xy_arr[sort_idx]

    def _update_drone_state(self, timestamp: float):
        """Update drone state from IMU callback if available."""
        self.drone_state.timestamp = timestamp

        if self.imu_callback:
            try:
                accel, gyro = self.imu_callback()
                self.drone_state.angular_velocity_rpy = gyro

                # Simple integration for attitude (for demo purposes)
                # In production, use a proper IMU filter (Mahony/Madgwick/EKF)
                dt = timestamp - self.drone_state.timestamp
                if dt > 0 and dt < 1.0:
                    self.drone_state.attitude_rpy += gyro * dt

                # Approximate camera-plane velocity from rotation
                yaw_rate = gyro[2]  # Assuming gyro in rpy order
                # Rough mapping: yaw rate to image plane motion
                self.drone_state.camera_velocity_xy = np.array(
                    [yaw_rate * 0.5, 0.0]
                )  # simplified
            except Exception:
                pass  # IMU unavailable, continue with last state

    def _get_safe_default_command(self) -> DroneCommand:
        """Return a safe default command (cruise or last command)."""
        if self.last_command:
            return self.last_command

        return DroneCommand(
            level=EvasionLevel.NONE,
            velocity_x=self.evasion_controller.cruise_speed,
            velocity_y=0.0,
            velocity_z=0.0,
            yaw_rate=0.0,
            hover=False,
            description="Default cruise (no events)",
        )

    def _estimate_event_rate(self) -> float:
        """Estimate incoming event rate in Hz."""
        now = time.time()
        self.event_rate_history.append((now, self.total_events_processed))

        if len(self.event_rate_history) < 2:
            return 0.0

        t0, e0 = self.event_rate_history[0]
        t1, e1 = self.event_rate_history[-1]
        dt = t1 - t0
        if dt < 1e-6:
            return 0.0
        return (e1 - e0) / dt

    def _reset_buffers(self):
        """Clear all event buffers and trackers."""
        self.event_t_buffer.clear()
        self.event_xy_buffer.clear()
        self.total_events_processed = 0
        self.processing_times.clear()
        self.event_rate_history.clear()