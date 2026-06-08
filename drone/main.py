#!/usr/bin/env python3
"""
Drone Collision Avoidance System - Main Entry Point

Event-based camera collision avoidance for drones using VecKM normal flow.
Connects to an event camera (or replays recorded data), processes events
in real-time, and outputs flight controller commands.

Usage:
    # Live operation with an event camera
    python -m drone.main --live

    # Replay from recorded .npy event data
    python -m drone.main --replay /path/to/events_t.npy /path/to/events_xy.npy

    # Simulation with synthetic event stream
    python -m drone.main --demo

Configuration is handled via drone/config.yaml or command-line arguments.
"""

import argparse
import sys
import os
import time
import json
import signal
from typing import Optional, Dict, Any
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone.drone_controller import (
    DroneController,
    DroneState,
    AvoidanceSystemStatus,
)
from drone.evasion_controller import EvasionLevel, DroneCommand


# ─── Default Configuration ──────────────────────────────────────────────

DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "training_set": "UNION",
    },
    "detector": {
        "min_events_per_object": 100,
        "spatial_cluster_radius": 0.15,
        "flow_similarity_threshold": 0.6,
    },
    "predictor": {
        "safety_time_threshold": 1.5,  # seconds
        "expansion_smoothing_alpha": 0.3,
    },
    "controller": {
        "cruise_speed": 2.0,  # m/s
        "max_lateral_speed": 3.0,  # m/s
        "max_vertical_speed": 2.0,  # m/s
    },
    "buffer": {
        "event_buffer_size": 100000,
        "min_events_per_step": 5000,
    },
    "logging": {
        "print_interval_seconds": 1.0,
        "log_to_file": None,  # set to a path to enable file logging
    },
}


# ─── Utility Functions ──────────────────────────────────────────────────

def format_command(cmd: DroneCommand) -> str:
    """Format a DroneCommand for console display."""
    level_colors = {
        EvasionLevel.NONE: "\033[32m",     # green
        EvasionLevel.CAUTION: "\033[33m",  # yellow
        EvasionLevel.WARNING: "\033[33m",  # yellow (bold)
        EvasionLevel.CRITICAL: "\033[31m", # red
        EvasionLevel.EMERGENCY: "\033[35m", # magenta
    }
    reset = "\033[0m"
    color = level_colors.get(cmd.level, "")

    arrows = {
        EvasionLevel.NONE: "→",
        EvasionLevel.CAUTION: "↗",
        EvasionLevel.WARNING: "⚠",
        EvasionLevel.CRITICAL: "⚡",
        EvasionLevel.EMERGENCY: "☠",
    }

    return (
        f"{color}[{cmd.level.name:>9}]{reset} {arrows[cmd.level]} "
        f"vx={cmd.velocity_x:+5.1f} vy={cmd.velocity_y:+5.1f} "
        f"vz={cmd.velocity_z:+5.1f} "
        f"yaw={np.degrees(cmd.yaw_rate):+6.1f}°/s "
        f"| {cmd.description}"
    )


def format_status(status: AvoidanceSystemStatus) -> str:
    """Format system status for console display."""
    return (
        f"Objects: {status.num_objects_detected:3d} | "
        f"Threats: {status.num_threats:3d} | "
        f"Danger: {status.overall_danger:.3f} | "
        f"TTC: {status.min_ttc:6.2f}s | "
        f"Proc: {status.processing_time_ms:5.1f}ms | "
        f"Events: {status.event_rate_hz/1000:6.1f}k/s"
    )


# ─── Simulated Event Camera ─────────────────────────────────────────────

class SimulatedEventCamera:
    """
    Generates synthetic event data for testing/demo purposes.
    Simulates approaching objects with varying collision trajectories.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.time_accumulator = 0.0
        self.object_params = self._init_objects()

    def _init_objects(self):
        """Initialize simulated objects with random trajectories."""
        num_objects = self.rng.randint(1, 4)
        objects = []
        for i in range(num_objects):
            objects.append({
                "center_x": self.rng.uniform(-0.8, 0.8),
                "center_y": self.rng.uniform(-0.6, 0.6),
                "size": self.rng.uniform(0.05, 0.2),
                "vx": self.rng.uniform(-0.3, 0.3),  # image plane velocity
                "vy": self.rng.uniform(-0.3, 0.3),
                "expansion": self.rng.uniform(0.01, 0.08),  # looming rate
                "event_rate": self.rng.uniform(20000, 80000),  # events/s for this object
            })
        return objects

    def generate_events(self, duration: float = 0.05) -> tuple:
        """
        Generate synthetic event data for a short time window.

        Returns:
            (events_t, events_xy) as numpy arrays
        """
        all_events_t = []
        all_events_xy = []

        for obj in self.object_params:
            obj["center_x"] += obj["vx"] * duration
            obj["center_y"] += obj["vy"] * duration
            obj["size"] += obj["expansion"] * duration

            # Clamp to valid image plane
            obj["center_x"] = np.clip(obj["center_x"], -1.0, 1.0)
            obj["center_y"] = np.clip(obj["center_y"], -1.0, 1.0)
            obj["size"] = np.clip(obj["size"], 0.01, 0.8)

            # Generate events within the object's spatial extent
            n_events = int(obj["event_rate"] * duration)
            if n_events < 10:
                continue

            # Object events: Gaussian distribution around center
            ex = self.rng.randn(n_events) * obj["size"] + obj["center_x"]
            ey = self.rng.randn(n_events) * obj["size"] + obj["center_y"]

            # Add flow: events move outward from center (looming)
            radial_x = ex - obj["center_x"]
            radial_y = ey - obj["center_y"]
            flow_mag = obj["expansion"] * 0.5 + np.abs(
                obj["vx"] * radial_x + obj["vy"] * radial_y
            ) * 0.1

            # Event timestamps within duration
            et = self.rng.uniform(
                self.time_accumulator,
                self.time_accumulator + duration,
                n_events,
            )
            et.sort()

            all_events_t.append(et)
            all_events_xy.append(np.stack([ex, ey], axis=1))

        # Add background noise events
        n_noise = int(5000 * duration)
        if n_noise > 0:
            noise_t = self.rng.uniform(
                self.time_accumulator,
                self.time_accumulator + duration,
                n_noise,
            )
            noise_x = self.rng.uniform(-1.0, 1.0, n_noise)
            noise_y = self.rng.uniform(-1.0, 1.0, n_noise)
            noise_t.sort()

            all_events_t.append(noise_t)
            all_events_xy.append(np.stack([noise_x, noise_y], axis=1))

        self.time_accumulator += duration

        # Re-initialize objects occasionally to keep things interesting
        if self.rng.rand() < 0.01:
            self.object_params = self._init_objects()
            self.time_accumulator = 0.0

        if not all_events_t:
            return (
                np.array([self.time_accumulator]),
                np.array([[0.0, 0.0]]),
            )

        events_t = np.concatenate(all_events_t)
        events_xy = np.concatenate(all_events_xy)

        # Sort by time
        sort_idx = np.argsort(events_t)
        return events_t[sort_idx], events_xy[sort_idx]


# ─── Main Execution ─────────────────────────────────────────────────────

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file, merging with defaults."""
    config = DEFAULT_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            user_config = json.load(f)
        _deep_update(config, user_config)
    return config


def _deep_update(base: dict, update: dict):
    """Recursively update a nested dictionary."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def run_live(config: Dict[str, Any]):
    """
    Run in live mode - connect to an actual event camera.
    This is a stub that demonstrates the interface.
    Users should replace this with their camera SDK integration.
    """
    print("=" * 70)
    print("  DRONE COLLISION AVOIDANCE SYSTEM - LIVE MODE")
    print("  Event Camera: Connect your camera SDK here")
    print("=" * 70)
    print()
    print("To use with a real event camera, integrate your camera SDK and")
    print("replace run_live() with code that streams events to the controller.")
    print()
    print("Example pseudocode:")
    print("  camera = YourEventCamera()")
    print("  camera.start()")
    print("  while flying:")
    print("      events = camera.get_events()")
    print("      cmd = controller.step(events.t, events.xy)")
    print("      flight_controller.send(cmd)")
    print()
    print("Running demo simulation instead...")
    print()
    run_demo(config)


def run_replay(config: Dict[str, Any], events_t_path: str, events_xy_path: str):
    """
    Replay recorded event data through the avoidance pipeline.

    Args:
        events_t_path: Path to .npy file with event timestamps
        events_xy_path: Path to .npy file with event coordinates
    """
    print("=" * 70)
    print("  DRONE COLLISION AVOIDANCE SYSTEM - REPLAY MODE")
    print(f"  Events T:  {events_t_path}")
    print(f"  Events XY: {events_xy_path}")
    print("=" * 70)

    # Load event data
    events_t = np.load(events_t_path)
    events_xy = np.load(events_xy_path)

    print(f"\nLoaded {len(events_t):,} events")
    print(f"Time range: {events_t.min():.3f}s to {events_t.max():.3f}s")
    print(f"Duration: {events_t.max() - events_t.min():.3f}s")
    print()

    # Initialize controller
    controller = DroneController(
        training_set=config["model"]["training_set"],
        min_events_per_object=config["detector"]["min_events_per_object"],
        spatial_cluster_radius=config["detector"]["spatial_cluster_radius"],
        flow_similarity_threshold=config["detector"]["flow_similarity_threshold"],
        safety_time_threshold=config["predictor"]["safety_time_threshold"],
        expansion_smoothing_alpha=config["predictor"]["expansion_smoothing_alpha"],
        cruise_speed=config["controller"]["cruise_speed"],
        max_lateral_speed=config["controller"]["max_lateral_speed"],
        max_vertical_speed=config["controller"]["max_vertical_speed"],
        event_buffer_size=config["buffer"]["event_buffer_size"],
        min_events_per_step=config["buffer"]["min_events_per_step"],
    )
    controller.arm()

    # Process in time windows
    print_interval = config["logging"]["print_interval_seconds"]
    last_print = 0.0
    step_count = 0

    t_min, t_max = events_t.min(), events_t.max()
    window = 0.05  # 50ms windows
    t_current = t_min

    try:
        while t_current < t_max:
            # Get events in current window
            mask = (events_t >= t_current) & (events_t < t_current + window)
            batch_t = events_t[mask]
            batch_xy = events_xy[mask]

            if len(batch_t) > 0:
                cmd = controller.step(
                    torch.from_numpy(batch_t).double(),
                    torch.from_numpy(batch_xy).float(),
                    timestamp=t_current,
                )

                # Print status periodically
                if t_current - last_print >= print_interval:
                    status = controller.get_status()
                    print(f"\rt={t_current:7.3f}s | {format_status(status)} | "
                          f"{format_command(cmd)}", end="")
                    last_print = t_current

            t_current += window
            step_count += 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    finally:
        controller.disarm()
        print(f"\nProcessed {step_count} steps over "
              f"{len(events_t):,} events.")
        status = controller.get_status()
        print(f"Final status: {format_status(status)}")


def run_demo(config: Dict[str, Any]):
    """
    Run with a simulated event camera for demonstration.

    Generates synthetic events with approaching objects and runs
    the full avoidance pipeline, printing status and commands.
    """
    print("=" * 70)
    print("  DRONE COLLISION AVOIDANCE SYSTEM - DEMO MODE")
    print("  Synthetic event camera simulating approaching objects")
    print("=" * 70)
    print()

    # Initialize simulated camera
    camera = SimulatedEventCamera(seed=42)

    # Initialize controller
    print("Initializing normal flow estimator (this may take a moment)...")
    controller = DroneController(
        training_set=config["model"]["training_set"],
        min_events_per_object=config["detector"]["min_events_per_object"],
        spatial_cluster_radius=config["detector"]["spatial_cluster_radius"],
        flow_similarity_threshold=config["detector"]["flow_similarity_threshold"],
        safety_time_threshold=config["predictor"]["safety_time_threshold"],
        expansion_smoothing_alpha=config["predictor"]["expansion_smoothing_alpha"],
        cruise_speed=config["controller"]["cruise_speed"],
        max_lateral_speed=config["controller"]["max_lateral_speed"],
        max_vertical_speed=config["controller"]["max_vertical_speed"],
        event_buffer_size=config["buffer"]["event_buffer_size"],
        min_events_per_step=config["buffer"]["min_events_per_step"],
    )
    controller.arm()
    print("System armed. Starting avoidance loop...\n")

    print_interval = config["logging"]["print_interval_seconds"]
    last_print = 0.0
    simulation_time = 0.0
    step_duration = 0.05  # 50ms per step
    step_count = 0

    # Running state
    running = [True]

    def signal_handler(sig, frame):
        print("\n\nShutdown signal received...")
        running[0] = False

    signal.signal(signal.SIGINT, signal_handler)

    try:
        while running[0]:
            # Generate synthetic events
            events_t, events_xy = camera.generate_events(duration=step_duration)

            if len(events_t) == 0:
                simulation_time += step_duration
                continue

            # Process through controller
            cmd, status = controller.process_event_packet(
                events_t,
                events_xy,
                timestamp=simulation_time,
            )

            # Print status
            if simulation_time - last_print >= print_interval:
                print(f"t={simulation_time:7.1f}s | {format_status(status)}")
                print(f"  {format_command(cmd)}")
                print()
                last_print = simulation_time

            simulation_time += step_duration
            step_count += 1

            # Slow down simulation to make output readable
            time.sleep(0.02)

    except KeyboardInterrupt:
        pass

    finally:
        controller.disarm()
        print(f"\nSimulation complete. {step_count} steps over "
              f"{simulation_time:.1f}s simulated time.")
        status = controller.get_status()
        print(f"Final status: {format_status(status)}")


def main():
    parser = argparse.ArgumentParser(
        description="Event-Based Drone Collision Avoidance System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m drone.main --demo           # Run with synthetic events
  python -m drone.main --live           # Run with real event camera (stub)
  python -m drone.main --config custom.json --demo
  python -m drone.main --replay events_t.npy events_xy.npy
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true",
                      help="Run with simulated event camera")
    mode.add_argument("--live", action="store_true",
                      help="Run with real event camera (requires integration)")
    mode.add_argument("--replay", nargs=2, metavar=("EVENTS_T", "EVENTS_XY"),
                      help="Replay recorded event data from .npy files")

    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON configuration file")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress non-critical output")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    if args.quiet:
        config["logging"]["print_interval_seconds"] = 10.0

    # Route to appropriate mode
    if args.demo:
        run_demo(config)
    elif args.live:
        run_live(config)
    elif args.replay:
        run_replay(config, args.replay[0], args.replay[1])


if __name__ == "__main__":
    main()