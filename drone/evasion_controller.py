"""
Evasion Controller for Event-Based Drone Collision Avoidance

Translates collision threat assessments into concrete drone maneuver
commands. Implements graded response levels from subtle course corrections
to emergency hard-evasion, with smooth transitions and hysteresis to
prevent oscillatory behavior.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .collision_predictor import CollisionAssessment, CollisionThreat


class EvasionLevel(Enum):
    """Severity level of the evasion response."""

    NONE = 0  # No threat, normal operation
    CAUTION = 1  # Low-level threat, gentle course correction
    WARNING = 2  # Significant threat, active avoidance
    CRITICAL = 3  # Imminent collision, aggressive evasion
    EMERGENCY = 4  # Collision unavoidable by normal means, brake-and-dodge


@dataclass
class DroneCommand:
    """
    Output command for drone flight controller.

    All values are in the drone's body-fixed frame:
      - x: forward (positive = toward camera optical axis)
      - y: left/right (positive = right)
      - z: up/down (positive = upward)
    Velocities in m/s, yaw_rate in rad/s.
    """

    level: EvasionLevel
    velocity_x: float  # forward velocity (m/s)
    velocity_y: float  # lateral velocity (m/s)
    velocity_z: float  # vertical velocity (m/s)
    yaw_rate: float  # rotation rate (rad/s)
    hover: bool  # if True, attempt to stop and hover
    description: str  # human-readable description


class EvasionController:
    """
    Translates collision threat assessments into drone flight commands.

    Implements a potential-field-inspired evasion policy with graded
    response levels and hysteresis to avoid oscillatory behavior.
    """

    def __init__(
        self,
        cruise_speed: float = 2.0,  # nominal forward speed (m/s)
        max_lateral_speed: float = 3.0,  # max sideways speed (m/s)
        max_vertical_speed: float = 2.0,  # max climb/descend speed (m/s)
        max_yaw_rate: float = 1.5,  # max rotation rate (rad/s)
        danger_threshold_caution: float = 0.15,
        danger_threshold_warning: float = 0.35,
        danger_threshold_critical: float = 0.60,
        danger_threshold_emergency: float = 0.85,
        hysteresis: float = 0.08,  # hysteresis to prevent oscillation between levels
        min_safe_distance: float = 0.5,  # m - minimum distance to maintain
        braking_deceleration: float = 3.0,  # m/s² - emergency braking
    ):
        """
        Args:
            cruise_speed: Nominal forward flight speed in m/s
            max_lateral_speed: Maximum sideways evasion speed
            max_vertical_speed: Maximum climb/descend speed
            max_yaw_rate: Maximum yaw rotation rate
            danger_threshold_*: Danger level thresholds (0-1) for each evasion level
            hysteresis: Prevents rapid toggling between levels
            min_safe_distance: Target minimum distance from obstacles
            braking_deceleration: Max deceleration for emergency stops
        """
        self.cruise_speed = cruise_speed
        self.max_lateral_speed = max_lateral_speed
        self.max_vertical_speed = max_vertical_speed
        self.max_yaw_rate = max_yaw_rate

        self.danger_thresholds = {
            EvasionLevel.CAUTION: danger_threshold_caution,
            EvasionLevel.WARNING: danger_threshold_warning,
            EvasionLevel.CRITICAL: danger_threshold_critical,
            EvasionLevel.EMERGENCY: danger_threshold_emergency,
        }
        self.hysteresis = hysteresis
        self.min_safe_distance = min_safe_distance
        self.braking_deceleration = braking_deceleration

        # State for hysteresis
        self._current_level: EvasionLevel = EvasionLevel.NONE
        self._last_safe_emission_time: float = 0.0

    def compute_command(
        self,
        assessment: CollisionAssessment,
        dt: float = 0.05,
    ) -> DroneCommand:
        """
        Compute drone flight command from collision assessment.

        Args:
            assessment: CollisionAssessment from CollisionPredictor
            dt: Time step in seconds (for dynamics computation)

        Returns:
            DroneCommand with velocity and yaw rate setpoints
        """
        # Determine evasion level with hysteresis
        target_level = self._determine_level(assessment.overall_danger_level)
        actual_level = self._apply_hysteresis(target_level)
        self._current_level = actual_level

        if actual_level == EvasionLevel.NONE:
            return self._command_none(assessment)
        elif actual_level == EvasionLevel.CAUTION:
            return self._command_caution(assessment)
        elif actual_level == EvasionLevel.WARNING:
            return self._command_warning(assessment)
        elif actual_level == EvasionLevel.CRITICAL:
            return self._command_critical(assessment)
        else:  # EMERGENCY
            return self._command_emergency(assessment)

    def _determine_level(self, danger: float) -> EvasionLevel:
        """Map overall danger value to evasion level (highest match wins)."""
        for level in [EvasionLevel.EMERGENCY, EvasionLevel.CRITICAL, EvasionLevel.WARNING, EvasionLevel.CAUTION]:
            if danger >= self.danger_thresholds[level]:
                return level
        return EvasionLevel.NONE

    def _apply_hysteresis(self, target_level: EvasionLevel) -> EvasionLevel:
        """
        Prevent rapid toggling between levels.
        Only downgrade if danger has meaningfully decreased.
        """
        if target_level.value < self._current_level.value:
            # Downgrading: require significant decrease
            # Only downgrade one level at a time
            return EvasionLevel(max(target_level.value, self._current_level.value - 1))
        return target_level

    def _command_none(self, assessment: CollisionAssessment) -> DroneCommand:
        """Normal cruising command - proceed forward at cruise speed."""
        return DroneCommand(
            level=EvasionLevel.NONE,
            velocity_x=self.cruise_speed,
            velocity_y=0.0,
            velocity_z=0.0,
            yaw_rate=0.0,
            hover=False,
            description="Normal cruise",
        )

    def _command_caution(self, assessment: CollisionAssessment) -> DroneCommand:
        """
        Gentle course correction.
        Reduce forward speed slightly and nudge away from threats.
        """
        evasion_vec = assessment.evasion_priority_vector

        # 20% speed reduction
        vx = self.cruise_speed * 0.8

        # Small lateral correction
        vy = evasion_vec[0] * self.max_lateral_speed * 0.2
        vz = -evasion_vec[1] * self.max_vertical_speed * 0.2

        # Small yaw toward evasion direction
        yaw_target = np.arctan2(evasion_vec[0], 1.0)
        yaw_rate = np.clip(yaw_target * 0.3, -self.max_yaw_rate * 0.3,
                           self.max_yaw_rate * 0.3)

        return DroneCommand(
            level=EvasionLevel.CAUTION,
            velocity_x=vx,
            velocity_y=vy,
            velocity_z=vz,
            yaw_rate=yaw_rate,
            hover=False,
            description=f"Caution: slowing, veering "
                         f"({vy:+.1f} m/s lateral)",
        )

    def _command_warning(self, assessment: CollisionAssessment) -> DroneCommand:
        """
        Active avoidance.
        Significant speed reduction + strong lateral component.
        """
        evasion_vec = assessment.evasion_priority_vector

        # 50% speed reduction
        vx = self.cruise_speed * 0.5

        # Strong lateral evasion
        vy = evasion_vec[0] * self.max_lateral_speed * 0.6
        vz = -evasion_vec[1] * self.max_vertical_speed * 0.4

        # Yaw toward clear path
        if assessment.safe_zones:
            best_bearing, _ = assessment.safe_zones[0]
            current_yaw_error = best_bearing
            yaw_rate = np.clip(
                current_yaw_error * 1.0, -self.max_yaw_rate * 0.6,
                self.max_yaw_rate * 0.6
            )
        else:
            yaw_rate = 0.0

        # If the threat is very close, add vertical evasion
        if assessment.most_dangerous and assessment.most_dangerous.time_to_collision < 1.0:
            # Dodge upward (objects usually below drone)
            vz = max(vz, self.max_vertical_speed * 0.5)

        return DroneCommand(
            level=EvasionLevel.WARNING,
            velocity_x=vx,
            velocity_y=vy,
            velocity_z=vz,
            yaw_rate=yaw_rate,
            hover=False,
            description=f"Warning: active avoidance "
                         f"({vy:+.1f} lateral, {vz:+.1f} vertical)",
        )

    def _command_critical(self, assessment: CollisionAssessment) -> DroneCommand:
        """
        Aggressive evasion maneuver.
        Stop forward motion, dodge hard.
        """
        evasion_vec = assessment.evasion_priority_vector

        # No forward motion - all energy to evasion
        vx = 0.0

        # Full lateral evasion
        vy = evasion_vec[0] * self.max_lateral_speed * 0.9
        vz = -evasion_vec[1] * self.max_vertical_speed * 0.8

        # Prefer climbing if threat is ahead
        if assessment.most_dangerous and assessment.most_dangerous.time_to_collision < 0.5:
            vz = max(vz, self.max_vertical_speed * 0.7)

        # Rotate toward safest bearing
        if assessment.safe_zones:
            best_bearing, _ = assessment.safe_zones[0]
            yaw_rate = np.clip(best_bearing * 1.5, -self.max_yaw_rate, self.max_yaw_rate)
        else:
            yaw_rate = 0.0

        return DroneCommand(
            level=EvasionLevel.CRITICAL,
            velocity_x=vx,
            velocity_y=vy,
            velocity_z=vz,
            yaw_rate=yaw_rate,
            hover=False,
            description=f"CRITICAL: hard evasion! "
                         f"({vy:+.1f} lateral, {vz:+.1f} vertical, "
                         f"yaw {np.degrees(yaw_rate):+.0f}°/s)",
        )

    def _command_emergency(self, assessment: CollisionAssessment) -> DroneCommand:
        """
        Emergency: collision appears unavoidable by evasion alone.
        Execute emergency stop + maximum-available dodge.
        """
        evasion_vec = assessment.evasion_priority_vector

        # Emergency braking (reverse thrust)
        vx = -self.cruise_speed * 0.5  # Reverse to bleed speed

        # Maximum lateral/vertical dodge
        vy = evasion_vec[0] * self.max_lateral_speed
        vz = max(-evasion_vec[1] * self.max_vertical_speed,
                 self.max_vertical_speed * 0.5)  # Always climb at least somewhat

        # Hard yaw to present minimal profile
        if assessment.safe_zones:
            best_bearing, _ = assessment.safe_zones[0]
            yaw_rate = np.clip(best_bearing * 2.0, -self.max_yaw_rate, self.max_yaw_rate)
        else:
            # No safe zone: yaw 90 degrees to present narrowest cross-section
            yaw_rate = self.max_yaw_rate

        most_dangerous_desc = ""
        if assessment.most_dangerous:
            most_dangerous_desc = (
                f" TTC={assessment.most_dangerous.time_to_collision:.2f}s"
            )

        return DroneCommand(
            level=EvasionLevel.EMERGENCY,
            velocity_x=vx,
            velocity_y=vy,
            velocity_z=vz,
            yaw_rate=yaw_rate,
            hover=False,
            description=f"EMERGENCY: brake + dodge!{most_dangerous_desc}",
        )

    def reset(self):
        """Reset controller state when re-arming."""
        self._current_level = EvasionLevel.NONE
        self._last_safe_emission_time = 0.0

    @property
    def current_level(self) -> EvasionLevel:
        """Current evasion level (with hysteresis applied)."""
        return self._current_level