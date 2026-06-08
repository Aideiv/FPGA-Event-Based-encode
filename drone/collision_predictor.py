"""
Collision Predictor for Event-Based Drone Avoidance

Predicts time-to-collision (TTC) and collision probability for
detected objects using normal flow analysis and tracking history.
Outputs collision threat assessments that drive evasion decisions.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from collections import deque

from .object_detector import DetectedObject


@dataclass
class CollisionThreat:
    """Assesses the collision risk of a single detected object."""

    object: DetectedObject
    time_to_collision: float  # estimated seconds until impact
    bearing: float  # angle in radians (0 = straight ahead, + = right)
    relative_speed: float  # speed relative to the drone (normalized units/s)
    expansion_rate: float  # rate of angular size increase (1/s)
    is_imminent: bool  # True if collision expected within safety window
    recommended_evasion_angle: float  # radians, direction to evade
    confidence: float  # 0.0 to 1.0 - confidence in this threat assessment


@dataclass
class CollisionAssessment:
    """Overall collision situation assessment across all tracked objects."""

    threats: List[CollisionThreat]
    most_dangerous: Optional[CollisionThreat]
    overall_danger_level: float  # 0.0 (safe) to 1.0 (must evade now)
    evasion_priority_vector: np.ndarray  # (2,) recommended evasion direction
    safe_zones: List[Tuple[float, float]]  # List of (bearing, width) of safe passages
    timestamp: float


class CollisionPredictor:
    """
    Predicts time-to-collision for detected objects and computes
    optimal evasion vectors.

    Uses looming-based TTC estimation from the event-camera optical flow:
    TTC ≈ θ / θ_dot where θ is angular size and θ_dot is expansion rate.
    This is a well-established principle from biological vision (locusts,
    pigeons, etc.) and robotics.

    Tracks object states over time with an extended Kalman filter approach
    to produce stable TTC estimates despite noisy event data.
    """

    def __init__(
        self,
        safety_time_threshold: float = 1.5,  # seconds
        danger_distance_threshold: float = 0.2,  # normalized focal plane dist for "close"
        expansion_smoothing_alpha: float = 0.3,  # EMA smoothing for expansion rate
        min_ttc: float = 0.1,  # minimum plausible TTC in seconds
        max_ttc: float = 30.0,  # maximum tracked TTC
        history_size: int = 15,
    ):
        """
        Args:
            safety_time_threshold: TTC below which evasion is required
            danger_distance_threshold: Angular distance to optical axis considered "central"
            expansion_smoothing_alpha: Smoothing factor for expansion rate EMA
            min_ttc: Minimum plausible TTC (clamp lower bound)
            max_ttc: Maximum TTC to track
            history_size: Number of past measurements to retain per object
        """
        self.safety_time_threshold = safety_time_threshold
        self.danger_distance_threshold = danger_distance_threshold
        self.expansion_smoothing_alpha = expansion_smoothing_alpha
        self.min_ttc = min_ttc
        self.max_ttc = max_ttc
        self.history_size = history_size

        # Per-object tracking state
        self.object_states: Dict[int, dict] = {}

    def assess(
        self,
        objects: List[DetectedObject],
        timestamp: float = 0.0,
        drone_velocity: Optional[np.ndarray] = None,
    ) -> CollisionAssessment:
        """
        Assess collision risk for all detected objects.

        Args:
            objects: List of detected objects from ObjectDetector
            timestamp: Current time in seconds
            drone_velocity: Optional (2,) drone velocity in normalized coords/s
                           (if available from IMU/odometry)

        Returns:
            CollisionAssessment with full threat analysis
        """
        if drone_velocity is None:
            drone_velocity = np.zeros(2)

        threats = []
        for obj in objects:
            threat = self._assess_object(obj, timestamp, drone_velocity)
            threats.append(threat)

        # Sort by danger (most dangerous first)
        threats.sort(key=lambda t: t.time_to_collision)

        # Compute overall danger level
        overall_danger = self._compute_overall_danger(threats)

        # Find most dangerous threat
        most_dangerous = threats[0] if threats else None

        # Compute optimal evasion vector
        evasion_vector = self._compute_evasion_vector(threats)

        # Find safe zones (bearing ranges free of threats)
        safe_zones = self._compute_safe_zones(objects)

        return CollisionAssessment(
            threats=threats,
            most_dangerous=most_dangerous,
            overall_danger_level=overall_danger,
            evasion_priority_vector=evasion_vector,
            safe_zones=safe_zones,
            timestamp=timestamp,
        )

    def _assess_object(
        self,
        obj: DetectedObject,
        timestamp: float,
        drone_velocity: np.ndarray,
    ) -> CollisionThreat:
        """Compute collision threat metrics for a single detected object."""

        # Initialize or update object state history
        if obj.id not in self.object_states:
            self.object_states[obj.id] = {
                "size_history": deque(maxlen=self.history_size),
                "time_history": deque(maxlen=self.history_size),
                "center_history": deque(maxlen=self.history_size),
                "smoothed_expansion": None,
            }

        state = self.object_states[obj.id]
        state["size_history"].append(obj.size)
        state["time_history"].append(timestamp)
        state["center_history"].append(obj.center_xy.copy())

        # Estimate expansion rate using size history
        expansion_rate = self._estimate_expansion_rate(state)
        if state["smoothed_expansion"] is None:
            state["smoothed_expansion"] = expansion_rate
        else:
            state["smoothed_expansion"] = (
                self.expansion_smoothing_alpha * expansion_rate
                + (1 - self.expansion_smoothing_alpha) * state["smoothed_expansion"]
            )
        smoothed_exp = state["smoothed_expansion"]

        # Time-to-collision from looming: TTC = size / expansion_rate
        # (Lee's τ hypothesis: tau = θ / θ_dot)
        if smoothed_exp > 1e-4:
            # Object is expanding (approaching)
            ttc = obj.size / smoothed_exp
            ttc = np.clip(ttc, self.min_ttc, self.max_ttc)
        elif smoothed_exp < -1e-4:
            # Object is shrinking (moving away) - set large TTC
            ttc = self.max_ttc
        else:
            # No significant expansion - use flow magnitude as proxy
            mean_speed = np.linalg.norm(obj.mean_flow)
            if mean_speed > 1e-4:
                ttc = obj.size / mean_speed
                ttc = np.clip(ttc, self.min_ttc, self.max_ttc)
            else:
                ttc = self.max_ttc

        # Bearing: angle of the object center relative to optical axis
        bearing = np.arctan2(obj.center_xy[0], -obj.center_xy[1])

        # Relative speed (accounting for drone's own motion)
        relative_flow = obj.mean_flow - drone_velocity
        relative_speed = np.linalg.norm(relative_flow)

        # Is a collision imminent?
        is_imminent = ttc < self.safety_time_threshold and obj.collision_urgency > 0.3

        # Recommended evasion direction: away from the threat
        # If the object is on the right, evade left (and vice versa)
        # Also factor in the object's motion direction
        ev_bearing = np.arctan2(-obj.center_xy[0], -obj.center_xy[1])

        # Tilt evasion toward clear side based on object motion
        flow_angle = np.arctan2(obj.mean_flow[1], obj.mean_flow[0])
        # If object is moving right, dodge left more aggressively
        cross_product = obj.center_xy[0] * obj.mean_flow[1] - obj.center_xy[1] * obj.mean_flow[0]
        if cross_product > 0:
            # Object crossing from right to left, dodge right
            ev_bearing += 0.3
        else:
            ev_bearing -= 0.3

        # Confidence: combination of track stability and flow certainty
        n_measurements = len(state["size_history"])
        track_confidence = min(1.0, n_measurements / 8.0)

        # Flow dispersion reduces confidence
        flow_confidence = np.exp(-obj.flow_dispersion / 2.0)

        confidence = 0.5 * track_confidence + 0.5 * flow_confidence

        return CollisionThreat(
            object=obj,
            time_to_collision=ttc,
            bearing=bearing,
            relative_speed=relative_speed,
            expansion_rate=smoothed_exp,
            is_imminent=is_imminent,
            recommended_evasion_angle=ev_bearing,
            confidence=confidence,
        )

    def _estimate_expansion_rate(self, state: dict) -> float:
        """Estimate rate of angular expansion from size history using linear regression."""
        sizes = list(state["size_history"])
        times = list(state["time_history"])

        if len(sizes) < 2:
            return 0.0

        # Linear regression of size vs time
        times_arr = np.array(times)
        sizes_arr = np.array(sizes)

        # Normalize time to avoid numerical issues
        t_mean = times_arr.mean()
        t_centered = times_arr - t_mean

        denominator = np.sum(t_centered**2)
        if denominator < 1e-10:
            return 0.0

        slope = np.sum(t_centered * (sizes_arr - sizes_arr.mean())) / denominator
        return float(slope)

    def _compute_overall_danger(self, threats: List[CollisionThreat]) -> float:
        """Compute overall danger level from all threats (0 = safe, 1 = must evade)."""
        if not threats:
            return 0.0

        # Weighted combination of TTC for all threats
        # Shorter TTC threats contribute more to danger
        danger_contributions = []
        for threat in threats:
            if threat.time_to_collision >= self.max_ttc:
                contrib = 0.0
            else:
                # Danger = 1 / (1 + TTC/safety_threshold)
                # At TTC = safety_threshold: danger = 0.5
                # At TTC = 0: danger = 1.0
                contrib = 1.0 / (1.0 + threat.time_to_collision / self.safety_time_threshold)
                contrib *= threat.confidence
            danger_contributions.append(contrib)

        # Softmax-like accumulation: overall danger = max + small bonus for multiple threats
        max_danger = max(danger_contributions)
        bonus = sum(
            d / (1 + i) for i, d in enumerate(sorted(danger_contributions, reverse=True)[1:], 1)
        )
        overall = min(max_danger + 0.2 * bonus, 1.0)
        return overall

    def _compute_evasion_vector(self, threats: List[CollisionThreat]) -> np.ndarray:
        """
        Compute optimal evasion direction as a 2D vector in the drone's
        body-fixed frame. Uses potential-field approach.

        Returns (2,) vector: direction to move; magnitude = urgency
        """
        if not threats:
            return np.zeros(2)

        evasion_vector = np.zeros(2)

        for threat in threats:
            if threat.time_to_collision > self.safety_time_threshold * 2:
                continue  # Skip non-threatening objects

            # Weight: higher for closer/urgent threats
            weight = 1.0 / (threat.time_to_collision + 0.1)
            weight *= threat.confidence

            # Vector pointing from object center to camera center
            obj_center = threat.object.center_xy
            center_dist = np.linalg.norm(obj_center)

            if center_dist > 1e-4:
                # Repulsive vector away from the object
                repulsive_dir = -obj_center / center_dist
            else:
                # Object dead center - use flow direction to break symmetry
                repulsive_dir = -threat.object.mean_flow
                repulsive_norm = np.linalg.norm(repulsive_dir)
                if repulsive_norm > 1e-4:
                    repulsive_dir /= repulsive_norm
                else:
                    repulsive_dir = np.array([0.0, -1.0])  # Default: go down

            # Add lateral bias: if multiple threats, push perpendicular to
            # the average threat direction to find clear passage
            perp = np.array([-obj_center[1], obj_center[0]])
            perp_norm = np.linalg.norm(perp)
            if perp_norm > 1e-6:
                perp /= perp_norm

            # Blend repulsive with perpendicular to "slide around" obstacles
            evasion_component = 0.7 * repulsive_dir + 0.3 * perp
            evasion_component /= np.linalg.norm(evasion_component) + 1e-6

            evasion_vector += weight * evasion_component

        # Normalize to get direction
        evasion_norm = np.linalg.norm(evasion_vector)
        if evasion_norm > 1e-4:
            evasion_vector /= evasion_norm

        return evasion_vector

    def _compute_safe_zones(
        self, objects: List[DetectedObject]
    ) -> List[Tuple[float, float]]:
        """
        Identify bearing ranges that are free of objects.
        Returns list of (bearing_center, angular_width) tuples.
        """
        if not objects:
            return [(-np.pi, 2 * np.pi)]  # Full circle is safe

        # Collect occupied bearing ranges
        occupied = []
        for obj in objects:
            bearing = np.arctan2(obj.center_xy[0], -obj.center_xy[1])
            angular_size = np.arctan(obj.size / max(np.linalg.norm(obj.center_xy), 1e-4))
            # Add padding for safety buffer
            angular_size = max(angular_size, 0.15)  # minimum 0.15 rad (~8.5°) buffer
            occupied.append((bearing - angular_size, bearing + angular_size))

        # Sort by start angle
        occupied.sort(key=lambda x: x[0])

        # Find gaps between occupied ranges
        safe_zones = []
        prev_end = -np.pi

        for start, end in occupied:
            if start > prev_end:
                gap_width = start - prev_end
                if gap_width > 0.1:  # Minimum 0.1 rad gap to count
                    gap_center = prev_end + gap_width / 2
                    safe_zones.append((gap_center, gap_width))
            prev_end = max(prev_end, end)

        # Check wrap-around gap
        final_gap = np.pi - prev_end
        if final_gap > 0.1:
            gap_center = prev_end + final_gap / 2
            safe_zones.append((gap_center, final_gap))

        # If nothing safe, return a fallback: go backward (the direction
        # likely to be clearest since objects are ahead)
        if not safe_zones:
            safe_zones.append((np.pi, 0.5))

        safe_zones.sort(key=lambda x: x[1], reverse=True)
        return safe_zones

    def reset(self):
        """Clear all tracked object states."""
        self.object_states.clear()