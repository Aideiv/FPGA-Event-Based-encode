"""
Drone Collision Avoidance Module for Event-Based Cameras

Uses event-camera normal flow estimation to detect objects,
predict collisions, and generate evasion maneuvers in real time.

SoTA-Referenced Modules:
  - ttc_dense:   EVReflex-style dense per-event TTC (Walters & Hadfield 2021)
                 + EV-TTC low-light handling (Bisulco et al. 2025)
  - object_detector:   EVDodgeNet-style spatio-flow clustering (Sanket et al. 2020)
  - collision_predictor: Looming-based TTC + Kalman tracking
  - evasion_controller: Graded response with hysteresis
"""

from .drone_controller import DroneController
from .object_detector import ObjectDetector
from .collision_predictor import CollisionPredictor
from .evasion_controller import EvasionController
from .ttc_dense import DenseTTCEstimator, DenseTTCMap

__all__ = [
    "DroneController",
    "ObjectDetector",
    "CollisionPredictor",
    "EvasionController",
    "DenseTTCEstimator",
    "DenseTTCMap",
]
