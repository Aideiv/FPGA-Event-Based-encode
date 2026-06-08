"""
Drone Collision Avoidance Module for Event-Based Cameras

Two complementary pipelines for collision avoidance:

Pipeline 1 — Flow-Based (VecKM encoder):
    Events → ObjectDetector → CollisionPredictor → EvasionController
    Uses normal flow for TTC estimation and graded evasion responses.
    SoTA references: EVDodgeNet, EVReflex, EV-TTC, Falanga et al.

Pipeline 2 — Direct Action (CNN, Bonazzi et al. 2025):
    Events → EventFrameAggregator → DirectActionPredictor
    Accumulates events into 80×80 frames, passes through DPU-optimized
    CNN for direct 5-class evasion action. Achieves ~2.14ms latency.

The DroneController can use one or both pipelines.
"""

from .drone_controller import DroneController
from .object_detector import ObjectDetector
from .collision_predictor import CollisionPredictor
from .evasion_controller import EvasionController
from .ttc_dense import DenseTTCEstimator, DenseTTCMap
from .event_frame_aggregator import EventFrameAggregator, EventFrame, EventFrameTemporalStack
from .direct_action_predictor import (
    DirectActionPredictor,
    DirectActionPrediction,
    DPULightNet,
    EvasionAction,
    ACTION_VELOCITY,
)
from .contrast_maximizer import ContrastMaximizer, CMaxResult
from .depth_estimator import DepthEstimator, DepthMap, EventDepthNet

__all__ = [
    "DroneController",
    "ObjectDetector",
    "CollisionPredictor",
    "EvasionController",
    "DenseTTCEstimator",
    "DenseTTCMap",
    "EventFrameAggregator",
    "EventFrame",
    "EventFrameTemporalStack",
    "DirectActionPredictor",
    "DirectActionPrediction",
    "DPULightNet",
    "EvasionAction",
    "ACTION_VELOCITY",
    "ContrastMaximizer",
    "CMaxResult",
    "DepthEstimator",
    "DepthMap",
    "EventDepthNet",
]
