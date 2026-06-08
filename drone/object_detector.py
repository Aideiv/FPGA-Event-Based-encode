"""
Event-Based Object Detector for Drone Collision Avoidance

Detects objects from event-camera streams by clustering events with
coherent normal flow patterns. Uses the VecKM normal flow estimator
to compute per-event motion vectors, then segments objects by motion
coherence and spatial proximity.
"""

import numpy as np
import torch
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from collections import deque

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.inference import NormalFlowEstimator


@dataclass
class DetectedObject:
    """Represents a detected object from event-camera data."""

    id: int
    center_xy: np.ndarray  # (2,) normalized focal plane coordinates
    size: float  # estimated angular size in normalized coordinates
    mean_flow: np.ndarray  # (2,) mean normal flow vector
    flow_dispersion: float  # standard deviation of flow directions
    event_count: int  # number of events associated with this object
    distance_estimate: float  # rough distance estimate (arbitrary units, smaller = closer)
    collision_urgency: float  # 0.0 (safe) to 1.0 (imminent collision)


class ObjectDetector:
    """
    Detects objects from event-camera streams using normal flow analysis.

    Leverages the VecKM normal flow estimator to compute per-event motion
    vectors, then clusters events by spatial and flow coherence to segment
    distinct objects. Maintains temporal consistency via Kalman-like
    tracking of object positions across frames.

    Attributes:
        estimator: The NormalFlowEstimator instance for computing flow
        min_events_per_object: Minimum events required to declare an object
        spatial_cluster_radius: Radius for spatial clustering (normalized coords)
        flow_similarity_threshold: Max cosine distance for flow coherence
        expansion_threshold: Flow divergence indicating expanding/approaching object
    """

    def __init__(
        self,
        estimator: Optional[NormalFlowEstimator] = None,
        training_set: str = "UNION",
        min_events_per_object: int = 100,
        spatial_cluster_radius: float = 0.15,
        flow_similarity_threshold: float = 0.6,
        expansion_threshold: float = 0.02,
        tracking_memory: int = 10,
    ):
        """
        Args:
            estimator: Pre-initialized NormalFlowEstimator, or None to create one
            training_set: Which pretrained model to use ('UNION', 'MVSEC', 'DSEC', 'EVIMO')
            min_events_per_object: Minimum events needed to declare an object cluster
            spatial_cluster_radius: Radius in normalized coordinates for spatial clustering
            flow_similarity_threshold: Cosine similarity threshold (0-1) for flow coherence
            expansion_threshold: Flow divergence rate indicating an approaching object
            tracking_memory: Number of past frames to maintain for object tracking
        """
        self.estimator = (
            estimator
            if estimator is not None
            else NormalFlowEstimator(training_set=training_set)
        )
        self.min_events_per_object = min_events_per_object
        self.spatial_cluster_radius = spatial_cluster_radius
        self.flow_similarity_threshold = flow_similarity_threshold
        self.expansion_threshold = expansion_threshold

        # Object tracking state
        self.tracking_memory = tracking_memory
        self.track_history: Dict[int, deque] = {}  # object_id -> deque of center_xy
        self.next_object_id: int = 0
        self.last_objects: List[DetectedObject] = []

    def detect(
        self,
        events_t: torch.Tensor,
        events_xy: torch.Tensor,
        timestamp: float = 0.0,
    ) -> List[DetectedObject]:
        """
        Process an event packet and return detected objects.

        Args:
            events_t: (n,) sorted event timestamps in seconds
            events_xy: (n, 2) undistorted normalized event coordinates
            timestamp: Current timestamp for temporal tracking

        Returns:
            List of DetectedObject instances, sorted by collision_urgency descending
        """
        if events_t.shape[0] < self.min_events_per_object:
            return []

        # Step 1: Compute normal flow for all events
        flow_predictions, flow_uncertainty = self.estimator.inference(
            events_t, events_xy
        )

        # Filter out high-uncertainty flow vectors
        uncertainty_mask = flow_uncertainty < 0.3
        valid_indices = torch.where(uncertainty_mask)[0]

        if len(valid_indices) < self.min_events_per_object:
            return []

        valid_events_xy = events_xy[valid_indices].numpy()
        valid_flow = flow_predictions[valid_indices].numpy()
        valid_flow_norm = np.linalg.norm(valid_flow, axis=-1, keepdims=True)
        valid_flow_norm = np.clip(valid_flow_norm, a_min=1e-6, a_max=None)
        valid_flow_dir = valid_flow / valid_flow_norm

        # Step 2: Cluster events by spatial proximity and flow coherence
        labels = self._cluster_events_spatio_flow(
            valid_events_xy, valid_flow_dir, valid_flow_norm[:, 0]
        )

        # Step 3: Extract object properties per cluster
        objects = []
        for label in np.unique(labels):
            if label < 0:
                continue  # Skip noise
            mask = labels == label
            if mask.sum() < self.min_events_per_object:
                continue

            cluster_xy = valid_events_xy[mask]
            cluster_flow = valid_flow[mask]
            cluster_flow_norms = np.linalg.norm(cluster_flow, axis=-1)

            # Object center: density-weighted mean position
            center_xy = np.median(cluster_xy, axis=0)

            # Object size: bounding radius
            distances_from_center = np.linalg.norm(
                cluster_xy - center_xy[None, :], axis=-1
            )
            size = np.percentile(distances_from_center, 80)

            # Mean flow (motion direction and speed)
            # Use circular mean for direction, weighted by flow magnitude
            angles = np.arctan2(cluster_flow[:, 1], cluster_flow[:, 0])
            weights = np.clip(cluster_flow_norms, 0, np.percentile(cluster_flow_norms, 95))
            sum_sin = np.sum(weights * np.sin(angles))
            sum_cos = np.sum(weights * np.cos(angles))
            mean_angle = np.arctan2(sum_sin, sum_cos)
            mean_magnitude = np.median(cluster_flow_norms)
            mean_flow = np.array(
                [mean_magnitude * np.cos(mean_angle), mean_magnitude * np.sin(mean_angle)]
            )

            # Flow dispersion: circular std of angles
            circular_variance = (
                np.sqrt(np.sum(np.cos(angles - mean_angle)) ** 2
                        + np.sum(np.sin(angles - mean_angle)) ** 2)
                / len(angles)
            )
            flow_dispersion = np.sqrt(-2 * np.log(max(circular_variance, 1e-6)))

            # Distance estimate: based on flow divergence (looming)
            flow_divergence = self._compute_flow_divergence(cluster_xy, cluster_flow)
            distance_estimate = 1.0 / (flow_divergence + 1e-6)

            # Collision urgency: combines flow magnitude, divergence, and direction
            center_norm = np.linalg.norm(center_xy)  # distance from image center
            # Objects closer to center and with higher divergence are more urgent
            collision_urgency = (
                0.4 * np.clip(flow_divergence / 0.1, 0, 1)  # divergent flow
                + 0.3 * np.clip(mean_magnitude / 5.0, 0, 1)  # high relative speed
                + 0.3 * np.exp(-center_norm / 0.3)  # central in field of view
            )

            # Step 4: Match to existing tracks or create new object
            matched_id = self._match_to_track(center_xy)
            if matched_id is None:
                matched_id = self.next_object_id
                self.next_object_id += 1

            # Update track history
            if matched_id not in self.track_history:
                self.track_history[matched_id] = deque(maxlen=self.tracking_memory)
            self.track_history[matched_id].append(center_xy.copy())

            obj = DetectedObject(
                id=matched_id,
                center_xy=center_xy,
                size=size,
                mean_flow=mean_flow,
                flow_dispersion=flow_dispersion,
                event_count=mask.sum(),
                distance_estimate=distance_estimate,
                collision_urgency=min(collision_urgency, 1.0),
            )
            objects.append(obj)

        # Prune stale tracks
        active_ids = {obj.id for obj in objects}
        stale = [oid for oid in self.track_history if oid not in active_ids]
        for oid in stale:
            del self.track_history[oid]

        # Sort by urgency
        objects.sort(key=lambda o: o.collision_urgency, reverse=True)
        self.last_objects = objects
        return objects

    def _cluster_events_spatio_flow(
        self,
        events_xy: np.ndarray,
        flow_dir: np.ndarray,
        flow_norm: np.ndarray,
    ) -> np.ndarray:
        """
        Cluster events by spatial proximity and flow direction coherence.

        Uses a DBSCAN-like approach: events are connected if they are
        spatially close AND have similar flow directions.
        """
        n = events_xy.shape[0]
        labels = -np.ones(n, dtype=int)
        visited = np.zeros(n, dtype=bool)
        current_label = 0

        # Grid-based acceleration: bin events spatially
        grid_size = self.spatial_cluster_radius
        grid_bins = {}

        for i in range(n):
            cell_x = int(events_xy[i, 0] / grid_size)
            cell_y = int(events_xy[i, 1] / grid_size)
            key = (cell_x, cell_y)
            if key not in grid_bins:
                grid_bins[key] = []
            grid_bins[key].append(i)

        # Process each event
        neighbor_offsets = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (-1, -1), (1, -1), (-1, 1),
        ]

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True

            # BFS/DFS to grow cluster
            cluster_indices = [i]
            labels[i] = current_label

            for seed_idx in cluster_indices:
                cell_x = int(events_xy[seed_idx, 0] / grid_size)
                cell_y = int(events_xy[seed_idx, 1] / grid_size)

                for dx, dy in neighbor_offsets:
                    neighbor_cell = (cell_x + dx, cell_y + dy)
                    if neighbor_cell not in grid_bins:
                        continue

                    for j in grid_bins[neighbor_cell]:
                        if visited[j]:
                            continue

                        spatial_dist = np.linalg.norm(
                            events_xy[seed_idx] - events_xy[j]
                        )
                        if spatial_dist > self.spatial_cluster_radius:
                            continue

                        # Flow direction similarity
                        flow_similarity = np.dot(flow_dir[seed_idx], flow_dir[j])
                        if flow_similarity < self.flow_similarity_threshold:
                            continue

                        # Magnitude similarity check
                        norm_ratio = min(flow_norm[seed_idx], flow_norm[j]) / max(
                            flow_norm[seed_idx], flow_norm[j], 1e-6
                        )
                        if norm_ratio < 0.3:
                            continue

                        visited[j] = True
                        labels[j] = current_label
                        cluster_indices.append(j)

            current_label += 1

        return labels

    def _compute_flow_divergence(
        self, cluster_xy: np.ndarray, cluster_flow: np.ndarray
    ) -> float:
        """
        Estimate the divergence of the flow field within an object cluster.
        Positive divergence suggests the object is approaching (looming effect).

        Uses a simple approach: compute the radial component of flow vectors
        relative to the cluster center, and average.
        """
        center = np.median(cluster_xy, axis=0)
        radial_vectors = cluster_xy - center[None, :]
        radial_norms = np.linalg.norm(radial_vectors, axis=-1, keepdims=True)
        radial_norms = np.clip(radial_norms, a_min=1e-6, a_max=None)
        radial_unit = radial_vectors / radial_norms

        # Project flow onto radial direction
        radial_flow = np.sum(cluster_flow * radial_unit, axis=-1)
        divergence = np.median(radial_flow)
        return float(divergence)

    def _match_to_track(self, center_xy: np.ndarray) -> Optional[int]:
        """
        Match a detected center to an existing object track.
        Uses nearest-neighbor with distance gating.
        """
        if not self.track_history:
            return None

        best_id = None
        best_dist = self.spatial_cluster_radius * 3

        for obj_id, history in self.track_history.items():
            last_center = history[-1]
            dist = np.linalg.norm(last_center - center_xy)
            if dist < best_dist:
                best_dist = dist
                best_id = obj_id

        return best_id

    def get_flow_field_for_visualization(
        self, events_t: torch.Tensor, events_xy: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convenience method: compute and return flow field for visualization
        or external processing.
        """
        flow, uncertainty = self.estimator.inference(events_t, events_xy)
        return events_xy.numpy(), flow.numpy(), uncertainty.numpy()