"""
Dense Time-to-Collision Map (EVReflex-style)

Computes per-event time-to-collision (TTC) directly from the normal flow
field, rather than relying on object clustering. Produces a dense TTC map
spanning the entire field of view, enabling detection of threats that don't
yet form coherent clusters (e.g., thin wires, distant obstacles, birds).

References:
    - Walters & Hadfield, "EVReflex: Dense Time-to-Impact Prediction for
      Event-based Obstacle Avoidance", IROS 2021.
    - Clady et al., "Asynchronous visual event-based time-to-contact",
      Front. Neurosci. 2014.
    - Bisulco et al., "EV-TTC: Event-Based Time to Collision under
      Low Light Conditions", RA-L 2025.
"""

import numpy as np
import torch
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class DenseTTCMap:
    """Per-event time-to-collision map spanning the full field of view."""

    events_xy: np.ndarray  # (n, 2) normalized coordinates
    ttc: np.ndarray  # (n,) time-to-collision in seconds for each event
    ttc_uncertainty: np.ndarray  # (n,) uncertainty estimate
    flow_divergence: np.ndarray  # (n,) local divergence at each event
    looming_mask: np.ndarray  # (n,) bool mask: events with positive divergence
    ttc_grid: np.ndarray  # (H, W) binned TTC map for spatial awareness


class DenseTTCEstimator:
    """
    Computes dense per-event time-to-collision from normal flow.

    Uses the relation TTC ≈ 1 / div(v) where div(v) is the optical flow
    divergence. For normal flow (which only measures motion perpendicular to
    edges), we use a local neighborhood estimation of the divergence from
    the component of flow normal to intensity gradients.

    Two modes:
      - Exact:  TTC = θ / θ̇  (Lee's tau, requires angular size tracking)
      - Proxy:  TTC ≈ 1 / max(div(v), ε)  (from flow divergence)
    """

    def __init__(
        self,
        min_divergence: float = 1e-4,
        max_ttc: float = 30.0,
        min_ttc: float = 0.05,
        neighborhood_radius: float = 0.02,
        ttc_grid_resolution: Tuple[int, int] = (64, 64),
        low_light_mode: bool = False,
        noise_scale: float = 0.01,
    ):
        """
        Args:
            min_divergence: Minimum divergence to avoid division by zero
            max_ttc: Maximum TTC to report (clamp upper bound)
            min_ttc: Minimum TTC (clamp lower bound)
            neighborhood_radius: Radius for local divergence estimation
            ttc_grid_resolution: (H, W) for the binned TTC map
            low_light_mode: Enable EV-TTC-style low-light handling
            noise_scale: Expected noise level for uncertainty propagation
        """
        self.min_divergence = min_divergence
        self.max_ttc = max_ttc
        self.min_ttc = min_ttc
        self.neighborhood_radius = neighborhood_radius
        self.ttc_grid_resolution = ttc_grid_resolution
        self.low_light_mode = low_light_mode
        self.noise_scale = noise_scale

    def compute(
        self,
        events_t: torch.Tensor,
        events_xy: torch.Tensor,
        flow_predictions: torch.Tensor,
        flow_uncertainty: torch.Tensor,
    ) -> DenseTTCMap:
        """
        Compute dense TTC from normal flow.

        Args:
            events_t: (n,) event timestamps
            events_xy: (n, 2) normalized coordinates
            flow_predictions: (n, 2) normal flow vectors
            flow_uncertainty: (n,) flow uncertainty

        Returns:
            DenseTTCMap with per-event TTC + binned grid
        """
        xy = events_xy.numpy()
        flow = flow_predictions.numpy()
        uncert = flow_uncertainty.numpy().flatten()
        t = events_t.numpy().flatten()

        n = xy.shape[0]
        if n < 10:
            return self._empty_map(xy)

        # ---- Step 1: Local divergence estimation ----
        # Use spatial k-NN to estimate flow divergence at each event
        divergence = np.zeros(n, dtype=np.float32)
        local_flow_var = np.zeros(n, dtype=np.float32)

        # Build spatial index (simple grid binning for speed)
        grid_size = self.neighborhood_radius
        grid: dict = {}
        for i in range(n):
            cell = (int(xy[i, 0] / grid_size), int(xy[i, 1] / grid_size))
            grid.setdefault(cell, []).append(i)

        for i in range(n):
            cx, cy = xy[i]
            cell = (int(cx / grid_size), int(cy / grid_size))
            neighbors = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nc = (cell[0] + dx, cell[1] + dy)
                    if nc in grid:
                        neighbors.extend(grid[nc])

            if len(neighbors) < 5:
                continue

            # Compute divergence from flow in neighborhood
            nbr_xy = xy[neighbors]
            nbr_flow = flow[neighbors]
            nbr_weights = 1.0 / (
                np.linalg.norm(nbr_xy - xy[i : i + 1], axis=-1) + self.noise_scale
            )
            nbr_weights = nbr_weights / (nbr_weights.sum() + 1e-10)

            # Divergence = trace of flow Jacobian = du/dx + dv/dy
            # Estimate via weighted least-squares in neighborhood
            dx = nbr_xy[:, 0] - cx
            dy = nbr_xy[:, 1] - cy
            du = nbr_flow[:, 0] - flow[i, 0]
            dv = nbr_flow[:, 1] - flow[i, 1]

            # Simple approximation: weighted sum of (du/dx + dv/dy)
            du_dx = np.sum(nbr_weights * du * dx) / (np.sum(nbr_weights * dx**2) + 1e-10)
            dv_dy = np.sum(nbr_weights * dv * dy) / (np.sum(nbr_weights * dy**2) + 1e-10)
            divergence[i] = du_dx + dv_dy

            # Local flow variance (for uncertainty)
            mean_flow = np.average(nbr_flow, axis=0, weights=nbr_weights)
            local_flow_var[i] = np.mean(
                np.linalg.norm(nbr_flow - mean_flow[None, :], axis=-1) ** 2
            )

        # ---- Step 2: Convert divergence to TTC ----
        # Positive divergence → object approaching (looming)
        looming_mask = divergence > self.min_divergence

        ttc = np.full(n, self.max_ttc, dtype=np.float32)
        ttc[looming_mask] = 1.0 / divergence[looming_mask]
        ttc = np.clip(ttc, self.min_ttc, self.max_ttc)

        # For non-looming events, use flow magnitude as proxy
        # (large lateral motion at close range → near object)
        non_looming = ~looming_mask
        flow_mag = np.linalg.norm(flow, axis=-1)
        for i in np.where(non_looming)[0]:
            if flow_mag[i] > 0.5:  # Fast lateral motion → likely close
                ttc[i] = min(1.0 / (flow_mag[i] + 1e-4), self.max_ttc)

        # ---- Step 3: Uncertainty propagation ----
        # TTC uncertainty from flow uncertainty + divergence uncertainty
        ttc_uncertainty = np.full(n, 1.0, dtype=np.float32)
        if self.low_light_mode:
            # EV-TTC style: scale uncertainty by event rate (proxy for SNR)
            # Lower event rate → higher uncertainty in low light
            event_rate = np.zeros(n)
            for i in range(n):
                if len(t) > 1:
                    dt = max(t[-1] - t[0], 1e-6)
                    event_rate[i] = n / dt
            snr_scale = 1.0 / (event_rate + 1.0)
            low_light_factor = np.clip(snr_scale * 100, 0.5, 5.0)
        else:
            low_light_factor = 1.0

        ttc_uncertainty[looming_mask] = (
            low_light_factor[looming_mask]
            * (uncert[looming_mask] + local_flow_var[looming_mask] / 10.0)
            * 2.0
        )
        ttc_uncertainty[~looming_mask] = 1.0  # High uncertainty for non-looming

        # ---- Step 4: Binned TTC grid ----
        H, W = self.ttc_grid_resolution
        ttc_grid = np.full((H, W), self.max_ttc, dtype=np.float32)
        weight_grid = np.zeros((H, W), dtype=np.float32)

        # Map normalized coords [-1, 1] to grid indices
        xi = np.clip(((xy[:, 0] + 1.0) / 2.0 * (W - 1)).astype(int), 0, W - 1)
        yi = np.clip(((xy[:, 1] + 1.0) / 2.0 * (H - 1)).astype(int), 0, H - 1)

        for i in range(n):
            w = 1.0 / (ttc_uncertainty[i] + 1e-6)
            if ttc[i] < ttc_grid[yi[i], xi[i]]:
                ttc_grid[yi[i], xi[i]] = ttc[i] * w
                weight_grid[yi[i], xi[i]] = w

        # Normalize grid
        valid = weight_grid > 0
        ttc_grid[valid] /= weight_grid[valid]

        return DenseTTCMap(
            events_xy=xy,
            ttc=ttc,
            ttc_uncertainty=ttc_uncertainty,
            flow_divergence=divergence,
            looming_mask=looming_mask,
            ttc_grid=ttc_grid,
        )

    def get_threat_regions(
        self, ttc_map: DenseTTCMap, threshold: float = 1.5
    ) -> List[dict]:
        """
        Extract connected threat regions from the dense TTC map.

        Returns list of dicts with:
            - 'center': (cx, cy) in normalized coords
            - 'min_ttc': minimum TTC in region
            - 'size': angular size of region
            - 'bearing': angle relative to optical axis
        """
        H, W = self.ttc_grid_resolution
        grid = ttc_map.ttc_grid
        threatened = grid < threshold

        if not threatened.any():
            return []

        # Simple connected components labeling
        labels = np.zeros((H, W), dtype=int)
        current_label = 0
        equivalences = []

        for y in range(H):
            for x in range(W):
                if not threatened[y, x]:
                    continue

                left = labels[y, x - 1] if x > 0 and threatened[y, x - 1] else 0
                up = labels[y - 1, x] if y > 0 and threatened[y - 1, x] else 0

                if left == 0 and up == 0:
                    current_label += 1
                    labels[y, x] = current_label
                elif left != 0 and up == 0:
                    labels[y, x] = left
                elif left == 0 and up != 0:
                    labels[y, x] = up
                elif left == up:
                    labels[y, x] = left
                else:
                    labels[y, x] = min(left, up)
                    equivalences.append((left, up))

        # Apply equivalences
        changed = True
        while changed:
            changed = False
            for a, b in equivalences:
                mask = labels == b
                if mask.any():
                    labels[mask] = a
                    changed = True

        # Extract regions
        regions = []
        for label in range(1, current_label + 1):
            mask = labels == label
            if mask.sum() < 2:
                continue

            ys, xs = np.where(mask)
            ttc_vals = grid[ys, xs]

            # Region center in normalized coords
            cy = (ys.mean() / (H - 1)) * 2.0 - 1.0
            cx = (xs.mean() / (W - 1)) * 2.0 - 1.0

            # Angular size (approximate)
            height = (ys.max() - ys.min()) / H * 2.0
            width = (xs.max() - xs.min()) / W * 2.0
            size = max(height, width)

            regions.append(
                {
                    "center": np.array([cx, cy]),
                    "min_ttc": float(ttc_vals.min()),
                    "mean_ttc": float(ttc_vals.mean()),
                    "size": size,
                    "bearing": np.arctan2(cx, -cy),
                    "pixel_count": int(mask.sum()),
                }
            )

        regions.sort(key=lambda r: r["min_ttc"])
        return regions

    def _empty_map(self, xy: np.ndarray) -> DenseTTCMap:
        """Return an empty TTC map."""
        H, W = self.ttc_grid_resolution
        return DenseTTCMap(
            events_xy=xy,
            ttc=np.array([]),
            ttc_uncertainty=np.array([]),
            flow_divergence=np.array([]),
            looming_mask=np.array([]),
            ttc_grid=np.full((H, W), self.max_ttc, dtype=np.float32),
        )
