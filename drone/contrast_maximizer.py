"""
Contrast Maximization Flow Refinement

Augments the VecKM normal flow estimator with contrast-maximization-based
flow refinement. The core idea from Gallego et al. (CVPR 2018) and the
CMax-SLAM framework (tub-rip/cmax_slam):

  * For static-scene regions (where event generation is caused by ego-motion
    rather than moving objects), contrast maximization produces sharper,
    more temporally-coherent flow estimates.
  * Event warping: Map events from a time window to a reference time t_ref
    using candidate flow hypotheses. The hypothesis that produces the
    sharpest "image of warped events" (IWE) is the most likely true flow.
  * Variance-of-IWE is used as the contrast measure. Lower variance = more
    focused IWE = better flow estimate.

Integration:
  * The CMax refiner operates on flow vectors that have high uncertainty
    (where the standard V ecKM encoder is less confident).
  * It tests perturbed flow hypotheses around the VecKM estimate and
    selects the one that sharpens the IWE.
  * This is particularly effective for:
      - Static backgrounds seen during ego-motion
      - Low-texture regions where the encoder struggles
      - Edge regions where normal flow is orthogonal to gradient

 References:
  - Gallego et al., "A Unifying Contrast Maximization Framework for
    Event Cameras", CVPR 2018
  - Shiba et al., "Secrets of Event-based Optical Flow, Depth and
    Ego-motion Estimation by Contrast Maximization", TPAMI 2024
  - tub-rip/cmax_slam, "CMax-SLAM: Event-based Rotational-Motion
    Bundle Adjustment and SLAM System using Contrast Maximization",
    TRO 2024
"""

import numpy as np
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass
import torch


@dataclass
class CMaxResult:
    """Output of the contrast maximization flow refiner."""

    refined_flow: np.ndarray  # (n, 2) refined flow vectors
    original_flow: np.ndarray  # (n, 2) original VecKM flow
    refinement_delta: np.ndarray  # (n, 2) difference vector
    contrast_improvement: np.ndarray  # (n,) IWE contrast improvement
    was_refined: np.ndarray  # (n,) bool: which events were refined
    iwe_variance_before: np.ndarray  # (n,) local IWE variance (before)
    iwe_variance_after: np.ndarray  # (n,) local IWE variance (after)


class ContrastMaximizer:
    """
    Refines normal flow estimates using contrast maximization.

    For events in static or slowly-moving scenes, contrast maximization
    provides a powerful alternative to learning-based flow estimation.
    This class tests perturbed flow hypotheses around the VecKM estimate
    and selects the one that produces the sharpest Image of Warped Events.

    Architecture:
      1. Take VecKM flow estimate f_i for event e_i at (x_i, y_i, t_i)
      2. For a time window T around e_i, warp nearby events to t_i
         using flow hypothesis f_h = f_i + delta_f
      3. Compute IWE by accumulating warped events in a local spatial patch
      4. Measure contrast as 1/variance of the IWE histogram
      5. Select f_h that maximizes contrast
    """

    def __init__(
        self,
        patch_size: int = 15,  # pixels — local patch for IWE computation
        time_window_us: float = 1000.0,  # μs — time window for warping
        num_hypotheses: int = 5,  # Number of flow perturbations to test
        perturbation_scale: float = 0.3,  # Fraction of flow magnitude to perturb
        min_contrast_improvement: float = 0.05,  # Min improvement to accept ref inement
        min_events_for_iwe: int = 10,  # Min events in patch for valid IWE
        device: str = "cpu",
    ):
        """
        Args:
            patch_size: Spatial patch size in pixels for IWE computation
            time_window_us: Time window in microseconds for event warping
            num_hypotheses: Number of flow direction + magnitude hypotheses
            perturbation_scale: Scale of flow perturbation relative to flow mag
            min_contrast_improvement: Min contrast improvement to accept refinement
            min_events_for_iwe: Minimum events in a patch for valid IWE
            device: "cpu" or "cuda"
        """
        self.patch_size = patch_size
        self.time_window_us = time_window_us
        self.num_hypotheses = num_hypotheses
        self.perturbation_scale = perturbation_scale
        self.min_contrast_improvement = min_contrast_improvement
        self.min_events_for_iwe = min_events_for_iwe
        self.device = device

        # Half window in seconds
        self.half_window_s = time_window_us / 2.0 / 1e6

    def refine(
        self,
        events_t: np.ndarray,  # (n,) event timestamps in seconds
        events_xy: np.ndarray,  # (n, 2) event pixel coordinates
        flow: np.ndarray,  # (n, 2) initial flow (pixels/sec)
        flow_uncertainty: Optional[np.ndarray] = None,  # (n,) uncertainty
    ) -> CMaxResult:
        """
        Refine flow vectors using contrast maximization.

        Only refines events with high uncertainty or where IWE contrast
        can be meaningfully improved.

        Args:
            events_t: Sorted event timestamps in seconds
            events_xy: Pixel coordinates (width, height)
            flow: Initial normal flow estimates (pixels/sec)
            flow_uncertainty: Optional uncertainty per event

        Returns:
            CMaxResult with refined flow + contrast metrics
        """
        n = len(events_t)
        if n < self.min_events_for_iwe:
            return self._empty_result(flow)

        # Build spatial + temporal index
        patch_half = self.patch_size // 2
        grid = self._build_spatiotemporal_grid(
            events_t, events_xy, self.half_window_s
        )

        # Pre-allocate results
        refined_flow = flow.copy()
        contrast_improvement = np.zeros(n, dtype=np.float32)
        iwe_var_before = np.zeros(n, dtype=np.float32)
        iwe_var_after = np.zeros(n, dtype=np.float32)
        was_refined = np.zeros(n, dtype=bool)

        # Process each event
        for i in range(n):
            tx, ty = events_xy[i]
            tt = events_t[i]

            # Get nearby events (spatial + temporal)
            neighbors = self._get_nearby_events(
                grid, tx, ty, tt, patch_half
            )

            if len(neighbors) < self.min_events_for_iwe:
                continue

            nbr_xy = events_xy[neighbors]
            nbr_t = events_t[neighbors]

            # Relative times from anchor event
            dt = nbr_t - tt  # seconds

            # Compute baseline IWE contrast
            baseline_iwe = self._compute_iwe(nbr_xy, dt, flow[i])
            var_before = self._compute_contrast(baseline_iwe)
            iwe_var_before[i] = var_before

            if var_before < 1e-6:
                continue  # Already sharp

            # Generate flow hypotheses
            flow_mag = max(np.linalg.norm(flow[i]), 1e-6)
            hypotheses = self._generate_hypotheses(
                flow[i], flow_mag, self.num_hypotheses
            )

            # Test each hypothesis
            best_var = var_before
            best_flow = flow[i].copy()
            best_iwe = baseline_iwe.copy()

            for h_flow in hypotheses:
                iwe = self._compute_iwe(nbr_xy, dt, h_flow)
                var_after = self._compute_contrast(iwe)

                # Lower variance = sharper image = better flow
                if var_after < best_var:
                    best_var = var_after
                    best_flow = h_flow
                    best_iwe = iwe

            # Accept refinement if significantly better
            improvement = (var_before - best_var) / (var_before + 1e-10)
            if improvement > self.min_contrast_improvement:
                refined_flow[i] = best_flow
                was_refined[i] = True
                contrast_improvement[i] = improvement
                iwe_var_after[i] = best_var

        # Compute delta
        refinement_delta = refined_flow - flow

        return CMaxResult(
            refined_flow=refined_flow,
            original_flow=flow,
            refinement_delta=refinement_delta,
            contrast_improvement=contrast_improvement,
            was_refined=was_refined,
            iwe_variance_before=iwe_var_before,
            iwe_variance_after=iwe_var_after,
        )

    def _build_spatiotemporal_grid(
        self,
        events_t: np.ndarray,
        events_xy: np.ndarray,
        time_window_s: float,
    ) -> dict:
        """
        Build a sparse grid index: cell → list of event indices.

        Grid cell key: (spatial_bin_x, spatial_bin_y, temporal_bin)
        """
        n = len(events_t)
        grid: dict = {}

        # Spatial binning: patch_size bins
        spatial_bin_size = max(self.patch_size, 1)

        for i in range(n):
            bx = int(events_xy[i, 0]) // spatial_bin_size
            by = int(events_xy[i, 1]) // spatial_bin_size
            bt = int(events_t[i] / time_window_s) if time_window_s > 0 else 0
            key = (bx, by, bt)
            if key not in grid:
                grid[key] = []
            grid[key].append(i)

        return grid

    def _get_nearby_events(
        self,
        grid: dict,
        cx: float,
        cy: float,
        ct: float,
        patch_half: int,
    ) -> np.ndarray:
        """
        Get event indices that are nearby in space and time.
        """
        spatial_bin_size = max(self.patch_size, 1)
        bx = int(cx) // spatial_bin_size
        by = int(cy) // spatial_bin_size
        bt = int(ct / self.half_window_s) if self.half_window_s > 0 else 0

        neighbors = []
        t_range = max(1, int(self.time_window_us / 2.0 / 1e6 / self.half_window_s))

        # Search adjacent cells
        for dtx in range(-1, 2):
            for dty in range(-1, 2):
                for dtt in range(-t_range, t_range + 1):
                    key = (bx + dtx, by + dty, bt + dtt)
                    if key in grid:
                        neighbors.extend(grid[key])

        if not neighbors:
            return np.array([], dtype=int)

        nbr = np.array(neighbors, dtype=int)
        nbr_xy = np.array(
            [
                (float(
                    bx * spatial_bin_size + spatial_bin_size // 2),
                 float(by * spatial_bin_size + spatial_bin_size // 2),
                )
                for _ in nbr
            ]
        )

        # Filter by spatial distance
        # Use a cheap approximate check (no need for exact for speed)
        spatial_ok = (
            np.abs(nbr_xy[:, 0] - cx) < self.patch_size
        ) & (
            np.abs(nbr_xy[:, 1] - cy) < self.patch_size
        )

        return nbr[spatial_ok]

    def _compute_iwe(
        self,
        events_xy: np.ndarray,  # (m, 2)
        dt: np.ndarray,  # (m,) time offsets in seconds
        flow: np.ndarray,  # (2,) flow vector (pixels/sec)
    ) -> np.ndarray:
        """
        Compute the Image of Warped Events for a local patch.

        Warp each event from its original time to the reference time
        using the flow hypothesis, then accumulate into a spatial patch.

        Args:
            events_xy: Event pixel coordinates
            dt: Time offsets from reference (negative = before, positive = after)
            flow: Candidate flow vector

        Returns:
            iwe_image: (patch_size, patch_size) float32 accumulation
        """
        m = len(events_xy)
        patch = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)

        half = self.patch_size // 2

        # Reference event is at (half, half) in the patch
        ref_x = events_xy[0, 0]  # Anchor at first event's position
        ref_y = events_xy[0, 1]

        # Warp each event: compensate for motion from event time to ref time
        for j in range(m):
            # Undo motion: move event back to where it would be at t_ref
            warped_x = events_xy[j, 0] - flow[0] * dt[j]
            warped_y = events_xy[j, 1] - flow[1] * dt[j]

            # Map to patch coordinates
            px = int(warped_x - ref_x + half)
            py = int(warped_y - ref_y + half)

            # Clip to patch
            if 0 <= px < self.patch_size and 0 <= py < self.patch_size:
                patch[py, px] += 1.0

        return patch

    def _compute_contrast(self, iwe: np.ndarray) -> float:
        """
        Compute contrast of the IWE.

        Uses variance as a proxy for contrast.
        Lower variance = sharper, more focused IWE = better flow.

        Returns variance (0 = perfectly focused, higher = blurry).
        """
        total = iwe.sum()
        if total < 2:
            return 1e10  # Too sparse — effectively no contrast

        # Normalize to probability distribution
        p = iwe / (total + 1e-10)

        # Weighted centroid
        ys, xs = np.meshgrid(
            np.arange(self.patch_size, dtype=np.float32),
            np.arange(self.patch_size, dtype=np.float32),
            indexing="ij",
        )
        cx = np.sum(p * xs)
        cy = np.sum(p * ys)

        # Variance = weighted sum of squared distances from centroid
        dx = xs - cx
        dy = ys - cy
        var_x = np.sum(p * dx * dx)
        var_y = np.sum(p * dy * dy)

        # Combined variance (trace of covariance)
        var = var_x + var_y

        # Scale by patch-size to normalize
        var = var / (self.patch_size * self.patch_size)

        return float(var)

    def _generate_hypotheses(
        self,
        base_flow: np.ndarray,  # (2,)
        flow_magnitude: float,
        num_hypotheses: int,
    ) -> List[np.ndarray]:
        """
        Generate perturbed flow hypotheses around the base estimate.

        Perturbs both direction (angle) and magnitude.
        """
        hyps = [base_flow.copy()]  # Always include original

        angle = np.arctan2(base_flow[1], base_flow[0])
        scale = self.perturbation_scale * max(flow_magnitude, 0.1)

        # Direction perturbations
        for d_angle in np.linspace(-0.5, 0.5, max(3, num_hypotheses // 2)):
            if abs(d_angle) < 1e-6:
                continue
            new_angle = angle + d_angle
            h_flow = np.array([flow_magnitude * np.cos(new_angle),
                               flow_magnitude * np.sin(new_angle)])
            hyps.append(h_flow)

        # Magnitude perturbations
        for d_mag in [1.0 - scale / max(flow_magnitude, 0.1),
                       1.0 + scale / max(flow_magnitude, 0.1)]:
            if abs(d_mag - 1.0) < 0.01:
                continue
            h_flow = base_flow * d_mag
            hyps.append(h_flow)

        # Limit to num_hypotheses
        return hyps[:num_hypotheses]

    def _empty_result(self, flow: np.ndarray) -> CMaxResult:
        """Return empty result when there are insufficient events."""
        n = len(flow)
        zeros = np.zeros(n, dtype=np.float32)
        return CMaxResult(
            refined_flow=flow.copy(),
            original_flow=flow.copy(),
            refinement_delta=np.zeros_like(flow),
            contrast_improvement=zeros,
            was_refined=np.zeros(n, dtype=bool),
            iwe_variance_before=zeros,
            iwe_variance_after=zeros,
        )

    def refine_masked(
        self,
        events_t: np.ndarray,
        events_xy: np.ndarray,
        flow: np.ndarray,
        mask: np.ndarray,  # (n,) bool: which events to refine
        **kwargs,
    ) -> CMaxResult:
        """
        Refine only events where mask is True.
        Unmasked events are passed through unchanged.
        """
        n = len(flow)
        result = self._empty_result(flow)

        if not mask.any():
            return result

        # Refine masked subset
        sub_result = self.refine(
            events_t[mask],
            events_xy[mask],
            flow[mask],
            **kwargs,
        )

        # Merge back
        result.refined_flow[mask] = sub_result.refined_flow
        result.refinement_delta[mask] = sub_result.refinement_delta
        result.contrast_improvement[mask] = sub_result.contrast_improvement
        result.was_refined[mask] = sub_result.was_refined
        result.iwe_variance_before[mask] = sub_result.iwe_variance_before
        result.iwe_variance_after[mask] = sub_result.iwe_variance_after

        return result