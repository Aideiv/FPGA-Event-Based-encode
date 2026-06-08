"""
Monocular Depth Estimation from Events (E2Depth / RAMNet style)

Provides depth estimation from event streams using a lightweight CNN
that maps event frames + normal flow to per-pixel depth.

References:
  - Hidalgo-Carrio et al., "Learning Monocular Dense Depth from Events"
    (E2Depth), 3DV 2020. https://arxiv.org/abs/2010.08350
  - Gehrig et al., "Combining Events and Frames using Recurrent
    Asynchronous Multimodal Networks for Monocular Depth Prediction"
    (RAMNet), RA-L 2021. https://github.com/uzh-rpg/rpg_ramnet
  - Bonazzi et al., CVPRW 2025: Depth-aware evasion decisions

Integration with collision avoidance:
  * Depth converts the 2D image-plane threat assessment into
    3D spatial awareness.
  * A 2D threat (e.g., looming at pixel (x,y)) with a depth estimate
    gives real-world 3D position relative to the drone.
  * This enables:
      - Prioritizing closer threats over farther ones
      - Computing safe 3D evasion trajectories (not just 2D directions)
      - Estimating object size from angular size + depth
      - Detecting thin/transparent obstacles via depth discontinuities
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import os


@dataclass
class DepthMap:
    """Per-pixel depth map from event data."""

    depth: np.ndarray  # (H, W) depth values in meters
    confidence: np.ndarray  # (H, W) confidence 0-1
    min_depth: float
    max_depth: float
    median_depth: float
    timestamp: float


# ---------------------------------------------------------------------------
# Lightweight Depth CNN (E2Depth-style)
# ---------------------------------------------------------------------------
class EventDepthNet(nn.Module):
    """Lightweight U-Net style depth estimation from event frames.

    Input:  Event frame (K-channel temporal stack) or (1, H, W)
    Output: (1, H, W) depth map in log-space, plus (1, H, W) confidence

    Designed for < 5ms inference on GPU, < 50ms on CPU.
    """

    def __init__(
        self,
        input_channels: int = 1,
        height: int = 80,
        width: int = 80,
        base_filters: int = 16,
    ):
        super().__init__()

        self.height = height
        self.width = width

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True),
        )

        # Decoder (upsampling)
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_filters * 8, base_filters * 4, 3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_filters * 4, base_filters * 2, 3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_filters * 2, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )

        # Output heads
        self.depth_head = nn.Conv2d(base_filters, 1, 3, padding=1)
        self.conf_head = nn.Sequential(
            nn.Conv2d(base_filters, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, C, H, W) event frame or temporal stack

        Returns:
            depth_log: (B, 1, H, W) log-depth (lower = closer)
            confidence: (B, 1, H, W) confidence in [0, 1]
        """
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d3 = self.dec3(e4) + e3
        d2 = self.dec2(d3) + e2
        d1 = self.dec1(d2) + e1

        depth_log = self.depth_head(d1)
        confidence = self.conf_head(d1)

        return depth_log, confidence


# ---------------------------------------------------------------------------
# Depth Estimator
# ---------------------------------------------------------------------------
class DepthEstimator:
    """Monocular depth estimation from event streams.

    Uses a lightweight U-Net to predict per-pixel depth from event frames.
    Can operate on:
      - Raw event frames (accumulated over ~10ms windows)
      - Temporal stacks of event frames (for motion parallax cues)
      - Flow-augmented inputs (flow magnitude channel for depth-from-motion)

    Integration with collision avoidance:
      depth_estimator = DepthEstimator()
      depth_map = depth_estimator.predict(event_frame)
      # 2D threat at (x, y) with depth d becomes a 3D threat:
      threat_xyz = pixel_to_3d(x, y, depth_map.depth[y, x], camera_matrix)
    """

    def __init__(
        self,
        model: Optional[EventDepthNet] = None,
        input_height: int = 80,
        input_width: int = 80,
        min_depth: float = 0.3,  # meters — closest depth
        max_depth: float = 30.0,  # meters — farthest depth
        device: Optional[str] = None,
    ):
        """
        Args:
            model: Pre-built EventDepthNet or None to create
            input_height: Input frame height
            input_width: Input frame width
            min_depth: Minimum plausible depth (meters)
            max_depth: Maximum plausible depth (meters)
            device: "cpu", "cuda", or None (auto)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if model is None:
            self.model = EventDepthNet(
                input_channels=5,  # Default: temporal stack
                height=input_height,
                width=input_width,
            )
        else:
            self.model = model

        self.model = self.model.to(self.device)
        self.model.eval()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.height = input_height
        self.width = input_width

    def load_weights(self, path: str):
        """Load pretrained depth model weights."""
        if not os.path.exists(path):
            print(f"  [DepthEstimator] Warning: weights not found at {path}")
            print(f"  Model will use random initialization.")
            return

        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        print(f"  Loaded depth weights from {path}")

    def predict(
        self,
        event_frame: np.ndarray,  # (C, H, W) event frame or temporal stack
        timestamp: float = 0.0,
    ) -> DepthMap:
        """Predict depth map from event frame.

        Args:
            event_frame: (C, H, W) event frame or temporal stack.
                         C=1 for single frame, C=K for temporal stack.
            timestamp: Current timestamp for logging

        Returns:
            DepthMap with depth in meters + confidence
        """
        import time
        t0 = time.perf_counter()

        # Prepare input
        if event_frame.ndim == 2:
            x = torch.from_numpy(event_frame).float().unsqueeze(0).unsqueeze(0)
        elif event_frame.ndim == 3:
            x = torch.from_numpy(event_frame).float().unsqueeze(0)
        else:
            x = torch.from_numpy(event_frame).float()  # (B, C, H, W)

        x = x.to(self.device)

        # Forward pass
        with torch.no_grad():
            depth_log, confidence = self.model(x)

        # Convert log-depth to linear depth
        depth_log_np = depth_log.squeeze().cpu().numpy()  # (H, W)
        confidence_np = confidence.squeeze().cpu().numpy()

        # Sigmoid-like mapping: depth = max_depth / (1 + exp(-depth_log))
        # This maps the unbounded output to [0, max_depth]
        depth_np = self.max_depth / (1.0 + np.exp(-np.clip(depth_log_np, -5, 5)))
        depth_np = np.clip(depth_np, self.min_depth, self.max_depth)

        # Mask depth where confidence is low
        depth_np[confidence_np < 0.1] = self.max_depth

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        return DepthMap(
            depth=depth_np,
            confidence=confidence_np,
            min_depth=float(np.min(depth_np[np.isfinite(depth_np)])),
            max_depth=float(np.max(depth_np[np.isfinite(depth_np)])),
            median_depth=float(np.median(depth_np[np.isfinite(depth_np)])),
            timestamp=timestamp,
        )

    def get_depth_at(
        self, depth_map: DepthMap, x_norm: float, y_norm: float
    ) -> Tuple[float, float]:
        """Sample depth at normalized coordinates (-1 to 1).

        Returns (depth_meters, confidence).
        """
        h, w = depth_map.depth.shape
        px = int((x_norm + 1.0) / 2.0 * (w - 1))
        py = int((y_norm + 1.0) / 2.0 * (h - 1))
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))
        return float(depth_map.depth[py, px]), float(depth_map.confidence[py, px])

    def get_closest_object(
        self,
        depth_map: DepthMap,
        min_confidence: float = 0.2,
    ) -> Optional[Dict]:
        """Find the closest confident depth region.

        Returns dict with depth, position, angular size, or None.
        """
        confident = depth_map.confidence > min_confidence
        if not confident.any():
            return None

        depths = depth_map.depth.copy()
        depths[~confident] = 999.0

        min_idx = np.unravel_index(np.argmin(depths), depths.shape)
        cy, cx = min_idx

        # Normalized coordinates (-1 to 1)
        h, w = depths.shape
        xn = cx / (w - 1) * 2.0 - 1.0
        yn = cy / (h - 1) * 2.0 - 1.0

        # Find region size (depth within 25% of minimum)
        region = depths < (depths[cy, cx] * 1.25)
        region_ys, region_xs = np.where(region)
        angular_width = (region_xs.max() - region_xs.min()) / w * 2.0
        angular_height = (region_ys.max() - region_ys.min()) / h * 2.0

        return {
            "depth_m": float(depths[cy, cx]),
            "position_xy": np.array([xn, yn]),
            "angular_size": max(angular_width, angular_height),
            "confidence": float(depth_map.confidence[cy, cx]),
            "bearing_rad": np.arctan2(xn, -yn),
        }

    def to_onnx(self, path: str = "models/models/depth_net.onnx"):
        """Export to ONNX for FPGA DPU or CPU inference."""
        dummy = torch.randn(1, 5, self.height, self.width).to(self.device)
        torch.onnx.export(
            self.model,
            dummy,
            path,
            input_names=["event_frame"],
            output_names=["depth_log", "confidence"],
            opset_version=13,
        )
        print(f"  Exported depth ONNX to {path}")
        return path