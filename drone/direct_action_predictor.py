"""
Direct Action Predictor — CNN-based evasion from event frames

Implements the direct action prediction approach from Bonazzi et al.
"Towards Low-Latency Event-based Obstacle Avoidance on an FPGA-Drone",
CVPRW 2025:

  * Input: Temporal stack of K event frames (K × 80 × 80)
  * Output: 5-class evasion action (STAY, LEFT, RIGHT, UP, DOWN)
  * CNN: DPU-optimized lightweight CNN → ~0.94ms on FPGA DPU

This is a Python reference implementation of the CNN pipeline.
The paper uses a custom 5-layer CNN deployed on a Xilinx DPU.
Our reference mirrors that architecture for HIL testing before
FPGA synthesis.

Architecture (from paper, adapted for CPU/GPU):
  Input: (K, 80, 80) temporal stack
  Conv1: 32 filters 3×3, stride 2, BN, ReLU → (16, 40, 40)
  Conv2: 64 filters 3×3, stride 2, BN, ReLU → (32, 20, 20)
  Conv3: 128 filters 3×3, stride 2, BN, ReLU → (64, 10, 10)
  Conv4: 128 filters 3×3, stride 1, BN, ReLU → (64, 10, 10)
  Gap: Global average pooling → (64,)
  FC:  64 → 5 (STAY, LEFT, RIGHT, UP, DOWN)

The direct action predictor is complementary to our flow-based pipeline
and can be used as:
  1. A fast first-pass filter (1kHz) to trigger the flow pipeline (100Hz)
  2. A fallback when flow estimation confidence is low
  3. A redundant safety check for the graded evasion levels
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
import os

if TYPE_CHECKING:
    from .event_frame_aggregator import EventFrameTemporalStack


# ---------------------------------------------------------------------------
# Action Classes (mirrors paper's 5-class output)
# ---------------------------------------------------------------------------
class EvasionAction(Enum):
    """Discrete evasion action from the CNN classifier."""

    STAY = 0  # No movement needed / hold position
    LEFT = 1  # Dodge left
    RIGHT = 2  # Dodge right
    UP = 3  # Climb
    DOWN = 4  # Descend


# Map paper's 5 actions to the action names used in the drone
ACTION_NAMES = {
    0: "STAY",
    1: "LEFT",
    2: "RIGHT",
    3: "UP",
    4: "DOWN",
}

# Map actions to velocity commands (body frame)
ACTION_VELOCITY = {
    0: (0.0, 0.0, 0.0),  # STAY: hover
    1: (0.0, -2.0, 0.0),  # LEFT:  lateral left at 2 m/s
    2: (0.0, 2.0, 0.0),  # RIGHT: lateral right at 2 m/s
    3: (0.0, 0.0, 1.5),  # UP:    climb at 1.5 m/s
    4: (0.0, 0.0, -1.0),  # DOWN:  descend at 1 m/s
}


@dataclass
class DirectActionPrediction:
    """Output of the direct action predictor."""

    action: EvasionAction  # Predicted evasion action
    probabilities: np.ndarray  # (5,) softmax probabilities
    confidence: float  # Max probability (0-1)
    velocity: Tuple[float, float, float]  # (vx, vy, vz) in m/s
    latency_ms: float  # Inference time in ms
    frame_count: int  # Number of frames this prediction is based on


# ---------------------------------------------------------------------------
# Lightweight DPU-Style CNN Model
# ---------------------------------------------------------------------------
class DPULightNet(nn.Module):
    """Lightweight 3D/2D CNN optimized for FPGA DPU deployment.

    Matches the architecture described in Bonazzi et al. (2025):
      - Input: (K, H, W) temporal stack of event frames
      - 4 convolutional layers with stride-2 downsampling
      - Global average pooling
      - Single FC layer → 5-way classification

    Designed for < 1ms inference on Xilinx DPU (B4096).
    """

    def __init__(
        self,
        input_channels: int = 5,  # K temporal stack frames
        input_height: int = 80,
        input_width: int = 80,
        num_classes: int = 5,
        conv_filters: Tuple[int, ...] = (32, 64, 128, 128),
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes

        # Build convolutional stack
        layers = []
        in_ch = input_channels
        h, w = input_height, input_width

        for i, filters in enumerate(conv_filters):
            stride = 2 if i < 3 else 1  # First 3 layers do stride-2
            padding = 1

            layers.append(
                nn.Conv2d(in_ch, filters, kernel_size=3, stride=stride, padding=padding, bias=False)
            )
            layers.append(nn.BatchNorm2d(filters))
            layers.append(nn.ReLU(inplace=True))

            if dropout > 0 and i == len(conv_filters) - 1:
                layers.append(nn.Dropout2d(dropout))

            in_ch = filters
            h = h // stride if stride > 1 else h
            w = w // stride if stride > 1 else w

        self.conv_stack = nn.Sequential(*layers)

        # Global average pooling → (batch, filters, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head
        self.classifier = nn.Linear(conv_filters[-1], num_classes)

        # Feature dimensions for export
        self.feature_dim = conv_filters[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, K, H, W) temporal stack of event frames

        Returns:
            logits: (B, num_classes) raw logits
        """
        # Handle 3D input (B, K, H, W) by treating K as channels
        if x.dim() == 4 and x.shape[1] == self.input_channels:
            # Already (B, K, H, W) → treat K as channels
            pass
        elif x.dim() == 4:
            # Already (B, C, H, W), treat as-is
            pass

        x = self.conv_stack(x)
        x = self.gap(x)  # (B, C, 1, 1)
        x = x.view(x.size(0), -1)  # (B, C)
        x = self.classifier(x)  # (B, num_classes)
        return x

    def export_to_onnx(self, path: str = "models/models/dpu_lite_net.onnx"):
        """Export the model to ONNX for FPGA DPU compilation."""
        dummy_input = torch.randn(1, self.input_channels, 80, 80)
        torch.onnx.export(
            self,
            dummy_input,
            path,
            input_names=["event_frame_stack"],
            output_names=["action_logits"],
            dynamic_axes={
                "event_frame_stack": {0: "batch"},
                "action_logits": {0: "batch"},
            },
            opset_version=13,
        )
        print(f"  Exported ONNX model to {path}")
        return path


# ---------------------------------------------------------------------------
# Direct Action Predictor
# ---------------------------------------------------------------------------
class DirectActionPredictor:
    """CNN-based direct action prediction for collision avoidance.

    Maps temporal stacks of event frames directly to evasion actions
    without intermediate flow estimation or clustering steps.

    Usage:
        predictor = DirectActionPredictor()
        predictor.load_weights("models/models/dpu_action.pth")

        # During flight:
        stack = aggregator.get_temporal_stack()  # (5, 80, 80)
        if stack is not None:
            prediction = predictor.predict(stack)
            drone.move(prediction.velocity)
    """

    def __init__(
        self,
        model: Optional[DPULightNet] = None,
        input_channels: int = 5,
        input_size: Tuple[int, int] = (80, 80),
        num_classes: int = 5,
        device: Optional[str] = None,
        confidence_threshold: float = 0.4,
        smoothing_alpha: float = 0.3,
    ):
        """
        Args:
            model: Pre-built DPULightNet or None to create one
            input_channels: Number of temporal stack frames (K)
            input_size: (H, W) of event frames
            num_classes: Number of evasion action classes
            device: "cpu", "cuda", or None (auto-detect)
            confidence_threshold: Min confidence to issue an action
            smoothing_alpha: EMA factor for action smoothing
        """
        self.input_channels = input_channels
        self.input_size = input_size
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.smoothing_alpha = smoothing_alpha

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if model is None:
            self.model = DPULightNet(
                input_channels=input_channels,
                input_height=input_size[0],
                input_width=input_size[1],
                num_classes=num_classes,
            )
        else:
            self.model = model

        self.model = self.model.to(self.device)
        self.model.eval()

        # Smoothing state
        self._smoothed_probs: Optional[np.ndarray] = None
        self._inference_count = 0

    def load_weights(self, path: str):
        """Load pretrained weights."""
        if not os.path.exists(path):
            print(f"  [DirectActionPredictor] Warning: weights not found at {path}")
            print(f"  Model will use random initialization.")
            return

        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        print(f"  Loaded weights from {path}")

    def predict(
        self,
        stack: "EventFrameTemporalStack",
    ) -> DirectActionPrediction:
        """Predict evasion action from a temporal stack of event frames.

        Args:
            stack: EventFrameTemporalStack with K frames

        Returns:
            DirectActionPrediction with action, probabilities, and velocity
        """
        import time

        t0 = time.perf_counter()

        # Extract frame data
        if hasattr(stack, "frames"):
            # EventFrameTemporalStack
            frame_data = np.stack([f.frame for f in stack.frames], axis=0)
        elif hasattr(stack, "stack_shape"):
            # Already has stack data
            frame_data = np.stack([f.frame for f in stack.frames], axis=0)
        else:
            # Raw numpy array
            frame_data = np.asarray(stack)

        # Ensure correct shape
        if frame_data.ndim == 3:
            frame_data = frame_data[np.newaxis, ...]  # (1, K, H, W)
        elif frame_data.ndim == 2:
            frame_data = frame_data[np.newaxis, np.newaxis, ...]  # (1, 1, H, W)

        # Convert to tensor
        x = torch.from_numpy(frame_data).float().to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self.model(x)  # (1, num_classes)
            probs = torch.softmax(logits, dim=-1)  # (1, num_classes)

        probs_np = probs.cpu().numpy().flatten()

        # Apply smoothing
        if self._smoothed_probs is not None:
            self._smoothed_probs = (
                self.smoothing_alpha * probs_np
                + (1 - self.smoothing_alpha) * self._smoothed_probs
            )
            probs_np = self._smoothed_probs
        else:
            self._smoothed_probs = probs_np

        # Determine action
        action_idx = int(np.argmax(probs_np))
        confidence = float(probs_np.max())

        # Only issue STAY if the STAY probability is above threshold
        # or if the max probability is below confidence threshold
        if action_idx == 0 and confidence < 0.5:
            # Check second-best action
            sorted_idx = np.argsort(probs_np)[::-1]
            if probs_np[sorted_idx[1]] > self.confidence_threshold:
                action_idx = sorted_idx[1]
                confidence = probs_np[action_idx]
        elif confidence < self.confidence_threshold:
            # Not confident enough → STAY
            action_idx = 0

        action = EvasionAction(action_idx)
        velocity = ACTION_VELOCITY[action_idx]

        # Timing
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        self._inference_count += 1

        return DirectActionPrediction(
            action=action,
            probabilities=probs_np,
            confidence=confidence,
            velocity=velocity,
            latency_ms=latency_ms,
            frame_count=getattr(stack, "stack_shape", (0, 0, 0))[0],
        )

    def get_action_name(self, action: EvasionAction) -> str:
        """Get human-readable action name."""
        return ACTION_NAMES.get(action.value, "UNKNOWN")

    def reset(self):
        """Reset smoothing state."""
        self._smoothed_probs = None
        self._inference_count = 0


# ---------------------------------------------------------------------------
# Training script entry (for user retraining on custom data)
# ---------------------------------------------------------------------------
def train_on_dataset(
    data_dir: str,
    output_path: str = "models/models/dpu_action.pth",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
):
    """Train the DPU-LiteNet on a labeled event frame dataset.

    Expected format:
        data_dir/
            train/
                STAY/    # Event frames with no threat
                LEFT/    # Frames where LEFT evasion is correct
                RIGHT/
                UP/
                DOWN/
            val/
                ... (same structure)

    Each file is .npy or .npz with shape (K, 80, 80).
    """
    # Placeholder — actual implementation depends on the dataset format
    # The paper uses 238 ball-throw recordings from their SwiftEagle dataset
    print("Training on dataset...")
    print(f"  Data: {data_dir}")
    print(f"  Output: {output_path}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}")
    print()
    print("  This function is a placeholder. Training code depends on")
    print("  the specific dataset format. See README for details.")
    print()
    print("  The Bonazzi et al. (2025) dataset is available at:")
    print("  https://github.com/pbonazzi/eva")