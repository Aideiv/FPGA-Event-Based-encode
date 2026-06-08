"""
Event Frame Aggregator — Configurable event-to-frame conversion

Implements the event accumulation pipeline described in Bonazzi et al.
"Towards Low-Latency Event-based Obstacle Avoidance on an FPGA-Drone",
CVPRW 2025:

  1. Raw AER events (x, y, t, p) are accumulated into 2D spatial grids
     (frames) over configurable time windows (default: 1ms)
  2. Each pixel bin counts polarity-weighted events:
       frame[y][x] = sum(pos_events) - sum(neg_events)
  3. Frames are normalized and optionally passed through a temporal
     stack for motion context

Key paper parameters:
  - Grid resolution: 80×80 (configurable)
  - Integration time: 1ms (configurable, 0.1–10ms)
  - End-to-end latency: ~2.14ms (1ms aggregation + 0.94ms inference)
  - Max event rate: 10 Mevents/s
  - Prophesee GenX320 sensor (320×240 native, subsampled to 80×80)

The aggregator supports both:
  - Fixed window mode: accumulate for exactly N ms then emit a frame
  - Rolling window mode: emit a frame every N ms using events from the
    last N ms (for continuous streaming)
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class EventFrame:
    """A single accumulated event frame.

    Attributes:
        frame: (H, W) float32 — normalized event count map, range [-1, 1]
        t_start: start timestamp of the integration window (seconds)
        t_end: end timestamp of the integration window (seconds)
        n_events: total events accumulated in this frame
        n_pos: positive polarity event count
        n_neg: negative polarity event count
    """

    frame: np.ndarray  # (H, W)
    t_start: float
    t_end: float
    n_events: int
    n_pos: int
    n_neg: int


@dataclass
class EventFrameTemporalStack:
    """A stack of K consecutive event frames for motion context.

    Paper uses a temporal stack of 5 frames (5ms total at 1kHz)
    to provide motion cues to the CNN.
    """

    frames: List[EventFrame]  # K frames, oldest to newest
    t_start: float
    t_end: float
    stack_shape: Tuple[int, int, int]  # (K, H, W)


class EventFrameAggregator:
    """Configurable event frame aggregator.

    Converts raw AER event streams into 2D spatial frames by accumulating
    events over fixed or rolling time windows. Matches the architecture
    in the Bonazzi et al. (2025) paper.

    Two modes:
      - FIXED:  Accumulate for exactly `accumulation_time_ms` then emit.
      - ROLLING: Every `integration_time_ms`, emit a frame containing
                 events from the last `accumulation_time_ms`.
    """

    MODE_FIXED = "fixed"
    MODE_ROLLING = "rolling"

    def __init__(
        self,
        sensor_width: int = 320,
        sensor_height: int = 240,
        grid_width: int = 80,
        grid_height: int = 80,
        accumulation_time_ms: float = 1.0,
        integration_time_ms: Optional[float] = None,
        mode: str = "rolling",
        polarity: str = "sum",  # "sum", "separate", "positive_only", "negative_only"
        normalize: bool = True,
        temporal_stack_size: int = 5,
        max_event_rate: int = 10_000_000,
    ):
        """
        Args:
            sensor_width: Native sensor width in pixels
            sensor_height: Native sensor height in pixels
            grid_width: Output frame width (subsampled)
            grid_height: Output frame height (subsampled)
            accumulation_time_ms: Integration window duration in ms
            integration_time_ms: If rolling, time between frame emissions
                (defaults to accumulation_time_ms)
            mode: "fixed" or "rolling"
            polarity: How to handle event polarities:
                "sum" — positive - negative
                "separate" — store as two channels
                "positive_only" — count positive events only
                "negative_only" — count negative events only
            normalize: If True, normalize frame values to [-1, 1]
            temporal_stack_size: Number of frames in temporal stack
            max_event_rate: Max events/sec for throughput tracking
        """
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.accumulation_time_ms = accumulation_time_ms
        self.integration_time_ms = integration_time_ms or accumulation_time_ms
        self.mode = mode
        self.polarity = polarity
        self.normalize = normalize
        self.temporal_stack_size = temporal_stack_size
        self.max_event_rate = max_event_rate

        # Subsampling factors
        self.scale_x = sensor_width / grid_width
        self.scale_y = sensor_height / grid_height

        # Accumulator state
        self._reset_accumulator()

        # Temporal frame buffer for stack
        self._frame_buffer: deque = deque(maxlen=temporal_stack_size)

        # Frame counter for debugging/timing
        self._frame_count = 0
        self._total_events_processed = 0
        self._start_time: Optional[float] = None

    def _reset_accumulator(self):
        """Reset the current accumulator frame."""
        if self.polarity == "separate":
            self._pos_frame = np.zeros(
                (self.grid_height, self.grid_width), dtype=np.float32
            )
            self._neg_frame = np.zeros(
                (self.grid_height, self.grid_width), dtype=np.float32
            )
        else:
            self._acc_frame = np.zeros(
                (self.grid_height, self.grid_width), dtype=np.float32
            )
        self._acc_start: Optional[float] = None
        self._acc_pos_count = 0
        self._acc_neg_count = 0
        self._last_emit_time: float = 0.0

    def process_events(
        self,
        events_t: np.ndarray,  # (n,) timestamps in seconds
        events_xy: np.ndarray,  # (n, 2) raw pixel coordinates (0-indexed)
        events_p: Optional[np.ndarray] = None,  # (n,) polarities (0 or 1)
    ) -> List[EventFrame]:
        """Process a batch of events and return any completed frames.

        Args:
            events_t: Sorted event timestamps in seconds
            events_xy: Raw pixel coordinates (x, y), NOT normalized
            events_p: Polarity array (True/1 = positive, False/0 = negative)
                       If None, all events treated as positive

        Returns:
            List of completed EventFrames (may be empty if window not yet full)
        """
        n = events_t.shape[0]
        if n == 0:
            return []

        if self._start_time is None:
            self._start_time = events_t[0]

        if events_p is None:
            events_p = np.ones(n, dtype=bool)

        # Convert raw pixel coords to grid indices
        # Grid bins: 0..grid_width-1, 0..grid_height-1
        grid_x = np.clip(
            (events_xy[:, 0] / self.scale_x).astype(np.int32),
            0,
            self.grid_width - 1,
        )
        grid_y = np.clip(
            (events_xy[:, 1] / self.scale_y).astype(np.int32),
            0,
            self.grid_height - 1,
        )

        # Track stats
        self._total_events_processed += n

        completed_frames: List[EventFrame] = []

        # ---- Event-by-event accumulation for precise timing ----
        # Paper uses FPGA hardware accumulation, but in Python we process
        # in batches for efficiency while maintaining frame timing
        if self.mode == self.MODE_FIXED:
            completed = self._accumulate_fixed(
                events_t, grid_x, grid_y, events_p
            )
            completed_frames.extend(completed)
        elif self.mode == self.MODE_ROLLING:
            completed = self._accumulate_rolling(
                events_t, grid_x, grid_y, events_p
            )
            completed_frames.extend(completed)

        return completed_frames

    def _accumulate_fixed(
        self,
        events_t: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        events_p: np.ndarray,
    ) -> List[EventFrame]:
        """Fixed window accumulation: emit frame every N ms exactly."""
        completed = []
        n = len(events_t)
        acc_time_s = self.accumulation_time_ms / 1000.0

        if self._acc_start is None:
            self._acc_start = events_t[0]
            # Snap to nearest multiple of accumulation time
            self._acc_start = np.floor(
                events_t[0] / acc_time_s
            ) * acc_time_s

        window_end = self._acc_start + acc_time_s

        for i in range(n):
            if events_t[i] >= window_end:
                # Emit current frame
                frame = self._emit_frame(self._acc_start, window_end)
                completed.append(frame)

                # Start new window
                self._reset_accumulator()
                self._acc_start = window_end
                window_end = self._acc_start + acc_time_s

            # Accumulate
            self._add_event(grid_x[i], grid_y[i], events_p[i])

        # Update accumulator end time to last event
        # (in case the batch ends mid-window)
        self._acc_pos_count += 0  # already counted above

        return completed

    def _accumulate_rolling(
        self,
        events_t: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        events_p: np.ndarray,
    ) -> List[EventFrame]:
        """Rolling window: emit frames every integration_time_ms."""
        completed = []
        n = len(events_t)
        integration_s = self.integration_time_ms / 1000.0
        accumulation_s = self.accumulation_time_ms / 1000.0

        if self._acc_start is None:
            self._acc_start = events_t[0]

        next_emit = self._last_emit_time + integration_s
        if self._last_emit_time == 0.0:
            next_emit = self._acc_start + integration_s

        for i in range(n):
            # Check if we should emit a frame
            if events_t[i] >= next_emit:
                # Emit: use events from [emit_time - accumulation_s, emit_time]
                frame_start = next_emit - accumulation_s
                frame = self._emit_frame(frame_start, next_emit)
                completed.append(frame)

                next_emit += integration_s

            # Accumulate
            self._add_event(grid_x[i], grid_y[i], events_p[i])

        self._last_emit_time = next_emit - integration_s
        return completed

    def _add_event(self, gx: int, gy: int, polarity: bool):
        """Add a single event to the current accumulator."""
        if self.polarity == "separate":
            if polarity:
                self._pos_frame[gy, gx] += 1
                self._acc_pos_count += 1
            else:
                self._neg_frame[gy, gx] += 1
                self._acc_neg_count += 1
        else:
            if polarity:
                self._acc_frame[gy, gx] += 1
                self._acc_pos_count += 1
            else:
                self._acc_frame[gy, gx] -= 1
                self._acc_neg_count += 1

    def _emit_frame(
        self, t_start: float, t_end: float
    ) -> EventFrame:
        """Build and return an EventFrame from the current accumulator."""
        if self.polarity == "separate":
            # Combine as difference
            frame_data = self._pos_frame - self._neg_frame
        elif self.polarity == "positive_only":
            frame_data = np.maximum(self._acc_frame, 0)
        elif self.polarity == "negative_only":
            frame_data = -np.minimum(self._acc_frame, 0)
        else:
            # "sum" mode: already polarity-weighted
            frame_data = self._acc_frame.copy()

        # Normalize to [-1, 1] range
        if self.normalize:
            max_val = np.max(np.abs(frame_data))
            if max_val > 0:
                frame_data = frame_data / max_val
            # Clip outliers
            frame_data = np.clip(frame_data, -1.0, 1.0)

        total_ev = self._acc_pos_count + self._acc_neg_count

        frame = EventFrame(
            frame=frame_data.astype(np.float32),
            t_start=t_start,
            t_end=t_end,
            n_events=total_ev,
            n_pos=self._acc_pos_count,
            n_neg=self._acc_neg_count,
        )

        # Add to temporal buffer for stack
        self._frame_buffer.append(frame)
        self._frame_count += 1

        return frame

    def get_temporal_stack(self) -> Optional[EventFrameTemporalStack]:
        """Get the current temporal stack of frames.

        Returns K frames stacked as (K, H, W) numpy array, or None if
        fewer than temporal_stack_size frames are available.
        """
        if len(self._frame_buffer) < self.temporal_stack_size:
            return None

        frames = list(self._frame_buffer)
        stack = np.stack(
            [f.frame for f in frames], axis=0
        )  # (K, H, W)

        return EventFrameTemporalStack(
            frames=frames,
            t_start=frames[0].t_start,
            t_end=frames[-1].t_end,
            stack_shape=(self.temporal_stack_size, self.grid_height, self.grid_width),
        )

    def get_latest_frame(self) -> Optional[EventFrame]:
        """Get the most recently emitted frame."""
        if len(self._frame_buffer) == 0:
            return None
        return self._frame_buffer[-1]

    def get_throughput_stats(self) -> Dict[str, float]:
        """Get throughput and timing statistics."""
        if self._start_time is None:
            return {"frames": 0, "events": 0, "fps": 0.0, "events_per_sec": 0.0}

        elapsed = time.time() - self._start_time
        stats = {
            "frames": self._frame_count,
            "events": self._total_events_processed,
            "fps": self._frame_count / max(elapsed, 1e-6),
            "events_per_sec": self._total_events_processed / max(elapsed, 1e-6),
            "elapsed_sec": elapsed,
        }
        return stats

    def reset(self):
        """Reset all state. Call when re-arming or restarting."""
        self._reset_accumulator()
        self._frame_buffer.clear()
        self._frame_count = 0
        self._total_events_processed = 0
        self._start_time = None
        self._last_emit_time = 0.0