from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple

import numpy as np


@dataclass
class SpeedEstimate:
    speed_kmh: float
    speed_valid: bool
    state: str


class SpeedEstimator:
    """Simple, robust speed estimator from track centroid motion."""

    def __init__(
        self,
        fps: float = 30.0,
        meters_per_pixel: float = 0.05,
        window_size: int = 8,
        stationary_px: float = 6.0,
        min_speed_kmh: float = 2.0,
        smoothing_alpha: float = 0.3,
    ) -> None:
        self._fps = max(1e-6, float(fps))
        self._meters_per_pixel = float(meters_per_pixel)
        self._window_size = max(3, int(window_size))
        self._stationary_px = float(stationary_px)
        self._min_speed_kmh = float(min_speed_kmh)
        self._alpha = float(smoothing_alpha)
        self._history: Dict[int, Deque[Tuple[int, float, float]]] = defaultdict(
            lambda: deque(maxlen=self._window_size)
        )
        self._smooth: Dict[int, float] = {}

    def set_fps(self, fps: float) -> None:
        if fps and fps > 1e-6:
            self._fps = float(fps)

    def update(self, track_id: int, frame_idx: int, cx: float, cy: float) -> SpeedEstimate:
        if track_id < 0:
            return SpeedEstimate(0.0, False, "NEW")

        hist = self._history[track_id]
        hist.append((int(frame_idx), float(cx), float(cy)))
        if len(hist) < 2:
            return SpeedEstimate(0.0, False, "NEW")

        f0, x0, y0 = hist[0]
        f1, x1, y1 = hist[-1]
        dt_frames = max(1, f1 - f0)
        disp_px = float(np.hypot(x1 - x0, y1 - y0))

        if disp_px < self._stationary_px:
            prev = self._smooth.get(track_id, 0.0) * 0.6
            self._smooth[track_id] = prev
            return SpeedEstimate(prev, prev >= self._min_speed_kmh, "STATIONARY")

        px_per_sec = disp_px * (self._fps / dt_frames)
        kmh = px_per_sec * self._meters_per_pixel * 3.6
        prev = self._smooth.get(track_id, kmh)
        smoothed = (self._alpha * kmh) + ((1.0 - self._alpha) * prev)
        self._smooth[track_id] = smoothed
        if smoothed < self._min_speed_kmh:
            return SpeedEstimate(0.0, False, "STATIONARY")
        return SpeedEstimate(smoothed, True, "MOVING")

