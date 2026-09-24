"""Depth map to polar distance vector. Needs only numpy.

Camera frame: x right, y down, z forward. The horizontal angle is atan2(-x, z),
so a positive angle is to the robot's left.
"""
import math
from collections import deque
from typing import Optional

import numpy as np

from cognav_perception.defaults import PERCEPTION_DEFAULTS as _D

NUM_BINS: int = _D.num_bins
FOV_DEG: float = _D.fov_deg
MAX_RANGE: float = _D.max_range
VERTICAL_MARGIN: float = _D.vertical_margin
FLOOR_MARGIN: float = _D.floor_margin
DEPTH_SCALE: float = _D.depth_scale
SAFETY_THRESHOLD: float = _D.safety_threshold
TEMPORAL_WINDOW: int = _D.temporal_window
FOV_PADDING_BINS: int = _D.fov_padding_bins
PADDING_FILL_DEPTH_M: float = _D.padding_fill_depth_m


def compute_distance_vector(
    depth_map: np.ndarray,
    K: np.ndarray,
    *,
    num_bins: int = NUM_BINS,
    fov_deg: float = FOV_DEG,
    max_range: float = MAX_RANGE,
    vertical_margin: float = VERTICAL_MARGIN,
    floor_margin: float = FLOOR_MARGIN,
    depth_scale: float = DEPTH_SCALE,
    undist_norm_xy: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """Back-project a depth map and keep the minimum range per angular bin.

    Returns shape (num_bins,), bin 0 leftmost, np.inf where nothing was seen.
    Points outside the height band [-vertical_margin, floor_margin] or beyond
    `max_range` are dropped. `undist_norm_xy`, if given, holds the
    distortion-corrected normalised pixel coordinates (x_n, y_n), each (H, W),
    and replaces the pinhole back-projection.
    """
    fov_rad = math.radians(fov_deg)
    half_fov = fov_rad / 2.0
    bin_width = fov_rad / num_bins

    H, W = depth_map.shape

    scaled_depth = depth_map.astype(np.float64) * depth_scale
    Z = scaled_depth.ravel()

    if undist_norm_xy is not None:
        x_n, y_n = undist_norm_xy
        if x_n.shape != (H, W) or y_n.shape != (H, W):
            raise ValueError(
                f"undist_norm_xy shapes {x_n.shape}, {y_n.shape} != depth shape {(H, W)}"
            )
        X = x_n.ravel() * Z
        Y = y_n.ravel() * Z
    else:
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        u = np.arange(W, dtype=np.float64)
        v = np.arange(H, dtype=np.float64)
        uu, vv = np.meshgrid(u, v)
        X = ((uu.ravel() - cx) * Z) / fx
        Y = ((vv.ravel() - cy) * Z) / fy

    mask = (Z > 0) & (Z <= max_range) & (Y >= -vertical_margin) & (Y <= floor_margin)
    X_f = X[mask]
    Z_f = Z[mask]

    distance_vector = np.full(num_bins, np.inf, dtype=np.float64)
    if len(X_f) == 0:
        return distance_vector

    angles = np.arctan2(-X_f, Z_f)
    ranges = np.sqrt(X_f ** 2 + Z_f ** 2)

    # floor rather than truncation, so points just past the left edge are
    # discarded instead of landing in bin 0.
    bin_indices = np.floor((half_fov - angles) / bin_width).astype(int)
    valid = (bin_indices >= 0) & (bin_indices < num_bins)
    np.minimum.at(distance_vector, bin_indices[valid], ranges[valid])
    return distance_vector


def pad_distance_vector(
    distance_vector: np.ndarray,
    padding_bins: int = FOV_PADDING_BINS,
    fill_value: float = PADDING_FILL_DEPTH_M,
) -> np.ndarray:
    """Add `padding_bins` virtual bins at depth `fill_value` on each side.

    The padding marks the edge of the field of view so that downstream code
    never proposes a bearing the camera did not observe.
    """
    if padding_bins <= 0:
        return distance_vector
    pad = np.full(padding_bins, fill_value, dtype=distance_vector.dtype)
    return np.concatenate([pad, distance_vector, pad])


class TemporalAggregator:
    """Sliding-window per-bin median over the frames that observed each bin.

    Unobserved frames (inf) are excluded, so an obstacle seen in a minority of
    frames survives while single-frame depth spikes are rejected.
    """

    def __init__(
        self,
        num_bins: int = NUM_BINS,
        window_size: int = TEMPORAL_WINDOW,
        danger_threshold: float = SAFETY_THRESHOLD,
    ) -> None:
        self.num_bins = num_bins
        self.window_size = window_size
        self.danger_threshold = danger_threshold
        self._history: deque[np.ndarray] = deque(maxlen=window_size)

    def update(self, distance_vector: np.ndarray) -> np.ndarray:
        self._history.append(distance_vector.copy())
        return self.get_filtered()

    def get_filtered(self) -> np.ndarray:
        if not self._history:
            return np.full(self.num_bins, np.inf, dtype=np.float64)
        stacked = np.stack(list(self._history), axis=0)
        finite = np.isfinite(stacked)
        observed = finite.any(axis=0)
        out = np.full(self.num_bins, np.inf, dtype=np.float64)
        if np.any(observed):
            masked = np.where(finite, stacked, np.nan)
            out[observed] = np.nanmedian(masked[:, observed], axis=0)
        return out

    @property
    def danger(self) -> np.ndarray:
        """Per-bin fraction of recent frames nearer than `danger_threshold`."""
        if not self._history:
            return np.zeros(self.num_bins, dtype=np.float64)
        stacked = np.stack(list(self._history), axis=0)
        return (stacked < self.danger_threshold).mean(axis=0)

    def reset(self) -> None:
        self._history.clear()
