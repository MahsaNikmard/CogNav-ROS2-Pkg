"""RGB image to padded polar distance vector, through Depth Anything V2.

Per frame: metric depth from DA2, per-bin minimum range over the horizontal
field of view (encoding.compute_distance_vector), a per-bin temporal median,
and virtual padding bins on both sides. Runs in the `perception` conda
environment, where torch and the DA2 sources are installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from depth_anything_v2.dpt import DepthAnythingV2

from cognav_perception.encoding import (
    DEPTH_SCALE,
    FLOOR_MARGIN,
    FOV_DEG,
    FOV_PADDING_BINS,
    MAX_RANGE,
    NUM_BINS,
    PADDING_FILL_DEPTH_M,
    SAFETY_THRESHOLD,
    TEMPORAL_WINDOW,
    VERTICAL_MARGIN,
    TemporalAggregator,
    compute_distance_vector,
    pad_distance_vector,
)

_DA2_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


@dataclass
class PerceptionFrame:
    """One frame's outputs. `distance_vector` is what the node publishes."""

    distance_vector: np.ndarray         # (num_bins + 2*pad,) filtered and padded
    raw_distance_vector: np.ndarray     # (num_bins,) current frame only
    bin_angles: np.ndarray              # (num_bins + 2*pad,) rad, positive left
    danger: np.ndarray                  # (num_bins,) in [0, 1]
    depth_map: np.ndarray               # (H, W) metres, after depth_scale
    fov_deg: float                      # field of view including padding


class Perception:
    """Owns the DA2 model and the temporal filter. Call `update` once per frame."""

    def __init__(
        self,
        intrinsics_path: str | Path,
        weights_path: str | Path,
        *,
        encoder: str = "vits",
        max_depth: float = 20.0,
        num_bins: int = NUM_BINS,
        fov_deg: float = FOV_DEG,
        max_range: float = MAX_RANGE,
        vertical_margin: float = VERTICAL_MARGIN,
        floor_margin: float = FLOOR_MARGIN,
        depth_scale: float = DEPTH_SCALE,
        safety_threshold: float = SAFETY_THRESHOLD,
        temporal_window: int = TEMPORAL_WINDOW,
        fov_padding_bins: int = FOV_PADDING_BINS,
        fov_padding_depth: float = PADDING_FILL_DEPTH_M,
        device: Optional[str] = None,
    ) -> None:
        intrinsics_path = Path(intrinsics_path)
        weights_path = Path(weights_path)
        if not intrinsics_path.exists():
            raise FileNotFoundError(f"Camera intrinsics not found: {intrinsics_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"DA2 weights not found: {weights_path}")
        if encoder not in _DA2_CONFIGS:
            raise ValueError(f"Unknown DA2 encoder '{encoder}'; expected one of {list(_DA2_CONFIGS)}")

        self.K = np.load(intrinsics_path).astype(np.float64)
        # Optional distortion coefficients in distortion.npy beside the
        # intrinsics; all zeros or absent means a pinhole camera.
        dist_path = intrinsics_path.with_name("distortion.npy")
        if dist_path.exists():
            D = np.load(dist_path).astype(np.float64).reshape(-1)
            self.D: Optional[np.ndarray] = D if np.any(D) else None
        else:
            self.D = None
        self._undist_norm_xy: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._undist_shape: Optional[tuple[int, int]] = None
        self.num_bins = num_bins
        self.fov_deg = fov_deg
        self.max_range = max_range
        self.vertical_margin = vertical_margin
        self.floor_margin = floor_margin
        self.depth_scale = depth_scale
        self.fov_padding_bins = fov_padding_bins
        self.fov_padding_depth = float(fov_padding_depth)

        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = DepthAnythingV2(**{**_DA2_CONFIGS[encoder], "max_depth": max_depth})
        self.model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True))
        self.model = self.model.to(self.device).eval()

        self.aggregator = TemporalAggregator(
            num_bins=num_bins,
            window_size=temporal_window,
            danger_threshold=safety_threshold,
        )

        self._bin_angles = self._compute_bin_angles()
        self._virtual_fov_deg = (
            fov_deg * (num_bins + 2 * fov_padding_bins) / num_bins
        )

    def _compute_bin_angles(self) -> np.ndarray:
        """Centre bearing of every bin in the padded vector, bin 0 leftmost."""
        fov_rad = math.radians(self.fov_deg)
        bin_width = fov_rad / self.num_bins
        half = fov_rad / 2.0
        n_total = self.num_bins + 2 * self.fov_padding_bins
        idx = np.arange(n_total, dtype=np.float64)
        return half + bin_width * (self.fov_padding_bins - 0.5 - idx)

    def _ensure_undist_grid(self, H: int, W: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Normalised undistorted pixel coordinates for an (H, W) image, cached.

        None when there are no distortion coefficients. Back-projection then
        uses X = x_n * Z and Y = y_n * Z without resampling the depth map.
        """
        if self.D is None:
            return None
        if self._undist_norm_xy is not None and self._undist_shape == (H, W):
            return self._undist_norm_xy
        uu, vv = np.meshgrid(
            np.arange(W, dtype=np.float64),
            np.arange(H, dtype=np.float64),
        )
        pts = np.stack([uu.ravel(), vv.ravel()], axis=1).astype(np.float64)
        undist = cv2.undistortPoints(pts[:, None, :], self.K, self.D).reshape(-1, 2)
        x_n = undist[:, 0].reshape(H, W)
        y_n = undist[:, 1].reshape(H, W)
        self._undist_norm_xy = (x_n, y_n)
        self._undist_shape = (H, W)
        return self._undist_norm_xy

    @torch.no_grad()
    def _infer_depth(self, image_bgr: np.ndarray) -> np.ndarray:
        """Metric depth (H, W) in metres from a BGR image."""
        return self.model.infer_image(image_bgr)

    def update(self, image_bgr: np.ndarray) -> PerceptionFrame:
        """Run one frame end to end."""
        depth_map = self._infer_depth(image_bgr)
        H, W = depth_map.shape
        raw_dv = compute_distance_vector(
            depth_map, self.K,
            num_bins=self.num_bins,
            fov_deg=self.fov_deg,
            max_range=self.max_range,
            vertical_margin=self.vertical_margin,
            floor_margin=self.floor_margin,
            depth_scale=self.depth_scale,
            undist_norm_xy=self._ensure_undist_grid(H, W),
        )
        smoothed = self.aggregator.update(raw_dv)
        padded = pad_distance_vector(
            smoothed,
            padding_bins=self.fov_padding_bins,
            fill_value=self.fov_padding_depth,
        )
        return PerceptionFrame(
            distance_vector=padded,
            raw_distance_vector=raw_dv,
            bin_angles=self._bin_angles,
            danger=self.aggregator.danger,
            depth_map=depth_map * self.depth_scale,
            fov_deg=self._virtual_fov_deg,
        )

    def reset(self) -> None:
        """Drop the temporal history."""
        self.aggregator.reset()
