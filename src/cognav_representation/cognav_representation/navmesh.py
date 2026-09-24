"""A scene's navigable surface, rasterised once and read without habitat-sim.

Produced by scripts/export_navmesh.py. Habitat is y-up, so the raster covers
the x-z floor plane and is indexed `free[row, col]`, with col along +x and row
along +z:

    x = origin_x + (col + 0.5) * resolution
    z = origin_z + (row + 0.5) * resolution

Odometry is zeroed at the episode start pose `start_hab` = (x0, y0, z0, yaw0),
so an odom point (X, Y) maps to

    x = x0 + X*cos(yaw0) - Y*sin(yaw0)
    z = z0 + X*sin(yaw0) + Y*cos(yaw0)
"""
from __future__ import annotations

import math

import numpy as np


class NavMesh:
    """A rasterised navigable surface in habitat's floor plane."""

    def __init__(self, free: np.ndarray, resolution: float,
                 origin: tuple[float, float], height: float = 0.0,
                 meta: dict | None = None):
        self.free = np.asarray(free, dtype=bool)
        self.res = float(resolution)
        self.origin = (float(origin[0]), float(origin[1]))
        self.height = float(height)
        self.meta = dict(meta or {})
        self._passable: np.ndarray | None = None
        self._passable_radius: float | None = None

    @classmethod
    def load(cls, path) -> "NavMesh":
        data = np.load(str(path), allow_pickle=False)
        return cls(
            free=data["free"],
            resolution=float(data["resolution"]),
            origin=(float(data["origin"][0]), float(data["origin"][1])),
            height=float(data["height"]) if "height" in data else 0.0,
            meta={k: data[k].tolist() for k in data.files
                  if k not in ("free", "resolution", "origin", "height")},
        )

    def save(self, path) -> None:
        np.savez_compressed(
            str(path), free=self.free, resolution=self.res,
            origin=np.asarray(self.origin, dtype=np.float64),
            height=self.height,
            **{k: np.asarray(v) for k, v in self.meta.items()})

    def to_cell(self, x: float, z: float) -> tuple[int, int]:
        """Habitat floor-plane point to (row, col)."""
        return (int(math.floor((z - self.origin[1]) / self.res)),
                int(math.floor((x - self.origin[0]) / self.res)))

    def inside(self, row: int, col: int) -> bool:
        return 0 <= row < self.free.shape[0] and 0 <= col < self.free.shape[1]

    def passable(self, radius: float) -> np.ndarray:
        """Navigable cells at least `radius` from any non-navigable cell.

        Cells off the raster count as blocked. Cached per radius.
        """
        if self._passable is not None and self._passable_radius == radius:
            return self._passable
        blocked = ~self.free
        pad = max(1, int(round(radius / self.res)))
        grown = blocked.copy()
        for dx in range(-pad, pad + 1):
            for dy in range(-pad, pad + 1):
                if dx * dx + dy * dy > pad * pad:
                    continue
                grown |= _shift(blocked, dx, dy)
        self._passable = self.free & ~grown
        self._passable_radius = radius
        return self._passable

    @property
    def area_m2(self) -> float:
        return float(self.free.sum()) * self.res * self.res


def _shift(a: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """Shift a boolean array, filling the exposed edge with True (blocked)."""
    out = np.ones_like(a)
    rows = slice(max(dr, 0), a.shape[0] + min(dr, 0))
    cols = slice(max(dc, 0), a.shape[1] + min(dc, 0))
    src_rows = slice(max(-dr, 0), a.shape[0] + min(-dr, 0))
    src_cols = slice(max(-dc, 0), a.shape[1] + min(-dc, 0))
    out[rows, cols] = a[src_rows, src_cols]
    return out


def odom_to_habitat(start_hab, X: float, Y: float) -> tuple[float, float]:
    """Odom-frame (X, Y) into habitat's floor plane, given the episode start."""
    x0, _y0, z0, yaw0 = (float(v) for v in start_hab[:4])
    c, s = math.cos(yaw0), math.sin(yaw0)
    return (x0 + X * c - Y * s, z0 + X * s + Y * c)
