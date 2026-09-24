"""ROS map_server occupancy maps: loading, reachable area and clearance.

The coverage denominator is the free space a body of the robot's radius can
reach from the start: free cells eroded by the radius, then flood-filled.
"""
from __future__ import annotations

import math
import os

import numpy as np
import yaml


class OccupancyMap:
    """A ROS map_server occupancy grid, in metres."""

    def __init__(self, yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as fh:
            meta = yaml.safe_load(fh)
        image = meta["image"]
        if not os.path.isabs(image):
            image = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), image)
        self.resolution = float(meta["resolution"])
        self.origin = tuple(float(v) for v in meta["origin"][:2])
        self.free_thresh = float(meta.get("free_thresh", 0.25))
        self.occupied_thresh = float(meta.get("occupied_thresh", 0.65))
        self.grid = _read_pgm(image)
        # map_server convention: a bright pixel is free.
        p = (255.0 - self.grid.astype(float)) / 255.0
        if int(meta.get("negate", 0)):
            p = 1.0 - p
        self.free = p < self.free_thresh
        self.occupied = p > self.occupied_thresh

    @property
    def shape(self):
        return self.grid.shape

    def world_to_cell(self, x: float, y: float):
        """(x, y) in metres to (row, col).

        Row 0 is the top of the image and the origin its bottom-left corner.
        """
        col = math.floor((x - self.origin[0]) / self.resolution)
        row = math.floor(self.grid.shape[0] - 1
                         - (y - self.origin[1]) / self.resolution + 0.5)
        return row, col

    def cell_to_world(self, row: int, col: int):
        """(row, col) to the metric centre of that cell."""
        x = (col + 0.5) * self.resolution + self.origin[0]
        y = (self.grid.shape[0] - 1 - row) * self.resolution + self.origin[1]
        return x, y

    def in_bounds(self, row: int, col: int) -> bool:
        h, w = self.grid.shape
        return 0 <= row < h and 0 <= col < w

    def free_area_m2(self) -> float:
        return float(self.free.sum()) * self.resolution ** 2

    def reachable_mask(self, start_xy, robot_radius: float):
        """Free cells a body of `robot_radius` can occupy, connected to `start_xy`.

        None when the start itself lies inside the inflation band.
        """
        r_cells = max(1, int(round(robot_radius / self.resolution)))
        eroded = _erode_disc(self.free, r_cells)
        start = self.world_to_cell(*start_xy)
        if not self.in_bounds(*start) or not eroded[start]:
            return None
        return _flood(eroded, start)


def clearance_field(occ: "OccupancyMap", robot_radius: float) -> np.ndarray:
    """Metres from each cell to the nearest non-free cell, minus the body radius.

    Measured from the body surface, as Habitat's `distance_to_closest_obstacle`
    is. Unknown cells count as blocked, and cells inside the inflation band
    come out negative.
    """
    try:
        from scipy.ndimage import distance_transform_edt

        dist = distance_transform_edt(occ.free, sampling=occ.resolution)
    except ImportError:
        dist = _edt_bruteforce(occ)
    return dist - robot_radius


def _edt_bruteforce(occ: "OccupancyMap") -> np.ndarray:
    """The same transform without scipy; quadratic, so for small maps only."""
    blocked = np.argwhere(~occ.free)
    h, w = occ.free.shape
    out = np.full((h, w), np.inf, dtype=float)
    if blocked.size == 0:
        return out
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    for br, bc in blocked:
        np.minimum(out, (rows - br) ** 2 + (cols - bc) ** 2, out=out)
    return np.sqrt(out) * occ.resolution


def _read_pgm(path) -> np.ndarray:
    with open(path, "rb") as fh:
        magic = fh.readline().strip()
        if magic not in (b"P5", b"P2"):
            raise ValueError(f"{path}: expected a binary or ascii PGM, got {magic!r}")
        dims = fh.readline()
        while dims.startswith(b"#"):
            dims = fh.readline()
        width, height = (int(v) for v in dims.split())
        maxval = int(fh.readline())
        if magic == b"P5":
            dtype = np.uint8 if maxval < 256 else ">u2"
            data = np.frombuffer(fh.read(), dtype=dtype, count=width * height)
        else:
            data = np.asarray(fh.read().split(), dtype=int)[:width * height]
    grid = data.reshape(height, width)
    if maxval != 255:
        grid = (grid.astype(float) * 255.0 / maxval)
    return grid.astype(np.uint8)


def _erode_disc(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Binary erosion by a disc, without a scipy dependency."""
    out = mask.copy()
    h, w = mask.shape
    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            if dr * dr + dc * dc > radius_cells * radius_cells:
                continue
            shifted = np.ones_like(mask)
            r0, r1 = max(0, dr), min(h, h + dr)
            c0, c1 = max(0, dc), min(w, w + dc)
            shifted[r0 - dr:r1 - dr, c0 - dc:c1 - dc] = mask[r0:r1, c0:c1]
            out &= shifted
    return out


def _flood(mask: np.ndarray, start) -> np.ndarray:
    """Four-connected flood fill, iterative so a large map cannot blow the stack."""
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    stack = [start]
    seen[start] = True
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return seen
