"""Free-space oracles: the planner's `point_is_free` and `segment_is_free`.

Both oracles work in the robot frame and apply the same restrictions, so the
arms differ only in the geometry the tests are answered from. The safety gate
does not use an oracle; it always checks the current polar vector.
"""
from __future__ import annotations

import math

from cognav_representation.geometry import (
    point_is_free,
    prepare_obstacles,
    segment_is_free,
    to_odom,
)
from cognav_representation.navmesh import odom_to_habitat


class PolarOracle:
    """Free space as the current polar vector reports it; out of view is refused."""

    name = "polar"

    def __init__(self, ranges, angles, bin_range=None):
        self.ranges = ranges
        self.angles = angles
        self.bin_range = bin_range
        self._prepared = prepare_obstacles(ranges, angles, bin_range)

    def point_is_free(self, px: float, py: float, radius: float) -> bool:
        return point_is_free(self.ranges, self.angles, px, py, radius,
                             self.bin_range, self._prepared)

    def segment_is_free(self, a, b, radius: float) -> bool:
        return segment_is_free(self.ranges, self.angles, a, b, radius,
                               self.bin_range, prepared=self._prepared)


class MeshOracle:
    """Free space as the scene's navmesh reports it, inside the same view.

    The upper-reference arm. It applies the polar arm's restrictions exactly:
    points inside the body radius are free, bearings outside the real bins'
    span and points beyond `max_range` are refused, the body is inflated by
    the same radius, and segments are sampled the same number of times. Only
    the geometry is exact.

    The raster is in Habitat's floor plane and is reached through the episode
    start pose, so the mapping is exact only with exact odometry.
    """

    name = "mesh"

    def __init__(self, navmesh, odom, start_hab, angles, bin_range=None,
                 max_range: float = 5.0, cell_step: float | None = None):
        self.nav = navmesh
        self.odom = odom
        self.start_hab = start_hab
        self.max_range = float(max_range)
        self.cell_step = float(cell_step if cell_step is not None else navmesh.res)
        lo, hi = bin_range if bin_range else (0, len(angles))
        self.a_lo = float(min(angles[lo], angles[hi - 1]))
        self.a_hi = float(max(angles[lo], angles[hi - 1]))

    def _cell_ok(self, px: float, py: float, ok) -> bool:
        x, z = odom_to_habitat(self.start_hab, *to_odom(self.odom, px, py))
        row, col = self.nav.to_cell(x, z)
        return bool(self.nav.inside(row, col) and ok[row, col])

    def _in_view(self, px: float, py: float) -> bool:
        if math.hypot(px, py) > self.max_range:
            return False
        theta = math.atan2(py, px)
        return self.a_lo <= theta <= self.a_hi

    def point_is_free(self, px: float, py: float, radius: float) -> bool:
        if math.hypot(px, py) <= radius:
            return True
        if not self._in_view(px, py):
            return False
        return self._cell_ok(px, py, self.nav.passable(radius))

    def segment_is_free(self, a, b, radius: float, samples: int = 6) -> bool:
        ok = self.nav.passable(radius)
        for k in range(samples + 1):
            t = k / samples
            px = a[0] + t * (b[0] - a[0])
            py = a[1] + t * (b[1] - a[1])
            # Every segment starts at the robot, inside its own footprint.
            if math.hypot(px, py) <= radius:
                continue
            if not self._in_view(px, py):
                return False
            if not self._cell_ok(px, py, ok):
                return False
        return True
