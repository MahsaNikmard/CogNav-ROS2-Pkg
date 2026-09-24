"""MeshOracle applies the polar oracle's restrictions to exact geometry.

Runs on a synthetic raster, without habitat-sim.
"""
import math

import numpy as np

from cognav_representation.navmesh import NavMesh, odom_to_habitat
from cognav_representation.oracle import MeshOracle, PolarOracle

from _harness import ANGLES, PADDING, REAL_BINS, Checker, scene

RADIUS = 0.18 + 0.18          # robot_radius + maneuver_clearance
RES = 0.05
HALF_M = 6.0                  # the synthetic room is 12 m square
MAX_RANGE = 5.0
FOV_EDGE = abs(ANGLES[PADDING])   # +-43.88 degrees, in radians


def room(free=True):
    """A square of navigable floor centred on the origin."""
    n = int(2 * HALF_M / RES)
    grid = np.full((n, n), free, dtype=bool)
    return NavMesh(grid, RES, origin=(-HALF_M, -HALF_M))


def wall_room(wall_x=2.0):
    """Navigable floor with everything beyond `wall_x` blocked."""
    nav = room()
    col = int((wall_x - (-HALF_M)) / RES)
    nav.free[:, col:] = False
    return nav


def oracle(nav, odom=(0.0, 0.0, 0.0), start=(0.0, 0.0, 0.0, 0.0)):
    return MeshOracle(nav, odom, start, ANGLES, REAL_BINS, max_range=MAX_RANGE)


def main():
    c = Checker()

    print("\n[1] points inside the footprint are free")
    blocked = oracle(room(free=False))
    c("a point inside the footprint is free even on a fully blocked map",
      blocked.point_is_free(0.5 * RADIUS, 0.0, RADIUS))
    c("a point outside it is not", not blocked.point_is_free(1.0, 0.0, RADIUS))

    print("\n[2] the field of view is refused on the same rule as polar")
    nav = room()
    o = oracle(nav)
    inside = FOV_EDGE * 0.5
    outside = FOV_EDGE + math.radians(5.0)
    r = 1.5
    c(f"a bearing inside +-{math.degrees(FOV_EDGE):.2f} deg is free",
      o.point_is_free(r * math.cos(inside), r * math.sin(inside), RADIUS))
    c("a bearing just outside it is refused",
      not o.point_is_free(r * math.cos(outside), r * math.sin(outside), RADIUS))
    c("directly behind is refused",
      not o.point_is_free(-r, 0.0, RADIUS))

    print("\n[3] the range limit matches perception's max_range")
    c(f"a point at {MAX_RANGE - 0.5:.1f} m ahead is free",
      o.point_is_free(MAX_RANGE - 0.5, 0.0, RADIUS))
    c(f"a point at {MAX_RANGE + 0.5:.1f} m ahead is refused",
      not o.point_is_free(MAX_RANGE + 0.5, 0.0, RADIUS))

    print("\n[4] geometry is actually read, not assumed")
    ow = oracle(wall_room(wall_x=2.0))
    c("floor short of the wall is free", ow.point_is_free(1.0, 0.0, RADIUS))
    c("the wall itself is refused", not ow.point_is_free(2.5, 0.0, RADIUS))
    c("a segment stopping short of the wall is free",
      ow.segment_is_free((0.0, 0.0), (1.4, 0.0), RADIUS))
    c("a segment crossing it is not",
      not ow.segment_is_free((0.0, 0.0), (3.0, 0.0), RADIUS))

    print("\n[5] the body is inflated, not just its centre")
    just_inside = 2.0 - RADIUS * 0.5
    c(f"a point {RADIUS * 0.5:.2f} m from the wall is refused for a "
      f"{RADIUS:.2f} m body",
      not ow.point_is_free(just_inside, 0.0, RADIUS))

    print("\n[6] the odom to habitat mapping follows the start pose")
    straight = odom_to_habitat((0.0, 0.0, 0.0, 0.0), 1.0, 0.0)
    turned = odom_to_habitat((0.0, 0.0, 0.0, math.pi / 2), 1.0, 0.0)
    c("a start yaw rotates the mapping", math.hypot(*[a - b for a, b in zip(straight, turned)]) > 1.0,
      f"{straight} vs {turned}")
    shifted = odom_to_habitat((3.0, 0.0, -2.0, 0.0), 1.0, 0.0)
    c("a start position translates it", shifted == (4.0, -2.0), f"{shifted}")

    print("\n[7] geometry is looked up through the start pose")
    off = MeshOracle(wall_room(wall_x=2.0), (0.0, 0.0, 0.0), (3.0, 0.0, -2.0, 0.0),
                     ANGLES, REAL_BINS, max_range=MAX_RANGE)
    c("ahead of a start placed past the wall is refused",
      not off.point_is_free(1.0, 0.0, RADIUS),
      "habitat x = 4.0 is beyond the wall at 2.0")

    print("\n[8] on open floor the two oracles agree")
    polar = PolarOracle(scene(4.5), ANGLES, REAL_BINS)
    mesh = oracle(room())
    agree = disagree = 0
    for k in range(400):
        theta = -math.pi + 2 * math.pi * k / 400
        for rr in (0.5, 1.5, 3.0, 4.0):
            px, py = rr * math.cos(theta), rr * math.sin(theta)
            if polar.point_is_free(px, py, RADIUS) == mesh.point_is_free(px, py, RADIUS):
                agree += 1
            else:
                disagree += 1
    c("open floor: the two oracles accept and refuse the same points",
      disagree == 0, f"{agree} agree, {disagree} disagree")

    c.finish("mesh oracle")


if __name__ == "__main__":
    main()
