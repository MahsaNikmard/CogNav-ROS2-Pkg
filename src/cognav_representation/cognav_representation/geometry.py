"""Geometry over the polar distance vector, in the robot frame.

Bearings are positive to the left. A non-finite range means nothing was
observed on that bearing and is treated as unobstructed.
"""
from __future__ import annotations

import math

import numpy as np


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def bearing_to_point(robot_xy, robot_yaw: float, point_xy) -> float:
    dx = point_xy[0] - robot_xy[0]
    dy = point_xy[1] - robot_xy[1]
    return wrap_angle(math.atan2(dy, dx) - robot_yaw)


def corridor_check(distances: np.ndarray, bin_angles: np.ndarray, ref_bearing: float, radius: float, warn_dist: float):
    """Bins nearer than `warn_dist` within `radius` of the ray at `ref_bearing`.

    Returns [(bin_index, distance)] nearest first.
    """
    blocked = []
    for i, (distance, theta) in enumerate(zip(distances, bin_angles)):
        if not np.isfinite(distance) or distance >= warn_dist:
            continue
        perpendicular_offset = distance * math.sin(theta - ref_bearing)
        if abs(perpendicular_offset) <= radius:
            blocked.append((i, float(distance)))
    blocked.sort(key=lambda item: item[1])
    return blocked


def safe_bin_mask(ranges, angles, min_range: float, half_width: float):
    """Per bin, whether a body of `half_width` can travel along that bearing.

    Bearing i is blocked if some bin j nearer than `min_range` has
    |d_j * sin(theta_j - theta_i)| <= half_width.
    """
    n = len(ranges)
    return [not corridor_check(ranges, angles, angles[i], half_width, min_range)
            for i in range(n)]


def find_gaps(ranges, angles, min_range: float, half_width: float, bin_range=None):
    """Contiguous runs of drivable bearings.

    Returns [(start_index, end_index, depth)], end inclusive, with depth the
    deepest finite reading in the run. Every bearing in a run already fits the
    body, so runs are not filtered by width.
    """
    lo, hi = bin_range if bin_range else (0, len(ranges))
    safe = safe_bin_mask(ranges, angles, min_range, half_width)

    gaps, start = [], None
    for i in range(lo, hi):
        if safe[i] and start is None:
            start = i
        elif not safe[i] and start is not None:
            gaps.append((start, i - 1))
            start = None
    if start is not None:
        gaps.append((start, hi - 1))

    out = []
    for a, b in gaps:
        window = [ranges[k] for k in range(a, b + 1) if math.isfinite(ranges[k])]
        out.append((a, b, max(window) if window else float("inf")))
    return out


def steer_through_gaps(gaps, angles, waypoint_bearing: float):
    """Bearing to steer and the gap chosen, or (None, None) without gaps.

    A waypoint inside a gap is steered at directly. Otherwise the robot aims at
    the nearest edge of the gap that needs the smallest deviation.
    """
    if not gaps:
        return None, None
    for a, b, depth in gaps:
        lo, hi = sorted((angles[a], angles[b]))
        if lo <= waypoint_bearing <= hi:
            return waypoint_bearing, (a, b, depth)

    best, best_bearing, best_cost = None, None, float("inf")
    for a, b, depth in gaps:
        lo, hi = sorted((angles[a], angles[b]))
        aim = max(lo, min(hi, waypoint_bearing))
        cost = abs(wrap_angle(aim - waypoint_bearing))
        if cost < best_cost:
            best, best_bearing, best_cost = (a, b, depth), aim, cost
    return best_bearing, best


def swept_arc_check(ranges, angles, v: float, w: float, half_width: float,
                    horizon_sec: float, bin_range=None, samples: int = 12):
    """Bins the body would sweep if it executed (v, w) for `horizon_sec`.

    The unicycle trajectory is sampled and a disc of `half_width` tested at each
    sample. Turning in place (v = 0) sweeps nothing. Returns
    [(bin_index, distance)] nearest first.
    """
    lo, hi = bin_range if bin_range else (0, len(ranges))
    points = []
    for i in range(lo, hi):
        d = ranges[i]
        if not math.isfinite(d):
            continue
        points.append((i, d, d * math.cos(angles[i]), d * math.sin(angles[i])))
    if not points or abs(v) < 1e-6:
        return []

    poses, x, y, theta = [], 0.0, 0.0, 0.0
    dt = horizon_sec / samples
    for _ in range(samples):
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        theta += w * dt
        poses.append((x, y))

    hit = {}
    for i, d, px, py in points:
        for cx, cy in poses:
            if math.hypot(px - cx, py - cy) <= half_width:
                hit[i] = float(d)
                break
    return sorted(hit.items(), key=lambda kv: kv[1])


def drivable_bins(ranges, angles, min_range: float, half_width: float, bin_range=None,
                  excluded=frozenset()):
    """Indices of real bins the body can travel along, minus `excluded`.

    The planner samples from these and the driver uses the same test through
    `find_gaps`, so the planner never commits where the driver cannot go.
    """
    lo, hi = bin_range if bin_range else (0, len(ranges))
    safe = safe_bin_mask(ranges, angles, min_range, half_width)
    return [i for i in range(lo, hi) if safe[i] and i not in excluded]


def bin_for_bearing(angles, bearing: float) -> int:
    """Index of the bin whose bearing is closest to `bearing`."""
    best, best_err = 0, float("inf")
    for i, a in enumerate(angles):
        err = abs(a - bearing)
        if err < best_err:
            best, best_err = i, err
    return best


def prepare_obstacles(ranges, angles, bin_range=None):
    """Field-of-view bounds and the finite readings, for repeated `point_is_free` calls."""
    lo, hi = bin_range if bin_range else (0, len(ranges))
    a = [angles[i] for i in range(lo, hi) if math.isfinite(ranges[i])]
    d = [ranges[i] for i in range(lo, hi) if math.isfinite(ranges[i])]
    a = np.asarray(a, dtype=float)
    d = np.asarray(d, dtype=float)
    return (float(min(angles[lo], angles[hi - 1])),
            float(max(angles[lo], angles[hi - 1])),
            a, d, d * np.cos(a), d * np.sin(a))


def point_is_free(ranges, angles, px: float, py: float, radius: float,
                  bin_range=None, prepared=None) -> bool:
    """Whether a body of `radius` may occupy the robot-frame point (px, py).

    Points inside the robot's own footprint are free. Points outside the field
    of view are not. Otherwise the point must lie at least `radius` short of
    the reading on its own bearing and at least `radius` from every observed
    obstacle point. Pass `prepared` from `prepare_obstacles` inside loops.
    """
    r = math.hypot(px, py)
    if r <= radius:
        return True
    theta = math.atan2(py, px)
    if prepared is None:
        prepared = prepare_obstacles(ranges, angles, bin_range)
    a_lo, a_hi, angs, dists, xs, ys = prepared
    if not (a_lo <= theta <= a_hi):
        return False
    if angs.size == 0:
        return True
    j = int(np.argmin(np.abs(angs - theta)))
    if r + radius > dists[j]:
        return False
    dx = xs - px
    dy = ys - py
    return bool(np.min(dx * dx + dy * dy) > radius * radius)


def segment_is_free(ranges, angles, p, q, radius: float, bin_range=None,
                    samples: int = 6, prepared=None) -> bool:
    """Is every point along p->q free? Endpoints included."""
    if prepared is None:
        prepared = prepare_obstacles(ranges, angles, bin_range)
    for k in range(samples + 1):
        t = k / samples
        if not point_is_free(ranges, angles,
                             p[0] + t * (q[0] - p[0]),
                             p[1] + t * (q[1] - p[1]),
                             radius, bin_range, prepared):
            return False
    return True


def to_odom(odom, px: float, py: float):
    """Robot-frame (px, py) into the odom frame."""
    x, y, yaw = odom
    c, s = math.cos(yaw), math.sin(yaw)
    return (x + px * c - py * s, y + px * s + py * c)


def to_robot(odom, gx: float, gy: float):
    """Odom-frame (gx, gy) into the robot frame."""
    x, y, yaw = odom
    dx, dy = gx - x, gy - y
    c, s = math.cos(-yaw), math.sin(-yaw)
    return (dx * c - dy * s, dx * s + dy * c)
