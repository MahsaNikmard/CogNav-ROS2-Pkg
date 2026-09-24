"""Per-episode metrics from a bt_runner tick log, and their aggregate.

Safety figures are always reported next to coverage, distance and motion duty,
because a robot that does not move never collides. Episodes without a contact
are censored for time-to-first-contact rather than dropped.

Two coverage measures are computed:

    traversal    area the body passed over, from the trail alone
    observation  area the camera saw, which needs `tree_log_ranges: full`

The observed area is the union of cells the rays crossed, walls and space
beyond doorways included, so it is reported in m2 without a fraction. The
fraction of the reachable floor is computed offline by scripts/seen_area.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from cognav_evaluation.occupancy import OccupancyMap
from cognav_evaluation.tick_log import Tick, parse_log, parse_preamble


@dataclass
class EpisodeMetrics:
    log: str = ""
    ticks: int = 0
    duration_s: float = 0.0
    tick_rate_hz: float = 0.0

    distance_m: float = 0.0
    motion_duty: float = 0.0
    mean_speed_mps: float = 0.0

    collisions: int = 0
    collision_ticks: int = 0
    collisions_per_100m: float = 0.0
    time_to_first_collision_s: float | None = None
    censored: bool = True
    #: From the arm's own polar vector; not comparable across arms.
    min_clearance_m: float | None = None
    clearance_p05_m: float | None = None
    #: From the platform (navmesh or lab map); the same instrument on every arm.
    true_min_clearance_m: float | None = None
    true_clearance_p05_m: float | None = None
    #: |believed - true| position, when the log carries a truth line.
    odom_drift_final_m: float | None = None
    odom_drift_max_m: float | None = None
    #: Contacts while perception reported nothing within 0.9 m.
    blind_collisions: int = 0
    blind_collision_frac: float | None = None
    nearest_at_contact_m: float | None = None

    coverage_m2: float | None = None
    coverage_frac: float | None = None
    #: Which measure coverage_m2 holds; values of different kinds do not compare.
    coverage_kind: str = "none"
    observed_m2: float | None = None
    traversal_m2: float | None = None
    traversal_frac: float | None = None
    coverage_per_m: float | None = None
    revisit_ratio: float | None = None
    #: Observed area against time, [[t_s, area_m2], ...], one sample per CURVE_SAMPLE_SEC.
    coverage_curve: list = field(default_factory=list)
    #: Time-averaged observed area, (1/T) times the integral of the curve.
    coverage_auc_m2: float | None = None

    stalled_ticks: int = 0
    #: True trajectory, [[x, y], ...], one point per PATH_STEP_M of travel.
    path_xy: list = field(default_factory=list)
    #: maneuver_clearance of the run, and how far and how often the true
    #: clearance fell below it.
    margin_m: float | None = None
    margin_violation_ticks: int | None = None
    margin_violation_frac: float | None = None
    margin_deficit_max_m: float | None = None
    margin_deficit_mean_m: float | None = None

    #: From the `# cognav end` footer: complete, canceled, shutdown, aborted,
    #: or unknown when the footer is missing.
    outcome: str = "unknown"

    max_age_ms: float = 0.0
    age_p95_ms: float = 0.0
    stale_ticks: int = 0


def _trail(ticks: list[Tick]):
    return [t.geometry_pose for t in ticks if t.geometry_pose is not None]


PATH_STEP_M = 0.05


def decimated_path(ticks, step_m: float = PATH_STEP_M):
    """The trajectory, one point per `step_m` of travel."""
    out = []
    last = None
    for x, y, _ in _trail(ticks):
        if last is None or math.hypot(x - last[0], y - last[1]) >= step_m:
            out.append([round(x, 3), round(y, 3)])
            last = (x, y)
    return out


def distance_travelled(ticks) -> float:
    trail = _trail(ticks)
    return sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(trail, trail[1:]))


def time_to_first_collision(ticks):
    """Seconds until the first contact, or None when the episode is censored."""
    for t in ticks:
        if t.contacts:
            return t.t
    return None


def revisit_ratio(ticks, radius: float = 0.30, lookback_m: float = 1.0):
    """Fraction of distance travelled within `radius` of earlier trail.

    Trail within `lookback_m` of arc length is not counted as earlier.
    """
    trail = _trail(ticks)
    if len(trail) < 3:
        return None
    arc = [0.0]
    for a, b in zip(trail, trail[1:]):
        arc.append(arc[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    total = arc[-1]
    if total <= 0.0:
        return None
    revisited = 0.0
    for i in range(1, len(trail)):
        step = arc[i] - arc[i - 1]
        for j in range(i):
            if arc[i] - arc[j] < lookback_m:
                break
            if math.hypot(trail[i][0] - trail[j][0],
                          trail[i][1] - trail[j][1]) <= radius:
                revisited += step
                break
    return revisited / total


def coverage_gridless(ticks, robot_radius: float, denominator_m2: float | None,
                      cell_m: float = 0.05):
    """(area_m2, fraction, kind) of the area the body passed over."""
    trail = _trail(ticks)
    if not trail:
        return None, None, "none"
    cells = set()
    rad = max(1, int(round(robot_radius / cell_m)))
    for x, y, _ in trail:
        c0, r0 = int(math.floor(x / cell_m)), int(math.floor(y / cell_m))
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if dr * dr + dc * dc <= rad * rad:
                    cells.add((r0 + dr, c0 + dc))
    area = len(cells) * cell_m * cell_m
    frac = (area / denominator_m2) if denominator_m2 else None
    return area, frac, "traversal-gridless"


CURVE_SAMPLE_SEC = 1.0


def coverage_auc(curve, duration_s: float | None = None):
    """Time-averaged area under a coverage curve, in m2.

    Trapezoidal from 0, holding the last sample until `duration_s`. None for a
    curve with fewer than two samples.
    """
    if not curve or len(curve) < 2:
        return None
    end = max(float(duration_s or 0.0), float(curve[-1][0]))
    if end <= 0.0:
        return None
    total, prev_t, prev_a = 0.0, 0.0, 0.0
    for t, a in curve:
        t = float(t)
        if t > prev_t:
            total += 0.5 * (prev_a + float(a)) * (t - prev_t)
        prev_t, prev_a = t, float(a)
    if end > prev_t:
        total += prev_a * (end - prev_t)
    return total / end


def time_to_area(curve, target_m2: float):
    """Seconds until the curve first reaches `target_m2`, or None (censored)."""
    for t, a in curve or ():
        if float(a) >= target_m2:
            return float(t)
    return None


def coverage_observed_gridless(ticks, denominator_m2: float | None,
                               fov_deg: float = 89.0, cell_m: float = 0.10,
                               max_range: float = 5.0, curve_out: list | None = None):
    """(area_m2, None, kind): the union of cells the rays crossed.

    Each finite bin marks cells along its ray up to its range; unobserved bins
    mark nothing. When `curve_out` is given, the running area is appended once
    per CURVE_SAMPLE_SEC of elapsed time. No fraction is returned; see the
    module docstring.
    """
    seen = set()
    half = math.radians(fov_deg) / 2.0
    used = 0
    cell_area = cell_m * cell_m
    next_sample = 0.0
    for t in ticks:
        if t.geometry_pose is None or not t.raw_ranges:
            continue
        used += 1
        x, y, yaw = t.geometry_pose
        n = len(t.raw_ranges)
        for i, d in enumerate(t.raw_ranges):
            if not math.isfinite(d) or d <= 0.0:
                continue
            d = min(float(d), max_range)
            bearing = yaw + half - (2.0 * half) * (i + 0.5) / n
            ca, sa = math.cos(bearing), math.sin(bearing)
            for k in range(int(d / cell_m) + 1):
                r = k * cell_m
                seen.add((int(math.floor((x + r * ca) / cell_m)),
                          int(math.floor((y + r * sa) / cell_m))))
        if curve_out is not None and t.t >= next_sample:
            curve_out.append([round(float(t.t), 3), len(seen) * cell_area])
            next_sample = float(t.t) + CURVE_SAMPLE_SEC
    if curve_out is not None and used and (
            not curve_out or curve_out[-1][1] < len(seen) * cell_area):
        curve_out.append([round(float(ticks[-1].t), 3), len(seen) * cell_area])
    if not used:
        return None, None, "none"
    return len(seen) * cell_area, None, "observation-gridless"


def coverage(ticks, occ: OccupancyMap | None, robot_radius: float,
             sensor_radius: float | None = None, fov_deg: float = 89.0):
    """(area_m2, fraction, kind) against the reachable area of an occupancy map.

    Observation coverage when the log carries ranges, traversal otherwise.
    """
    trail = _trail(ticks)
    if occ is None or not trail:
        return None, None, "none"
    reach = occ.reachable_mask((trail[0][0], trail[0][1]), robot_radius)
    if reach is None:
        return None, None, "start-unreachable"
    seen = np.zeros_like(reach, dtype=bool)
    res = occ.resolution
    has_raw = any(t.raw_ranges for t in ticks)

    if has_raw:
        kind = "observation"
        half = math.radians(fov_deg) / 2.0
        for t in ticks:
            if t.geometry_pose is None or not t.raw_ranges:
                continue
            x, y, yaw = t.geometry_pose
            n = len(t.raw_ranges)
            for i, d in enumerate(t.raw_ranges):
                if not math.isfinite(d):
                    d = sensor_radius or 0.0
                    if d <= 0.0:
                        continue
                bearing = yaw + half - (2 * half) * (i / max(n - 1, 1))
                steps = max(1, int(d / res))
                for k in range(steps + 1):
                    rr = res * k
                    cell = occ.world_to_cell(x + rr * math.cos(bearing),
                                             y + rr * math.sin(bearing))
                    if occ.in_bounds(*cell):
                        seen[cell] = True
    else:
        kind = "traversal"
        rad = max(1, int(round(robot_radius / res)))
        for x, y, _ in trail:
            r0, c0 = occ.world_to_cell(x, y)
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if dr * dr + dc * dc > rad * rad:
                        continue
                    if occ.in_bounds(r0 + dr, c0 + dc):
                        seen[r0 + dr, c0 + dc] = True

    seen &= reach
    area = float(seen.sum()) * res * res
    denom = float(reach.sum()) * res * res
    return area, (area / denom if denom else None), kind


def summarise(ticks: list[Tick], log_path: str = "", occ=None,
              robot_radius: float = 0.18,
              area_m2: float | None = None,
              outcome: str = "unknown",
              fov_deg: float = 89.0,
              maneuver_clearance: float | None = None) -> EpisodeMetrics:
    if not ticks:
        raise ValueError("no tick blocks recognised; is this a bt_tree log?")
    m = EpisodeMetrics(log=os.path.basename(log_path), ticks=len(ticks))
    m.outcome = outcome
    m.duration_s = ticks[-1].t - ticks[0].t
    m.tick_rate_hz = (len(ticks) - 1) / m.duration_s if m.duration_s > 0 else 0.0

    m.distance_m = distance_travelled(ticks)
    moving = [t for t in ticks if t.moving]
    m.motion_duty = len(moving) / len(ticks)
    m.mean_speed_mps = m.distance_m / m.duration_s if m.duration_s > 0 else 0.0

    m.collisions = sum(t.contacts for t in ticks)
    m.collision_ticks = sum(1 for t in ticks if t.contacts)
    m.collisions_per_100m = (100.0 * m.collisions / m.distance_m
                             if m.distance_m > 0.05 else 0.0)
    contact_ticks = [t for t in ticks if t.contacts]
    clear_view = 0.9   # safety_margin + warning_margin
    blind = [t for t in contact_ticks
             if t.nearest_m is not None and t.nearest_m > clear_view]
    m.blind_collisions = sum(t.contacts for t in blind)
    if contact_ticks:
        m.blind_collision_frac = m.blind_collisions / max(m.collisions, 1)
        seen = [t.nearest_m for t in contact_ticks if t.nearest_m is not None]
        if seen:
            m.nearest_at_contact_m = float(np.median(seen))

    ttfc = time_to_first_collision(ticks)
    m.time_to_first_collision_s = ttfc
    m.censored = ttfc is None

    clears = [t.clearance for t in ticks if t.clearance is not None]
    nearest = [t.nearest_m for t in ticks if t.nearest_m is not None]
    pool = clears or nearest
    if pool:
        m.min_clearance_m = min(pool)
        m.clearance_p05_m = float(np.percentile(pool, 5))
    true_pool = [t.true_clearance for t in ticks if t.true_clearance is not None]
    if true_pool:
        m.true_min_clearance_m = min(true_pool)
        m.true_clearance_p05_m = float(np.percentile(true_pool, 5))

    drift = [math.hypot(t.true_pose[0] - t.pose[0], t.true_pose[1] - t.pose[1])
             for t in ticks if t.true_pose is not None and t.pose is not None]
    if drift:
        m.odom_drift_final_m = float(drift[-1])
        m.odom_drift_max_m = float(max(drift))

    m.traversal_m2, m.traversal_frac, _ = coverage_gridless(
        ticks, robot_radius, area_m2)
    curve: list = []
    m.observed_m2, _, obs_kind = coverage_observed_gridless(
        ticks, area_m2, fov_deg=fov_deg, curve_out=curve)
    if m.observed_m2 is not None:
        m.coverage_curve = curve
        m.coverage_auc_m2 = coverage_auc(curve, m.duration_s)
    if occ is not None:
        m.coverage_m2, m.coverage_frac, m.coverage_kind = coverage(
            ticks, occ, robot_radius)
    elif m.observed_m2 is not None:
        m.coverage_m2, m.coverage_frac, m.coverage_kind = (
            m.observed_m2, None, obs_kind)
    else:
        m.coverage_m2, m.coverage_frac, m.coverage_kind = (
            m.traversal_m2, m.traversal_frac, "traversal-gridless")
    if m.coverage_m2 is not None and m.distance_m > 0.05:
        m.coverage_per_m = m.coverage_m2 / m.distance_m
    m.revisit_ratio = revisit_ratio(ticks)

    m.stalled_ticks = sum(1 for t in ticks if not t.moving)
    m.path_xy = decimated_path(ticks)

    # The navmesh is already eroded by the body radius, so a true clearance
    # below maneuver_clearance means the body is inside the policy's own margin.
    if maneuver_clearance is not None:
        m.margin_m = float(maneuver_clearance)
        band = [m.margin_m - t.true_clearance for t in ticks
                if t.true_clearance is not None]
        if band:
            inside = [d for d in band if d > 0.0]
            m.margin_violation_ticks = len(inside)
            m.margin_violation_frac = len(inside) / len(band)
            m.margin_deficit_max_m = max(max(band), 0.0)
            m.margin_deficit_mean_m = (sum(inside) / len(inside)) if inside else 0.0

    ages = [t.age_ms for t in ticks]
    m.max_age_ms = max(ages)
    m.age_p95_ms = float(np.percentile(ages, 95))
    m.stale_ticks = sum(1 for t in ticks if not t.fresh)
    return m


def _stat(values):
    """Median, quartiles and count of a list, ignoring None."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    a = np.asarray(vals, dtype=float)
    return {"median": float(np.median(a)),
            "q1": float(np.percentile(a, 25)),
            "q3": float(np.percentile(a, 75)),
            "n": len(vals)}


def aggregate(episodes: list[EpisodeMetrics]) -> dict:
    """Median and IQR per metric for one arm on one scene.

    Time to first contact is summarised only over the episodes that collided,
    with the number of censored episodes alongside.
    """
    if not episodes:
        return {}
    collided = [m for m in episodes if not m.censored]
    total_contacts = sum(m.collisions for m in episodes)
    blind = sum(m.blind_collisions for m in episodes)
    unverified = sorted(m.log for m in episodes if m.outcome != "complete")
    return {
        "episodes": len(episodes),
        "unverified": len(unverified),
        "unverified_logs": unverified[:10],
        "duration_s": _stat([m.duration_s for m in episodes]),
        "coverage_frac": _stat([m.coverage_frac for m in episodes]),
        "coverage_m2": _stat([m.coverage_m2 for m in episodes]),
        "coverage_per_m": _stat([m.coverage_per_m for m in episodes]),
        "coverage_auc_m2": _stat([m.coverage_auc_m2 for m in episodes]),
        "distance_m": _stat([m.distance_m for m in episodes]),
        "motion_duty": _stat([m.motion_duty for m in episodes]),
        "revisit_ratio": _stat([m.revisit_ratio for m in episodes]),
        "collisions": _stat([m.collisions for m in episodes]),
        "collisions_per_100m": _stat([m.collisions_per_100m for m in episodes]),
        "episodes_with_contact": len(collided),
        "censored": len(episodes) - len(collided),
        "ttfc_median_among_collided": _stat(
            [m.time_to_first_collision_s for m in collided]),
        "total_contacts": total_contacts,
        "blind_contacts": blind,
        "blind_fraction": (blind / total_contacts) if total_contacts else None,
        "coverage_kinds": sorted({m.coverage_kind for m in episodes}),
        "observed_m2": _stat([m.observed_m2 for m in episodes]),
        "traversal_m2": _stat([m.traversal_m2 for m in episodes]),
        "traversal_frac": _stat([m.traversal_frac for m in episodes]),
        "min_clearance_m": _stat([m.min_clearance_m for m in episodes]),
        "true_min_clearance_m": _stat([m.true_min_clearance_m for m in episodes]),
        "margin_violation_frac": _stat([m.margin_violation_frac for m in episodes]),
        "margin_deficit_max_m": _stat([m.margin_deficit_max_m for m in episodes]),
        "odom_drift_final_m": _stat([m.odom_drift_final_m for m in episodes]),
        "odom_drift_max_m": _stat([m.odom_drift_max_m for m in episodes]),
        "true_clearance_p05_m": _stat([m.true_clearance_p05_m for m in episodes]),
        "tick_rate_hz": _stat([m.tick_rate_hz for m in episodes]),
        "max_age_ms": _stat([m.max_age_ms for m in episodes]),
    }


def _fmt(stat, scale=1.0, unit="", nd=1):
    if stat is None:
        return "n/a"
    return (f"{stat['median'] * scale:.{nd}f}{unit}  "
            f"[{stat['q1'] * scale:.{nd}f}, {stat['q3'] * scale:.{nd}f}]")


def render_aggregate(agg: dict, label: str = "") -> str:
    if not agg:
        return "  no episodes"
    n = agg["episodes"]
    head = f"\n=== {label}  {n} episodes ==="
    if agg.get("unverified"):
        head += (f"\n  !! {agg['unverified']} of {n} did not report a completed "
                 "mission; their coverage is a lower bound"
                 f"\n     {', '.join(agg['unverified_logs'])}")
    out = [head, f"  {'metric':24} {'median [IQR]':28} n"]
    rows = [
        ("coverage m2 (seen)", agg.get("observed_m2"), 1.0, "", 1),
        ("coverage % (traversal)", agg.get("traversal_frac"), 100.0, "", 1),
        ("coverage m2", agg["coverage_m2"], 1.0, "", 1),
        ("coverage m2 time-avg", agg.get("coverage_auc_m2"), 1.0, "", 1),
        ("coverage m2 per m", agg["coverage_per_m"], 1.0, "", 2),
        ("distance m", agg["distance_m"], 1.0, "", 1),
        ("motion duty %", agg["motion_duty"], 100.0, "", 0),
        ("revisit %", agg["revisit_ratio"], 100.0, "", 1),
        ("collisions", agg["collisions"], 1.0, "", 0),
        ("collisions per 100 m", agg["collisions_per_100m"], 1.0, "", 1),
        ("min clearance m (self)", agg["min_clearance_m"], 1.0, "", 2),
        ("min clearance m (true)", agg.get("true_min_clearance_m"), 1.0, "", 2),
        ("inside safety margin %", agg.get("margin_violation_frac"), 100.0, "", 1),
        ("worst margin deficit m", agg.get("margin_deficit_max_m"), 1.0, "", 2),
        ("odom drift m (final)", agg.get("odom_drift_final_m"), 1.0, "", 2),
        ("clearance p05 m (true)", agg.get("true_clearance_p05_m"), 1.0, "", 2),
        ("tick rate Hz", agg["tick_rate_hz"], 1.0, "", 1),
    ]
    for name, stat, scale, unit, nd in rows:
        cnt = stat["n"] if stat else 0
        out.append(f"  {name:24} {_fmt(stat, scale, unit, nd):28} {cnt}")
    out.append(f"\n  contact      {agg['episodes_with_contact']} of {n} episodes "
               f"({agg['censored']} censored, i.e. never collided)")
    if agg["total_contacts"]:
        out.append(f"  first contact {_fmt(agg['ttfc_median_among_collided'], 1.0, 's')} "
                   f"among those that collided; compare across arms only with "
                   f"matching censored counts")
        out.append(f"  BLIND        {agg['blind_contacts']} of {agg['total_contacts']} "
                   f"contacts happened with nothing reported inside 0.9 m "
                   f"({100 * agg['blind_fraction']:.0f}%)")
    return "\n".join(out)


def headline(m: EpisodeMetrics) -> str:
    """Coverage, collisions, distance and motion duty together."""
    cov = ("n/a" if m.coverage_m2 is None
           else f"{m.coverage_m2:.1f} m2 ({m.coverage_kind})"
           if m.coverage_frac is None
           else f"{100 * m.coverage_frac:.1f}% ({m.coverage_m2:.1f} m2, {m.coverage_kind})")
    ttfc = "none (censored)" if m.censored else f"{m.time_to_first_collision_s:.1f}s"
    return (f"  coverage      {cov}\n"
            f"  collisions    {m.collisions} in {m.duration_s:.0f}s "
            f"({m.collisions_per_100m:.1f} per 100 m)\n"
            f"  first contact {ttfc}\n"
            f"  distance      {m.distance_m:.1f} m over {m.duration_s:.0f}s "
            f"({m.mean_speed_mps:.2f} m/s)\n"
            f"  motion duty   {100 * m.motion_duty:.0f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--map", help="ROS map_server YAML for the coverage denominator")
    ap.add_argument("--robot-radius", type=float, default=0.18)
    ap.add_argument("--area", type=float, default=None,
                    help="coverage denominator in m2, overriding the log preamble")
    ap.add_argument("--json", help="write per-episode metrics and the aggregate here")
    ap.add_argument("--brief", action="store_true", help="one line per log")
    ap.add_argument("--label", default="", help="name for the aggregate block")
    ap.add_argument("--no-aggregate", action="store_true")
    args = ap.parse_args()

    occ = OccupancyMap(args.map) if args.map else None
    out, scored = [], []
    for path in args.logs:
        run = parse_preamble(path)
        area = args.area if args.area is not None else run.get("navigable_area_m2")
        ticks = parse_log(path)
        if not ticks:
            first = ""
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    first = next((line.strip() for line in fh if line.strip()), "")
            except OSError:
                pass
            print(f"{os.path.basename(path)}: no ticks"
                  + (f": {first[:90]}" if first else ""))
            continue
        margin = run.get("maneuver_clearance")
        m = summarise(ticks, path, occ, args.robot_radius, area_m2=area,
                      outcome=run.get("outcome", "unknown"),
                      fov_deg=float(run.get("fov_deg", 89.0)),
                      maneuver_clearance=(float(margin) if margin is not None
                                          else None))
        out.append(asdict(m))
        scored.append(m)
        if run and not args.brief and not args.no_aggregate:
            desc = "  ".join(f"{k}={run[k]}" for k in
                             ("scene", "depth_scale", "num_bins", "seed")
                             if k in run)
            if desc:
                print(f"  run: {desc}")
        if args.brief:
            cov = ("n/a" if m.coverage_m2 is None
                   else f"{m.coverage_m2:5.1f}m2" if m.coverage_frac is None
                   else f"{100 * m.coverage_frac:4.1f}%")
            print(f"cov {cov}  coll {m.collisions:>3}  dist {m.distance_m:5.1f}m  "
                  f"duty {100 * m.motion_duty:3.0f}%  "
                  f"{m.ticks} ticks")
            continue
        print(f"\n=== {os.path.basename(path)}  ({m.ticks} ticks @ {m.tick_rate_hz:.1f} Hz) ===")
        print(headline(m))
        if m.collisions:
            print(f"  BLIND         {m.blind_collisions}/{m.collisions} contacts while "
                  f"perception reported >0.9 m clear; median reported range at "
                  f"contact {m.nearest_at_contact_m} m")
        revisit = "n/a" if m.revisit_ratio is None else f"{100 * m.revisit_ratio:.1f}%"
        print(f"  revisit       {revisit}")
        print(f"  clearance     min {m.min_clearance_m} m, p05 {m.clearance_p05_m} m")
        print(f"  frame age     p95 {m.age_p95_ms:.0f} ms, max {m.max_age_ms:.0f} ms, "
              f"stale ticks {m.stale_ticks}")
    agg = aggregate(scored) if len(scored) > 1 else {}
    if agg and not args.no_aggregate:
        print(render_aggregate(agg, args.label or "all episodes"))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"episodes": out, "aggregate": agg}, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
