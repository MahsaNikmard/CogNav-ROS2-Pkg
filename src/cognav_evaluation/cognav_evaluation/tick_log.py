"""Parse a bt_runner tick log (written by cognav_bt_runner.tree_log) into per-tick records."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

_TICK = re.compile(
    r"^──\s*tick\s+(?P<tick>\d+)\s+t=\+(?P<t>[\d.]+)s\s+age=(?P<age>[\d.]+)ms\s+"
    r"fresh=(?P<fresh>[YN])\s+brake=(?P<brake>[YN])\s+blocked=(?P<blocked>\d+)"
    r"(?:\s+clearance=(?P<clearance>[\d.]+)m)?"
    r"(?:\s+true_clr=(?P<true_clearance>[\d.]+)m)?\s+"
    r"cmd=\((?P<v>[+-][\d.]+),\s*(?P<w>[+-][\d.]+)\)")
_POSE = re.compile(
    r"^\s+pose\s+x=(?P<x>[+-][\d.]+)\s+y=(?P<y>[+-][\d.]+)\s+yaw=(?P<yaw>[+-][\d.]+)deg")
# Written only when the true pose differs from odometry.
_TRUE_POSE = re.compile(
    r"^\s+truth\s+x=(?P<x>[+-][\d.]+)\s+y=(?P<y>[+-][\d.]+)\s+yaw=(?P<yaw>[+-][\d.]+)deg")
_WAYPOINT = re.compile(
    r"^\s+waypoint\s+\((?P<wx>[+-][\d.]+),\s*(?P<wy>[+-][\d.]+)\)[^=]*"
    r"(?:chain=(?P<chain>\d+))?")
_RANGES = re.compile(
    r"^\s+ranges\s+nearest=(?P<nearest>[\d.]+)m\s+@\s+(?P<at>[+-][\d.]+)deg\s+"
    r"observed=(?P<obs>\d+)/(?P<total>\d+)")
_RAW = re.compile(r"^\s+raw\s+(?P<values>.+)$")
_CONTACT = re.compile(r"^\s+CONTACT\s+(?P<hits>\d+)\s+navmesh contact")
_NODE = re.compile(r"^(?P<indent>\s*)(?:\[[-oR]\]|-->)\s+(?P<name>\S+)\s+(?P<status>[✓✕·])")


@dataclass
class Tick:
    tick: int
    t: float
    age_ms: float
    fresh: bool
    brake: bool
    blocked: int
    v: float
    w: float
    #: From the arm's own polar vector.
    clearance: float | None = None
    #: From the platform; the same instrument on every arm.
    true_clearance: float | None = None
    pose: tuple[float, float, float] | None = None
    true_pose: tuple[float, float, float] | None = None
    waypoint: tuple[float, float] | None = None
    chain: int = 0
    nearest_m: float | None = None
    nearest_deg: float | None = None
    observed: int = 0
    total_bins: int = 0
    contacts: int = 0
    branch: str | None = None
    raw_ranges: list[float] = field(default_factory=list)

    @property
    def geometry_pose(self):
        """The true pose when logged, else odometry. Use it for every metric."""
        return self.true_pose if self.true_pose is not None else self.pose

    @property
    def moving(self) -> bool:
        return abs(self.v) > 1e-6 or abs(self.w) > 1e-6


#: Top-level rungs of the missions; `branch` is the first that succeeded.
_RUNGS = ("MissionComplete", "Navigate", "Escape", "Brake")


_PREAMBLE = re.compile(r"^#\s{2,}(?P<key>[a-z0-9_]+):\s*(?P<value>.+?)\s*$")


def parse_preamble(path) -> dict:
    """Read the `# cognav run` preamble and the `# cognav end` footer."""
    out = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                if out or line.strip():
                    break
                continue
            m = _PREAMBLE.match(line)
            if m:
                out[m["key"]] = m["value"]
    for key in ("navigable_area_m2", "depth_scale", "max_v", "fov_deg",
                "safety_margin"):
        if key in out:
            try:
                out[key] = float(out[key])
            except ValueError:
                pass
    for key in ("num_bins", "seed"):
        if key in out:
            try:
                out[key] = int(float(out[key]))
            except ValueError:
                pass
    out.update(parse_footer(path))
    return out


_FOOTER = re.compile(r"^#\s*cognav end\s+(?P<fields>.*\S)\s*$")


def parse_footer(path) -> dict:
    """Fields of the `# cognav end` footer, or {} when the log has none.

    Only the last 4 kB of the file are read.
    """
    with open(path, "rb") as fh:
        try:
            fh.seek(-4096, 2)
        except OSError:
            fh.seek(0)
        tail = fh.read().decode("utf-8", errors="replace")
    for line in reversed(tail.splitlines()):
        m = _FOOTER.match(line)
        if not m:
            continue
        out = {}
        for token in m["fields"].split():
            key, _, value = token.partition("=")
            out["outcome" if key == "outcome" else f"end_{key}"] = value
        for key in ("end_ticks", "end_contacts"):
            if key in out:
                try:
                    out[key] = int(out[key])
                except ValueError:
                    pass
        if "end_elapsed_s" in out:
            try:
                out["end_elapsed_s"] = float(out["end_elapsed_s"])
            except ValueError:
                pass
        return out
    return {}


def parse_log(path) -> list[Tick]:
    ticks: list[Tick] = []
    current: Tick | None = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _TICK.match(line)
            if m:
                g = m.groupdict()
                current = Tick(
                    tick=int(g["tick"]), t=float(g["t"]), age_ms=float(g["age"]),
                    fresh=g["fresh"] == "Y", brake=g["brake"] == "Y",
                    blocked=int(g["blocked"]), v=float(g["v"]), w=float(g["w"]),
                    clearance=float(g["clearance"]) if g["clearance"] else None,
                    true_clearance=(float(g["true_clearance"])
                                    if g.get("true_clearance") else None))
                ticks.append(current)
                continue
            if current is None:
                continue
            m = _TRUE_POSE.match(line)
            if m is not None and current is not None:
                current.true_pose = (float(m["x"]), float(m["y"]),
                                     math.radians(float(m["yaw"])))
                continue

            m = _POSE.match(line)
            if m:
                current.pose = (float(m["x"]), float(m["y"]),
                                math.radians(float(m["yaw"])))
                continue
            m = _WAYPOINT.match(line)
            if m:
                current.waypoint = (float(m["wx"]), float(m["wy"]))
                current.chain = int(m["chain"] or 0)
                continue
            m = _RANGES.match(line)
            if m:
                current.nearest_m = float(m["nearest"])
                current.nearest_deg = float(m["at"])
                current.observed = int(m["obs"])
                current.total_bins = int(m["total"])
                continue
            m = _RAW.match(line)
            if m:
                current.raw_ranges = [
                    float("inf") if v == "inf" else float(v)
                    for v in m["values"].split()]
                continue
            m = _CONTACT.match(line)
            if m:
                current.contacts += int(m["hits"])
                continue
            m = _NODE.match(line)
            if m and current.branch is None:
                name, status = m["name"], m["status"]
                if status == "✓" and name in _RUNGS:
                    current.branch = name
    return ticks
