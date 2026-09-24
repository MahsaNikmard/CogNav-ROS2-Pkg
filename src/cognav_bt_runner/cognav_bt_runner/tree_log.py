"""Per-tick tree log: the inputs of each tick and the status of every node.

Each tick is rendered as a block: a header with the state the tree was ticked
against, then one line per node.

    ── tick 42  t=+16.80s  age=5ms  fresh=Y  brake=N  blocked=3  cmd=(0.35, -0.42)
    [o] Priorities                    ✓ SUCCESS
        [-] MissionComplete           ✕ FAILURE
            --> IsMissionComplete     ✕ FAILURE
            --> Brake                 ·
        [-] Navigate                  ✓ SUCCESS
            [R] ProposeUntilSafe      ✓ SUCCESS
                [-] ProposeAndVerify  ✓ SUCCESS
                    --> RRTExploreWaypoint ✓ SUCCESS
                    --> FollowTheGap  ✓ SUCCESS
                    --> IsCommandSafe ✓ SUCCESS
            --> HasWaypoint           ✓ SUCCESS
            --> HasMotion             ✓ SUCCESS
            --> ExecuteCommand        ✓ SUCCESS
        [-] Escape                    ·
        --> Brake                     ·

A node not ticked this tick is drawn with `·` instead of a stale status.
`cognav_evaluation.tick_log` parses this format.
"""

from __future__ import annotations

import math
from pathlib import Path

from cognav_representation.geometry import wrap_angle

from cognav_bt_behaviors.status import Status

_GLYPH = {
    Status.SUCCESS: "✓ SUCCESS",
    Status.FAILURE: "✕ FAILURE",
    Status.RUNNING: "* RUNNING",
}
_NOT_VISITED = "·"

#: Column the status is printed at, so glyphs line up across depths.
_STATUS_COLUMN = 46


def render_tick(node, tick_id: int, header: str) -> str:
    """One tick block: header line plus the tree."""
    lines = [header]
    _render_node(node, tick_id, 0, lines)
    return "\n".join(lines) + "\n"


def _render_node(node, tick_id: int, depth: int, lines: list) -> None:
    visited = node.last_tick == tick_id
    label = f"{'    ' * depth}{node.glyph} {node.name}"
    status = _GLYPH.get(node.status, str(node.status)) if visited else _NOT_VISITED
    lines.append(f"{label:<{_STATUS_COLUMN}}{status}")
    for child in node.children:
        _render_node(child, tick_id, depth + 1, lines)


def _profile_char(distance, margin, warn):
    if not math.isfinite(distance):
        return " "
    if distance < margin:
        return "#"
    if distance < warn:
        return "+"
    if distance < 2.0:
        return ":"
    return "."


def _scene_profile(snapshot, margin, warn, columns=72) -> list:
    """One character per column across the field of view, left to right.

    `#` is inside the safety margin, `+` inside the warning band, `:` under
    2 m, `.` further, and a space is unobserved.
    """
    start, end = snapshot.real_bin_range
    ranges = snapshot.scan_ranges[start:end]
    angles = snapshot.scan_angles[start:end]
    if len(ranges) == 0:
        return []
    step = max(1, len(ranges) // columns)
    chars, buckets = [], []
    for i in range(0, len(ranges), step):
        window = ranges[i:i + step]
        finite = [r for r in window if math.isfinite(r)]
        nearest = min(finite) if finite else float("inf")
        buckets.append(nearest)
        chars.append(_profile_char(nearest, margin, warn))
    left = math.degrees(angles[0])
    right = math.degrees(angles[-1])
    return [
        f"    scene     {left:+.0f}deg [{''.join(chars)}] {right:+.0f}deg",
        f"              # <{margin:.2f}m   + <{warn:.2f}m   : <2m   . beyond   ' ' unobserved",
    ]


def build_header(tick_id: int, elapsed_sec: float, snapshot, blackboard,
                 margin: float = 0.5, warn: float = 0.9, ranges_mode: str = "profile") -> str:
    """Header lines: timing, pose, waypoint, ranges, contacts and the command."""
    if snapshot is None:
        return f"\n── tick {tick_id}  t=+{elapsed_sec:.2f}s  no snapshot (waiting for perception and odometry)"

    age = "n/a" if snapshot.age_sec is None else f"{snapshot.age_sec * 1e3:.0f}ms"
    fresh = "Y" if snapshot.perception_fresh else "N"
    brake = "Y" if blackboard.get("brake_now", False) else "N"
    blocked = blackboard.get("blocked_bins", [])
    cmd = blackboard.get("cmd_vel", (0.0, 0.0))
    clear = blackboard.get("clearance")
    gov = ""
    if clear is not None and clear != float("inf"):
        gov += f"  clearance={clear:.2f}m"
    # `clearance` comes from the polar vector; `true_clr` from the platform.
    true_clear = blackboard.get("true_clearance")
    if true_clear is not None and true_clear == true_clear:  # not NaN
        gov += f"  true_clr={true_clear:.2f}m"
    lines = [
        f"\n── tick {tick_id}  t=+{elapsed_sec:.2f}s  age={age}  fresh={fresh}  "
        f"brake={brake}  blocked={len(blocked)}{gov}  cmd=({cmd[0]:+.2f}, {cmd[1]:+.2f})"
    ]

    if snapshot.odom is not None:
        x, y, yaw = snapshot.odom
        skew = "n/a" if snapshot.stamp_skew_sec is None else f"{snapshot.stamp_skew_sec * 1e3:.0f}ms"
        lines.append(
            f"    pose      x={x:+.2f} y={y:+.2f} yaw={math.degrees(yaw):+.1f}deg"
            f"   odom/frame skew={skew}"
        )
        # Written only when the platform's true pose differs from odometry.
        truth = blackboard.get("true_pose")
        if truth is not None and any(abs(a - b) > 5e-3 for a, b in zip(truth, (x, y, yaw))):
            tx, ty, tyaw = truth
            lines.append(
                f"    truth     x={tx:+.2f} y={ty:+.2f} yaw={math.degrees(tyaw):+.1f}deg"
                f"   drift={math.hypot(tx - x, ty - y):.2f}m"
            )

    path = blackboard.get("reference_path", [])
    if path and snapshot.odom is not None:
        x, y, yaw = snapshot.odom
        wx, wy = path[0]
        bearing = math.degrees(wrap_angle(math.atan2(wy - y, wx - x) - yaw))
        lines.append(
            f"    waypoint  ({wx:+.2f}, {wy:+.2f}) odom"
            f"   bearing={bearing:+.1f}deg  dist={math.hypot(wx - x, wy - y):.2f}m"
            f"   chain={len(path)}"
        )
    else:
        lines.append("    waypoint  none committed")

    if snapshot.scan_ranges is not None and ranges_mode != "none":
        start, end = snapshot.real_bin_range
        real = snapshot.scan_ranges[start:end]
        finite = [r for r in real if math.isfinite(r)]
        if finite:
            nearest = min(finite)
            idx = int(list(real).index(nearest))
            bearing = math.degrees(snapshot.scan_angles[start + idx])
            lines.append(
                f"    ranges    nearest={nearest:.2f}m @ {bearing:+.1f}deg"
                f"   observed={len(finite)}/{len(real)}"
            )
        else:
            lines.append(f"    ranges    nothing observed in {len(real)} real bins")
        lines.extend(_scene_profile(snapshot, margin, warn))
        if ranges_mode == "full":
            values = " ".join("inf" if not math.isfinite(r) else f"{r:.2f}" for r in real)
            lines.append(f"    raw       {values}")

    hits = blackboard.get("contacts_this_tick", 0)
    if hits:
        lines.append(
            f"    CONTACT   {hits} navmesh contact(s) this tick, "
            f"{blackboard.get('contacts_total', 0)} total"
            f"   <-- the robot hit something the camera did not report"
        )

    if blocked:
        nearest = blocked[0]
        lines.append(
            f"    blocked   {len(blocked)} bins, nearest bin {nearest[0]} at {nearest[1]:.2f}m"
        )

    return "\n".join(lines)


class TreeLogger:
    """Writes a tick block per tick. Disabled instances cost nothing."""

    def __init__(self, path: Path | None, every_n: int = 1, logger=None,
                 ranges_mode: str = "profile"):
        self.path = path
        self.every_n = max(1, int(every_n))
        self.ranges_mode = ranges_mode
        self._handle = None
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # Line buffered, so a killed run still leaves a complete log.
        self._handle = path.open("w", buffering=1, encoding="utf-8")
        if logger is not None:
            logger.info(f"BT tick log: {path}")

    @property
    def enabled(self) -> bool:
        return self._handle is not None

    def write_tick(self, node, tick_id: int, elapsed_sec: float, snapshot, blackboard,
                   margin: float = 0.5, warn: float = 0.9) -> None:
        if self._handle is None or tick_id % self.every_n:
            return
        self._handle.write(render_tick(
            node, tick_id,
            build_header(tick_id, elapsed_sec, snapshot, blackboard,
                         margin, warn, self.ranges_mode),
        ))

    def write_preamble(self, fields: dict) -> None:
        """Write the `# cognav run` block that makes the log self-describing."""
        if self._handle is None:
            return
        self._handle.write("# cognav run\n")
        for key in sorted(fields):
            value = fields[key]
            if value not in (None, ""):
                self._handle.write(f"#   {key}: {value}\n")
        self._handle.write("\n")
        self._handle.flush()

    def write_note(self, text: str) -> None:
        """Record an event between ticks, such as perception starvation."""
        if self._handle is not None:
            self._handle.write(f"\n!! {text}\n")

    def close(self, outcome: str = "unknown", **fields) -> None:
        """Write the `# cognav end` footer, which marks a finished episode, and close."""
        if self._handle is None:
            return
        parts = " ".join(f"{k}={v}" for k, v in sorted(fields.items()))
        self._handle.write(f"\n# cognav end outcome={outcome}"
                           + (f" {parts}" if parts else "") + "\n")
        self._handle.flush()
        self._handle.close()
        self._handle = None
