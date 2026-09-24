"""Retry after a rejected command, and the Escape rung when no forward command exists.

The deadlock scene has a single obstacle at 0.30 m on the rightmost real bin
and open space elsewhere: every forward arc is blocked, turning in place is not.
"""
import numpy as np

from _harness import ANGLES, N_BINS, PADDING, REAL_BINS, snapshot
from _check import Checker

from cognav_representation.geometry import swept_arc_check
from cognav_bt_runner.xml_tree import RetryNode, load_tree
from cognav_bt_behaviors.status import Status

from cognav_bt_behaviors.base import BehaviorConfig

# The shipped footprint.
_CFG = BehaviorConfig()
HALF_WIDTH = _CFG.robot_radius + _CFG.maneuver_clearance
DEADLOCK_BIN = 81                 # rightmost real bin


def deadlock_scene():
    """One obstacle at 0.30 m, everything else open."""
    r = np.full(N_BINS + 2 * PADDING, 1.5)
    r[PADDING:PADDING + N_BINS] = 2.5
    r[DEADLOCK_BIN] = 0.30
    return r


class _Stub:
    """Minimal stand-in for a tree node, to test RetryNode in isolation."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0
        self.name, self.glyph, self.status, self.last_tick = "stub", "-->", None, -1

    @property
    def children(self):
        return []

    def tick(self, snapshot, blackboard, context):
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return result


class _Ctx:
    tick_id = 0


def main():
    c = Checker()

    print("\n[1] retry stops at the first success and is bounded")
    for results, attempts, expect_status, expect_calls in [
        ([Status.SUCCESS], 3, Status.SUCCESS, 1),
        ([Status.FAILURE, Status.SUCCESS], 3, Status.SUCCESS, 2),
        ([Status.FAILURE], 3, Status.FAILURE, 3),
        ([Status.FAILURE], 1, Status.FAILURE, 1),
    ]:
        stub = _Stub(results)
        node = RetryNode(stub, attempts)
        status = node.tick(None, {}, _Ctx())
        c(f"{[r.name for r in results]} x{attempts} -> {status.name} in {stub.calls}",
          status == expect_status and stub.calls == expect_calls,
          f"expected {expect_status.name} in {expect_calls}")

    print("\n[2] no forward command escapes the deadlock")
    ranges = deadlock_scene()
    forward_blocked = True
    for v in (0.05, 0.1, 0.2, 0.35, 0.5):
        for w in (-1.0, -0.5, 0.0, 0.5, 1.0):
            if not swept_arc_check(ranges, ANGLES, v, w, HALF_WIDTH, 1.5, REAL_BINS):
                forward_blocked = False
    c("every (v>0, w) combination is blocked", forward_blocked)
    rotate = swept_arc_check(ranges, ANGLES, 0.0, 0.6, HALF_WIDTH, 1.5, REAL_BINS)
    c("rotation in place is clear", not rotate,
      "a disc turning about its centre sweeps no new space")

    print("\n[3] the full mission loads")
    from ament_index_python.packages import get_package_share_directory
    from pathlib import Path
    xml = (Path(get_package_share_directory("cognav_mission"))
           / "mission" / "random_explore.xml").read_text()
    try:
        root = load_tree(xml)
        names = []

        def walk(n):
            names.append(n.name)
            for ch in n.children:
                walk(ch)
        walk(root)
        c("loads and validates", True)
        c("Retry is in the tree", "ProposeUntilSafe" in names)
        c("Escape rung is in the tree", "Escape" in names)
        c("ExecuteCommand sits outside the retry", "ExecuteCommand" in names)
    except Exception as exc:  # noqa: BLE001
        c("loads and validates", False, str(exc))

    print("\n[4] a malformed retry is rejected at load")
    def _load(inner):
        return load_tree(
            '<root BTCPP_format="4" main_tree_to_execute="T"><BehaviorTree ID="T">'
            + inner + "</BehaviorTree></root>")
    for label, inner in [
        ("two children", '<Retry><Brake/><Brake/></Retry>'),
        ("no children", '<Retry></Retry>'),
        ("num_attempts=0", '<Retry num_attempts="0"><Brake/></Retry>'),
        ("num_attempts=abc", '<Retry num_attempts="abc"><Brake/></Retry>'),
    ]:
        try:
            _load(inner)
            c(label, False, "loaded when it should have been rejected")
        except ValueError:
            c(label, True)

    # A zero proposal passes the gate, so without HasMotion Navigate would
    # succeed while the robot stands still.
    print("\n[5] every kind of deadlock reaches the escape rung")
    import rclpy
    from cognav_bt_behaviors.base import BehaviorConfig, BehaviorContext
    from cognav_bt_behaviors.blackboard import Blackboard
    from _harness import mission_xml, scene

    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node("escape_test")
    cfg = BehaviorConfig(safety_margin=0.50, warning_margin=0.40, max_v=0.5)

    # stall_ticks is read from the mission port.
    import re as _re
    _port = _re.search(r'<IsStalled[^>]*stall_ticks="(\d+)"', mission_xml())
    STALL = int(_port.group(1)) if _port else 3
    print(f"  (mission commits to stall_ticks={STALL}; driving {STALL + 3} ticks)")

    def drive(ranges, ticks):
        tree = load_tree(mission_xml())
        bb = Blackboard()
        context = BehaviorContext(node, None, cfg, enable_cmd_vel=False)
        trace = []
        for t in range(1, ticks + 1):
            context.tick_id = t
            tree.tick(snapshot(ranges, stamp=float(t)), bb, context)
            won = next((n.name for n in tree.children if n.last_tick == t
                        and n.status is not None and n.status == Status.SUCCESS), "Brake")
            trace.append((bb.get("cmd_vel", (0.0, 0.0)), won, bb.get("brake_now")))
        return trace

    # A cluttered scene where no forward command is drivable and brake_now
    # stays False; only the stall detector sees it.
    bar = "." * 12 + ":" * 5 + "+" * 16 + "#" * 4 + "+" + "#" * 34
    band = {".": 2.5, ":": 1.5, "+": 0.70, "#": 0.35}
    t140 = np.full(N_BINS + 2 * PADDING, 1.5)
    t140[PADDING:PADDING + N_BINS] = [band[ch] for ch in bar]
    t140[PADDING + bar.rindex("#")] = 0.31
    stalled = drive(t140, STALL + 3)
    c("clutter: Navigate never succeeds",
      all(won != "Navigate" for _, won, _ in stalled))
    c("clutter: brake_now stays False",
      all(b is False for _, _, b in stalled))
    c("clutter: escapes by turning in place",
      stalled[-1][1] == "Escape" and stalled[-1][0][0] == 0.0 and stalled[-1][0][1] != 0.0,
      f"final {stalled[-1]}")

    wedge = drive(scene(0.32), STALL + 3)          # whole FOV inside the clearance band
    c("wedge: never reports Navigate success",
      all(won != "Navigate" for _, won, _ in wedge))
    c("wedge: escapes by turning in place",
      wedge[-1][1] == "Escape" and wedge[-1][0][0] == 0.0 and wedge[-1][0][1] != 0.0,
      f"final {wedge[-1]}")

    gate = drive(deadlock_scene(), STALL + 3)      # single obstacle, every arc blocked
    c("gate deadlock: escapes by turning in place",
      gate[-1][1] == "Escape" and gate[-1][0][0] == 0.0 and gate[-1][0][1] != 0.0,
      f"final {gate[-1]}")

    open_scene = drive(scene(3.0), 3)      # regression: normal driving unaffected
    c("open scene: still drives forward under Navigate",
      all(won == "Navigate" and cmd[0] > 0.0 for cmd, won, _ in open_scene),
      f"final {open_scene[-1]}")

    c.finish("retry feedback and the escape rung")


if __name__ == "__main__":
    main()
