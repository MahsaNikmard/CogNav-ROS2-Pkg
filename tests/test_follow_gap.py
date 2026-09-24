"""Follow-the-gap steering, and safety as a pass/fail gate."""
import math

import rclpy
from cognav_bt_behaviors.base import BehaviorConfig, BehaviorContext
from cognav_bt_behaviors.blackboard import Blackboard
from cognav_bt_behaviors.driving import FollowTheGap
from cognav_representation.geometry import find_gaps, steer_through_gaps
from cognav_bt_runner.xml_tree import load_tree

from _harness import ANGLES, N_BINS, PADDING, REAL_BINS, Checker, mission_xml, scene, snapshot

check = Checker()
rclpy.init()
node = rclpy.create_node("fgm")
CFG = BehaviorConfig(robot_radius=0.18, maneuver_clearance=0.10,
                     safety_margin=0.50, warning_margin=0.40, max_v=0.5, k_w=1.5)


def ctx():
    return BehaviorContext(node, None, CFG, enable_cmd_vel=False)


def deg(rad):
    return math.degrees(rad)


HALF = CFG.robot_radius + CFG.maneuver_clearance
MINR = CFG.safety_margin + CFG.warning_margin


def drive(ranges, waypoint):
    bb = Blackboard()
    bb.set("reference_path", [waypoint])
    FollowTheGap().tick(snapshot(ranges), bb, ctx())
    bb.set("cmd_vel", bb.get("cmd_proposal", (0.0, 0.0)))
    return bb


print("\n1. a bearing is drivable only if the body clears both walls")
slit = PADDING + N_BINS // 2
narrow = scene(0.85)
narrow[slit] = 4.0                                   # one-bin slit
check("single-bin slit is not a gap",
      not find_gaps(narrow, ANGLES, MINR, HALF, REAL_BINS))

wide = scene(0.85)
wide[slit - 20:slit + 21] = 4.0                      # opening the body fits
gaps = find_gaps(wide, ANGLES, MINR, HALF, REAL_BINS)
check("a wide opening is a gap", len(gaps) == 1,
      f"{len(gaps)} gaps, {gaps[0][1]-gaps[0][0]+1} bearings wide")

check("walls beyond min_range are not obstacles at all",
      len(find_gaps(scene(1.20), ANGLES, MINR, HALF, REAL_BINS)) == 1)

print("\n2. waypoint inside a gap: steer at the waypoint, ignore depth")
# One obstacle splits the view: a shallow region where the waypoint is, and a
# deeper one on the far side of it.
two = scene(1.60)                               # open, shallow-ish
two[PADDING + 8:PADDING + 14] = 0.85            # obstacle, left of centre
two[PADDING + 0:PADDING + 6] = 5.0              # deeper region beyond it
gaps = find_gaps(two, ANGLES, MINR, HALF, REAL_BINS)
wp_bearing = math.radians(-20.0)                # waypoint to the right
steer, gap = steer_through_gaps(gaps, ANGLES, wp_bearing)
deepest = max(g[2] for g in gaps)
check("steers at the waypoint, not the deepest gap",
      steer is not None and abs(steer - wp_bearing) < 1e-9,
      f"{len(gaps)} gaps, steer={deg(steer):+.1f}deg, chosen depth={gap[2]:.1f}m "
      f"vs deepest {deepest:.1f}m")

print("\n3. waypoint blocked: skirt via the gap needing least deviation")
blocked = scene(1.60)
blocked[PADDING + 30:PADDING + 44] = 0.85       # obstacle straight ahead
gaps = find_gaps(blocked, ANGLES, MINR, HALF, REAL_BINS)
steer, gap = steer_through_gaps(gaps, ANGLES, 0.0)   # waypoint dead ahead
lo, hi = sorted((ANGLES[gap[0]], ANGLES[gap[1]]))
check("steers into the open gap", steer is not None and lo <= steer <= hi,
      f"steer={deg(steer):+.1f}deg into [{deg(lo):+.0f},{deg(hi):+.0f}]deg")
check("aims at the gap edge nearest the waypoint",
      abs(steer - max(lo, min(hi, 0.0))) < 1e-9,
      f"deviation {abs(deg(steer)):.1f}deg")

print("\n4. speed follows gap depth and steering effort")
far = drive(scene(4.0), (3.0, 0.0)).get("cmd_vel")
shallow = drive(scene(0.95), (3.0, 0.0)).get("cmd_vel")
check("deep gap ahead -> full speed", far[0] > 0.45, f"v={far[0]:.2f}")
check("shallow gap -> slower", 0.0 < shallow[0] < far[0], f"v={shallow[0]:.2f}")

print("\n5. nothing passable: hold rather than invent a direction")
bb = drive(scene(0.30), (3.0, 0.0))
check("zero proposal", bb.get("cmd_proposal") == (0.0, 0.0))

print("\n6. a command that would hit is never published")
tree = load_tree(mission_xml())
bb = Blackboard()
c = ctx(); c.tick_id = 1
tree.tick(snapshot(scene(0.30)), bb, c)
check("trailing Brake publishes the stop", bb.get("cmd_vel") == (0.0, 0.0),
      "Navigate aborted at the arc gate")

rclpy.shutdown()
check.finish("follow-the-gap steering")
