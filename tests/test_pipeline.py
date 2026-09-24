"""The Navigate pipeline: plan, propose, verify, execute, against one snapshot."""
import rclpy
from cognav_bt_behaviors.base import BehaviorConfig, BehaviorContext
from cognav_bt_behaviors.blackboard import Blackboard
from cognav_bt_behaviors.planning import RRTExploreWaypoint
from cognav_bt_behaviors.safety import IsCommandSafe
from cognav_bt_behaviors.status import Status
from cognav_bt_runner.xml_tree import load_tree

from _harness import Checker, leaf_names, mission_xml, scene, snapshot

check = Checker()
rclpy.init()
node = rclpy.create_node("pipeline")
CFG = BehaviorConfig(robot_radius=0.18, safety_margin=0.50, warning_margin=0.40,
                     maneuver_clearance=0.10, max_v=0.5)


def ctx():
    return BehaviorContext(node, None, CFG, enable_cmd_vel=False)


def run(ranges, ticks=1, bb=None):
    tree = load_tree(mission_xml())
    bb = bb or Blackboard()
    c = ctx()
    for t in range(1, ticks + 1):
        c.tick_id = t
        tree.tick(snapshot(ranges, stamp=float(t)), bb, c)
    return bb


print("\n1. leaf order matches the pipeline, and the contract holds")
tree = load_tree(mission_xml())
nav = [n for n in tree.children if n.name == "Navigate"][0]
names = leaf_names(nav)
check("plan, propose, verify, execute",
      names.index("RRTExploreWaypoint") < names.index("FollowTheGap")
      < names.index("IsCommandSafe") < names.index("ExecuteCommand"),
      " -> ".join(names))

print("\n2. the planner always succeeds; the gate fails on an unsafe or stale frame")
sw = RRTExploreWaypoint(ports={"seed": 1})
bb = Blackboard()
walled = scene(0.30)                              # nothing feasible anywhere
check("planner succeeds with an empty path",
      sw.tick(snapshot(walled), bb, ctx()) == Status.SUCCESS
      and bb.get("reference_path") == [])
safety = IsCommandSafe()
bb.set("cmd_proposal", (0.4, 0.0))
check("safety FAILS when the command would hit, aborting the sequence",
      safety.tick(snapshot(walled), bb, ctx()) == Status.FAILURE)
check("safety FAILS on a stale frame",
      safety.tick(snapshot(walled, fresh=False), bb, ctx()) == Status.FAILURE)

print("\n3. safety verifies THIS tick's proposal, not the previous one")
check("safety verifies the command about to run, freshly",
      "cmd_proposal" in IsCommandSafe.READS and not IsCommandSafe.READS_CARRIED)

print("\n4. the planner does not depend on safety's output")
check("planner reads no safety key",
      not ({"blocked_bins", "brake_now"} & set(RRTExploreWaypoint.READS)),
      f"reads={sorted(RRTExploreWaypoint.READS) or '-'}")

rclpy.shutdown()
check.finish("Navigate pipeline")
