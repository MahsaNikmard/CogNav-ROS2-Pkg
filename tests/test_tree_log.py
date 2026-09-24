"""The tick log header and tree body render in every runner state."""
import rclpy

from cognav_bt_behaviors.base import BehaviorConfig, BehaviorContext
from cognav_bt_behaviors.blackboard import Blackboard
from cognav_bt_runner.tree_log import build_header, render_tick
from cognav_bt_runner.xml_tree import load_tree

from _check import Checker
from _harness import mission_xml, scene, snapshot


def main():
    c = Checker()
    rclpy.init()
    node = rclpy.create_node("tree_log_test")
    cfg = BehaviorConfig(robot_radius=0.18, safety_margin=0.50, warning_margin=0.40,
                         maneuver_clearance=0.10, max_v=0.5)

    def run(ranges, ticks=3):
        tree = load_tree(mission_xml())
        bb = Blackboard()
        ctx = BehaviorContext(node, None, cfg, enable_cmd_vel=False)
        for t in range(1, ticks + 1):
            ctx.tick_id = t
            snap = snapshot(ranges, stamp=float(t))
            tree.tick(snap, bb, ctx)
        return tree, bb, snap

    print("\n[1] build_header renders in every state")
    for label, ranges in [("open", scene(3.0)), ("wedged", scene(0.32)),
                          ("corridor", scene(1.2))]:
        _, bb, snap = run(ranges)
        try:
            head = build_header(7, 2.8, snap, bb)
            c(f"{label:9} -> {head.strip().splitlines()[0][:58]}", bool(head))
        except Exception as exc:  # noqa: BLE001
            c(f"{label:9} header", False, f"{type(exc).__name__}: {exc}")

    print("\n[2] header with no snapshot at all")
    try:
        head = build_header(1, 0.0, None, Blackboard())
        c("renders the waiting line", "no snapshot" in head)
    except Exception as exc:  # noqa: BLE001
        c("renders the waiting line", False, f"{type(exc).__name__}: {exc}")

    print("\n[3] header with an empty blackboard")
    _, _, snap = run(scene(3.0))
    try:
        head = build_header(1, 0.0, snap, Blackboard())
        c("no key is assumed present", bool(head))
    except Exception as exc:  # noqa: BLE001
        c("no key is assumed present", False, f"{type(exc).__name__}: {exc}")

    print("\n[4] the tree body renders")
    tree, bb, snap = run(scene(0.32))
    body = render_tick(tree, 3, "── tick 3")
    for name in ("Priorities", "Navigate", "ProposeUntilSafe", "Escape",
                 "RRTExploreWaypoint", "IsCommandSafe", "HasMotion", "Recover"):
        c(f"{name} appears", name in body)
    c("statuses are drawn", "SUCCESS" in body and "FAILURE" in body)

    print("\n[5] the header carries what a run is debugged from")
    _, bb, snap = run(scene(1.2))
    head = build_header(12, 4.8, snap, bb)
    for field in ("tick 12", "age=", "fresh=", "brake=", "blocked=", "cmd=("):
        c(f"{field!r} present", field in head)

    # A mission can override `seed` per leaf; the logged seed must show it.
    print("\n[6] the effective seed, not the runner parameter")
    from cognav_bt_runner.bt_runner import _effective_seeds

    tree = load_tree(mission_xml())
    reported = _effective_seeds(tree, -1)
    c("the mission's port override is named", "RRTExploreWaypoint=7" in reported,
      reported)
    c("the runner parameter is still shown", reported.startswith("-1"))
    c("a tree with no overrides reports the parameter alone",
      _effective_seeds(load_tree(
          "<root main_tree_to_execute='T'><BehaviorTree ID='T'>"
          "<Sequence><Brake/></Sequence></BehaviorTree></root>"), 5) == "5")

    c.finish("tick log rendering")


if __name__ == "__main__":
    main()
