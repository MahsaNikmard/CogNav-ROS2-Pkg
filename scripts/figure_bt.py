#!/usr/bin/env python3
"""Render a mission file as a Graphviz figure.

    python3 scripts/figure_bt.py \
        --mission src/cognav_mission/mission/random_explore.xml \
        --out figures/cognav_bt_tree

Needs Graphviz. With the workspace sourced, the mission is also loaded through
cognav_bt_runner.xml_tree.load_tree, so an invalid mission fails here.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

#: Shape and fill per node kind.
STYLE = {
    "fallback": dict(shape="octagon", fillcolor="#cfe8f3", label_suffix="\\n[Selector]"),
    "sequence": dict(shape="box", fillcolor="#fbe3c8", label_suffix="\\n[Sequence]"),
    "retry": dict(shape="box", style="rounded,filled", fillcolor="#e6dcf0",
                  label_suffix="\\n[Retry {n}]"),
    "leaf": dict(shape="ellipse", fillcolor="#eeeeee", label_suffix=""),
}

_CONTROL = {
    "Fallback": "fallback", "Selector": "fallback", "ReactiveFallback": "fallback",
    "Sequence": "sequence", "ReactiveSequence": "sequence",
    "Retry": "retry", "RetryUntilSuccessful": "retry",
}


def _kind(tag: str) -> str:
    return _CONTROL.get(tag, "leaf")


def emit(element, lines, counter, parent=None):
    tag = element.tag.split("}", 1)[-1]
    kind = _kind(tag)
    node_id = f"n{counter[0]}"
    counter[0] += 1

    name = element.attrib.get("name") or tag
    style = STYLE[kind]
    suffix = style["label_suffix"].format(n=element.attrib.get("num_attempts", "3"))
    label = f"{name}{suffix}"

    attrs = [f'label="{label}"', f'shape={style["shape"]}',
             f'style="{style.get("style", "filled")}"',
             f'fillcolor="{style["fillcolor"]}"', 'fontname="Helvetica"',
             "fontsize=11"]
    lines.append(f'  {node_id} [{", ".join(attrs)}];')
    if parent is not None:
        lines.append(f"  {parent} -> {node_id};")

    for child in list(element):
        emit(child, lines, counter, node_id)
    return node_id


def build_dot(mission_path: str) -> str:
    root = ET.parse(mission_path).getroot()
    tree_id = root.attrib.get("main_tree_to_execute")
    behavior_tree = None
    for candidate in root.findall("BehaviorTree"):
        if tree_id is None or candidate.attrib.get("ID") == tree_id:
            behavior_tree = candidate
            break
    if behavior_tree is None:
        raise SystemExit(f"no matching BehaviorTree in {mission_path}")
    children = list(behavior_tree)
    if len(children) != 1:
        raise SystemExit("a BehaviorTree must hold exactly one root control node")

    # ordering=out keeps siblings in tick order, left to right.
    lines = ["digraph bt {", "  rankdir=TB;", "  ordering=out;",
             "  nodesep=0.35;", "  ranksep=0.45;",
             "  node [margin=0.08];",
             '  edge [arrowsize=0.7, color="#555555"];']
    emit(children[0], lines, [0])
    lines.append("}")
    return "\n".join(lines)


def validate(mission_path: str) -> str:
    """Load the mission as the runner does, when the workspace is sourced."""
    try:
        from cognav_bt_runner.xml_tree import load_tree
    except Exception:
        return "not validated (workspace not sourced)"
    try:
        with open(mission_path, encoding="utf-8") as fh:
            load_tree(fh.read())
        return "loads and passes contract validation"
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"mission does not load: {exc}") from exc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission",
                    default="src/cognav_mission/mission/random_explore.xml")
    ap.add_argument("--out", default="figures/cognav_bt_tree",
                    help="output stem; .dot and .pdf are written")
    ap.add_argument("--format", default="pdf")
    args = ap.parse_args(argv)

    print(f"  mission   {args.mission}")
    print(f"  {validate(args.mission)}")

    dot = build_dot(args.mission)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    dot_path = f"{args.out}.dot"
    with open(dot_path, "w", encoding="utf-8") as fh:
        fh.write(dot + "\n")
    print(f"  wrote     {dot_path}")

    out_path = f"{args.out}.{args.format}"
    try:
        subprocess.run(["dot", f"-T{args.format}", dot_path, "-o", out_path],
                       check=True)
    except FileNotFoundError:
        print("  graphviz 'dot' not found; the .dot file is written, render it "
              "elsewhere", file=sys.stderr)
        return 1
    print(f"  wrote     {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
