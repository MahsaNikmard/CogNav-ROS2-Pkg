#!/usr/bin/env bash
#
# Arm table, sourced by run_episode.sh and run_experiments.sh. Reads $ARM and
# sets LAUNCH_PKG, LAUNCH_FILE and ARM_ARGS.
#
#   arm             Retry   Escape   free-space test
#   tree            yes     yes      polar vector
#   no_retry        no      yes      polar vector
#   no_escape       yes     no       polar vector
#   navigate_only   no      no       polar vector
#   mesh            yes     yes      scene navmesh, same view and range
#
# The four tree arms differ only in the mission file.

MISSION_DIR="cognav_mission/mission"
LAUNCH_PKG="cognav_bt_runner"
LAUNCH_FILE="cognav_bt_launch.py"

case "${ARM:-}" in
    tree)          ARM_ARGS=("mission_xml_path:=${MISSION_DIR}/random_explore.xml") ;;
    no_retry)      ARM_ARGS=("mission_xml_path:=${MISSION_DIR}/no_retry.xml") ;;
    no_escape)     ARM_ARGS=("mission_xml_path:=${MISSION_DIR}/no_escape.xml") ;;
    navigate_only) ARM_ARGS=("mission_xml_path:=${MISSION_DIR}/navigate_only.xml") ;;
    mesh)          ARM_ARGS=("mission_xml_path:=${MISSION_DIR}/random_explore.xml"
                             "free_space_source:=mesh") ;;
    *)
        echo "unknown arm: '${ARM:-}'" >&2
        echo "valid arms: tree no_retry no_escape navigate_only mesh" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

COGNAV_ARMS=(tree no_retry no_escape navigate_only mesh)
