#!/usr/bin/env bash
#
# Run one episode in the foreground.
#
#   sudo ./scripts/run_episode.sh --arm tree scene:=hm3d_large
#   sudo ./scripts/run_episode.sh --arm mesh --episode 0 scene:=Bowlus
#   ./scripts/run_episode.sh --list-arms
#
# Arguments after the flags are passed to `ros2 launch` unchanged. A scene
# named after a manifest in episodes/ is resolved to its path and depth_scale.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"

ARM="tree"
EPISODE=""
MANIFEST=""
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arm)        ARM="$2"; shift 2 ;;
        --episode)    EPISODE="$2"; shift 2 ;;
        --manifest)   MANIFEST="$2"; shift 2 ;;
        --list-arms)
            echo "  tree           full mission: Retry and Escape"
            echo "  no_retry       Escape only"
            echo "  no_escape      Retry only"
            echo "  navigate_only  neither"
            echo "  mesh           full mission, free space from the scene navmesh"
            exit 0 ;;
        *) PASSTHROUGH+=("$1"); shift ;;
    esac
done

source "${HERE}/arm_dispatch.sh"
require_image
require_data

RESOLVED=()
SCENE_ALIAS=""
for arg in "${PASSTHROUGH[@]:-}"; do
    [[ -z "${arg}" ]] && continue
    if [[ "${arg}" == scene:=* ]]; then
        alias_name="${arg#scene:=}"
        man="${PLANNER_DIR}/episodes/${alias_name}.json"
        if [[ -f "${man}" ]]; then
            rel="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['scene_rel'])" "${man}")"
            arg="scene:=${CONTAINER_HABITAT_DATA}/${rel}"
            ds="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('depth_scale') or '')" "${man}")"
            [[ -n "${ds}" ]] && RESOLVED+=("depth_scale:=${ds}")
            SCENE_ALIAS="${alias_name}"
        fi
    fi
    RESOLVED+=("${arg}")
done

LAUNCH_ARGS=("${ARM_ARGS[@]}" "${RESOLVED[@]}")

# The mesh arm needs the scene's navmesh raster and the episode start pose.
MESH_MOUNT=()
if [[ "${ARM}" == "mesh" ]]; then
    if [[ -z "${MANIFEST}" && -n "${SCENE_ALIAS}" ]]; then
        MANIFEST="${PLANNER_DIR}/episodes/${SCENE_ALIAS}.json"
    fi
    [[ -f "${MANIFEST}" ]] || {
        echo "!! --arm mesh needs a manifest; pass scene:=<alias> or --manifest <file>" >&2
        exit 1; }
    alias_name="$(basename "${MANIFEST}" .json)"
    mesh_npz="${PLANNER_DIR}/navmesh/${alias_name}.npz"
    [[ -f "${mesh_npz}" ]] || {
        echo "!! no navmesh raster: ${mesh_npz}" >&2
        echo "   Export it with scripts/export_navmesh.py (see README)." >&2
        exit 1; }
    start_hab="$(EPISODE="${EPISODE:-0}" python3 - "${MANIFEST}" <<'PY'
import json, os, sys
m = json.load(open(sys.argv[1]))
e = m["episodes"][int(os.environ["EPISODE"])]
print(",".join(str(v) for v in e["start_hab"]))
PY
)"
    LAUNCH_ARGS+=("mesh_path:=/opt/ws/navmesh/${alias_name}.npz"
                  "start_hab:=${start_hab}")
    MESH_MOUNT=(-v "${PLANNER_DIR}/navmesh:/opt/ws/navmesh:ro")
fi

mapfile -t GUI_ARGS < <(docker_gui_args)
mapfile -t MOUNTS < <(docker_mount_args)

echo "arm=${ARM}  package=${LAUNCH_PKG}  launch=${LAUNCH_FILE}"
printf '  %s\n' "${LAUNCH_ARGS[@]}"
echo

docker run --rm -it --gpus all \
    --name "${COGNAV_CONTAINER_NAME}" \
    "${GUI_ARGS[@]}" "${MOUNTS[@]}" "${MESH_MOUNT[@]}" \
    -e COGNAV_PLATFORM=habitat \
    "${COGNAV_IMAGE}" \
    bash -lc "ros2 launch ${LAUNCH_PKG} ${LAUNCH_FILE} $(printf '%q ' "${LAUNCH_ARGS[@]}")"
