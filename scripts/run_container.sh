#!/usr/bin/env bash
#
# Start the container without launching the stack, for calibration, navmesh
# export or inspection.
#
#   sudo ./scripts/run_container.sh          # interactive shell
#   sudo ./scripts/run_container.sh -d       # detached, for calibrate_depth_scale.sh
#
# Calibration starts its own habitat-sim and DA2 instances, so run it in this
# container rather than next to a running stack.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"

DETACH=()
INTERACTIVE=(-it)
CMD=(bash)
if [[ "${1:-}" == "-d" || "${1:-}" == "--detach" ]]; then
    shift
    DETACH=(-d)
    INTERACTIVE=()
    # Keeps a detached container alive for docker exec.
    CMD=(sleep infinity)
fi
[[ $# -gt 0 ]] && CMD=("$@")

require_image
require_data

if docker ps --format '{{.Names}}' | grep -Fxq "${COGNAV_CONTAINER_NAME}"; then
    echo "already running: ${COGNAV_CONTAINER_NAME}"
    echo "  shell into it:  docker exec -it ${COGNAV_CONTAINER_NAME} bash"
    echo "  stop it:        docker rm -f ${COGNAV_CONTAINER_NAME}"
    exit 0
fi

mapfile -t GUI_ARGS < <(docker_gui_args)
mapfile -t MOUNTS < <(docker_mount_args)
mkdir -p "${PLANNER_DIR}/navmesh"

echo "image      ${COGNAV_IMAGE}"
echo "container  ${COGNAV_CONTAINER_NAME}"
echo "scenes     ${COGNAV_HABITAT_DATA}"
echo "logs       ${COGNAV_LOG_DIR}"
echo

docker run --rm "${DETACH[@]}" "${INTERACTIVE[@]}" --gpus all --network host \
    --name "${COGNAV_CONTAINER_NAME}" \
    "${GUI_ARGS[@]}" "${MOUNTS[@]}" \
    -v "${PLANNER_DIR}/episodes:/opt/ws/episodes:ro" \
    -v "${PLANNER_DIR}/navmesh:/opt/ws/navmesh" \
    -e COGNAV_PLATFORM="${COGNAV_PLATFORM:-habitat}" \
    "${COGNAV_IMAGE}" \
    "${CMD[@]}"

if [[ ${#DETACH[@]} -gt 0 ]]; then
    echo
    echo "running detached. Next:"
    echo "  ./scripts/calibrate_depth_scale.sh <scene>"
    echo "  docker rm -f ${COGNAV_CONTAINER_NAME}    when done"
fi
