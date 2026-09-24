#!/usr/bin/env bash
#
# Settings and Docker helpers, sourced by the other scripts.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLANNER_DIR="${COGNAV_PLANNER_DIR:-$(cd "${HERE}/.." && pwd)}"

COGNAV_IMAGE="${COGNAV_IMAGE:-cognav-planner}"
COGNAV_CONTAINER_NAME="${COGNAV_CONTAINER_NAME:-cognav}"

# Mount point of the scene data inside the container. Episode manifests store
# scene paths relative to it.
CONTAINER_HABITAT_DATA="/opt/ws/install/cognav_platform_adapter/share/cognav_platform_adapter/simulation/habitat_data"

# Scene data on the host; scripts/link_data.sh creates it.
COGNAV_HABITAT_DATA="${COGNAV_HABITAT_DATA:-${PLANNER_DIR}/habitat_data}"

COGNAV_OUT_DIR="${COGNAV_OUT_DIR:-${PLANNER_DIR}}"
COGNAV_LOG_DIR="${COGNAV_LOG_DIR:-${COGNAV_OUT_DIR}/logs}"

# Scenes and episodes per scene used by run_experiments.sh. Each manifest
# holds 20 episodes; the batch takes the first N.
COGNAV_SCENES=(Albertville Adrian hm3d_medium Bowlus hm3d_large)
COGNAV_EPISODES="${COGNAV_EPISODES:-10}"

require_image() {
    docker image inspect "${COGNAV_IMAGE}" >/dev/null 2>&1 && return 0
    echo "!! docker image not found: ${COGNAV_IMAGE}" >&2
    echo "   Build it once with:  sudo ./scripts/build_image.sh" >&2
    exit 1
}

require_data() {
    [[ -d "${COGNAV_HABITAT_DATA}" ]] && return 0
    echo "!! scene data not found: ${COGNAV_HABITAT_DATA}" >&2
    echo "   Run ./scripts/link_data.sh first." >&2
    exit 1
}

# X11 forwarding when a display is available, including the Xauthority cookie
# of the invoking user under sudo.
docker_gui_args() {
    local args=(-v /tmp/.X11-unix:/tmp/.X11-unix -e QT_X11_NO_MITSHM=1)
    [[ -n "${DISPLAY:-}" ]] && args+=(-e "DISPLAY=${DISPLAY}")
    local xauthority="${XAUTHORITY:-${HOME:-}/.Xauthority}"
    if [[ -n "${SUDO_USER:-}" && ! -r "${xauthority}" ]]; then
        xauthority="$(getent passwd "${SUDO_USER}" | cut -d: -f6)/.Xauthority"
    fi
    if [[ -r "${xauthority}" ]]; then
        args+=(-v "${xauthority}:/tmp/.docker.xauth:ro" -e XAUTHORITY=/tmp/.docker.xauth)
    fi
    printf '%s\n' "${args[@]}"
}

# Scene data read-only, and a writable log directory mounted at /opt/ws/logs.
# The log directory is ${1:-COGNAV_LOG_DIR} and is handed back to the sudo
# user so results can be scored without root.
docker_mount_args() {
    local log_dir="${1:-${COGNAV_LOG_DIR}}"
    mkdir -p "${log_dir}"
    if [[ -n "${SUDO_UID:-}" && -n "${SUDO_GID:-}" ]]; then
        chown "${SUDO_UID}:${SUDO_GID}" "${log_dir}" 2>/dev/null || true
    fi
    printf '%s\n' \
        -v "${COGNAV_HABITAT_DATA}:${CONTAINER_HABITAT_DATA}:ro" \
        -v "${log_dir}:/opt/ws/logs"
}

# Container path of a scene named by one of the manifests in episodes/.
# Prints nothing for any other name, which is then passed through unchanged.
scene_path_for() {
    local alias_name="$1"
    local manifest="${PLANNER_DIR}/episodes/${alias_name}.json"
    [[ -f "${manifest}" ]] || return 0
    local rel
    rel="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['scene_rel'])" \
           "${manifest}" 2>/dev/null)" || return 0
    [[ -n "${rel}" ]] && printf '%s\n' "${CONTAINER_HABITAT_DATA}/${rel}"
}
