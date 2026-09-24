#!/usr/bin/env bash
#
# Open RViz inside the running container.
#
#   ./scripts/run_rviz.sh [config.rviz]
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

display_args=()
if [[ -n "${DISPLAY:-}" ]]; then
  display_args=(-e "DISPLAY=${DISPLAY}")
fi

xauthority_args=()
if docker exec "${COGNAV_CONTAINER_NAME}" test -r /tmp/.docker.xauth >/dev/null 2>&1; then
  xauthority_args=(-e XAUTHORITY=/tmp/.docker.xauth)
else
  echo "No Xauthority cookie in container '${COGNAV_CONTAINER_NAME}'; RViz may not reach the display." >&2
fi

container_platform=""
if container_platform="$(docker exec "${COGNAV_CONTAINER_NAME}" printenv COGNAV_PLATFORM 2>/dev/null)"; then
  container_platform="${container_platform,,}"
fi

rviz_use_sim_time="${COGNAV_RVIZ_USE_SIM_TIME:-}"
if [[ -z "${rviz_use_sim_time}" ]]; then
  if [[ "${container_platform}" == "habitat" ]]; then
    rviz_use_sim_time=true
  else
    rviz_use_sim_time=false
  fi
fi

docker exec -it \
  "${display_args[@]}" \
  "${xauthority_args[@]}" \
  -e QT_X11_NO_MITSHM=1 \
  -e "COGNAV_RVIZ_USE_SIM_TIME=${rviz_use_sim_time}" \
  "${COGNAV_CONTAINER_NAME}" \
  /opt/cognav/run_rviz.sh "$@"
