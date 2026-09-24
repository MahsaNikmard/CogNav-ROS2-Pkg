#!/usr/bin/env bash
# Run rviz2 with the system ROS environment; optional first argument is a config.
set -euo pipefail

_cognav_restore_nounset=0
if [[ $- == *u* ]]; then
    _cognav_restore_nounset=1
    set +u
fi

# RViz needs the system Python and Qt, not the perception conda environment.
if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    # shellcheck source=/dev/null
    source /opt/conda/etc/profile.d/conda.sh
    while [[ -n "${CONDA_SHLVL:-}" && "${CONDA_SHLVL}" != "0" ]]; do
        conda deactivate || break
    done
fi

export PATH=/opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset CONDA_DEFAULT_ENV
unset CONDA_PREFIX
unset PYTHONHOME
unset PYTHONPATH

# shellcheck source=/dev/null
source /opt/ros/humble/setup.bash
if [[ -f /opt/ws/install/setup.bash ]]; then
    # shellcheck source=/dev/null
    source /opt/ws/install/setup.bash
fi

if (( _cognav_restore_nounset )); then
    set -u
fi

rviz_config="${COGNAV_RVIZ_CONFIG:-}"
if [[ $# -gt 0 && "${1}" != -* ]]; then
    rviz_config="${1}"
    shift
fi
config_args=()
if [[ -n "${rviz_config}" ]]; then
    if [[ ! -f "${rviz_config}" ]]; then
        echo "RViz config not found: ${rviz_config}" >&2
        exit 1
    fi
    config_args=(-d "${rviz_config}")
fi

rviz_ros_args=()
if [[ "${COGNAV_RVIZ_USE_SIM_TIME:-false}" =~ ^(1|true|yes|on)$ ]]; then
    rviz_ros_args=(--ros-args -p use_sim_time:=true)
fi

exec rviz2 "${config_args[@]}" "$@" "${rviz_ros_args[@]}"
