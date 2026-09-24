#!/usr/bin/env bash
# Run rqt_graph with the system ROS environment.
set -euo pipefail

_cognav_restore_nounset=0
if [[ $- == *u* ]]; then
    _cognav_restore_nounset=1
    set +u
fi

# rqt needs the system Python and Qt, not the perception conda environment.
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

exec ros2 run rqt_graph rqt_graph "$@"
