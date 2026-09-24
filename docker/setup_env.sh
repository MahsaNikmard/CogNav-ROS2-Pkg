#!/usr/bin/env bash
# Activate the perception environment and source ROS and the workspace.

_cognav_interactive=0
if [[ $- == *i* ]]; then
    _cognav_interactive=1
fi

if [[ -n "${COGNAV_ENV_READY:-}" && ${_cognav_interactive} -eq 0 ]]; then
    return 0 2>/dev/null || exit 0
fi

_cognav_restore_nounset=0
if [[ $- == *u* ]]; then
    _cognav_restore_nounset=1
    set +u
fi

export COGNAV_ENV_READY=1

# Name any supplemental group ids the runtime injected, to silence shell warnings.
if [[ -w /etc/group ]]; then
    while IFS= read -r gid; do
        [[ -z "${gid}" ]] && continue
        if ! getent group "${gid}" >/dev/null 2>&1; then
            printf 'cognav-gid-%s:x:%s:\n' "${gid}" "${gid}" >> /etc/group
        fi
    done < <(id -G 2>/dev/null | tr ' ' '\n')
fi

if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    source /opt/conda/etc/profile.d/conda.sh
    if [[ "${CONDA_DEFAULT_ENV:-}" != "perception" || ${_cognav_interactive} -eq 1 ]]; then
        conda activate perception
    fi
fi

if [[ -f /opt/ros/humble/setup.bash ]]; then
    source /opt/ros/humble/setup.bash
fi

if [[ -f /opt/ws/install/setup.bash ]]; then
    source /opt/ws/install/setup.bash
fi

if (( _cognav_restore_nounset )); then
    set -u
fi
