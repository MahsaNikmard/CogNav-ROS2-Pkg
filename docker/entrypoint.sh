#!/usr/bin/env bash
set -euo pipefail

source /opt/cognav/setup_env.sh

if [[ $# -eq 0 ]]; then
    set -- bash
fi

exec "$@"
