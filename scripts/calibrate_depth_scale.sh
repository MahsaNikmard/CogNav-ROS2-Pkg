#!/usr/bin/env bash
#
# Measure depth_scale for a scene inside the container started by
# `run_container.sh -d`.
#
#   ./scripts/calibrate_depth_scale.sh hm3d_large
#   ./scripts/calibrate_depth_scale.sh van_gogh_room 32
#   ./scripts/calibrate_depth_scale.sh --hypersim logs/hypersim/ai_002_001
#
# Habitat renders RGB and ground-truth depth in the habitat environment, and
# DA2 is evaluated in the perception environment. Samples are kept under
# logs/calibration/<scene>. The --hypersim form is a control on the tooling:
# DA2 is fine-tuned on Hypersim, so it should report a ratio near 1.0.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

hypersim_root=""
if [[ "${1:-}" == "--hypersim" ]]; then
    if [[ -z "${2:-}" ]]; then
        echo "--hypersim needs a scene directory, e.g. logs/hypersim/ai_002_001" >&2
        exit 1
    fi
    host_root="${2}"
    [[ "${host_root}" = /* ]] || host_root="${PLANNER_DIR}/${host_root}"
    if [[ ! -d "${host_root}" ]]; then
        echo "No such Hypersim scene directory: ${host_root}" >&2
        echo "Expected a tree containing images/scene_cam_XX_final_hdf5/." >&2
        exit 1
    fi
    # logs/ is mounted at /opt/ws/logs inside the container.
    case "${host_root}" in
        "${PLANNER_DIR}/logs/"*) hypersim_root="/opt/ws/logs/${host_root#"${PLANNER_DIR}/logs/"}" ;;
        *) echo "Keep the scene under ${PLANNER_DIR}/logs/ so it is visible in the container." >&2
           exit 1 ;;
    esac
    scene_alias="hypersim_$(basename "${host_root}")"
    samples="${3:-24}"
else
    scene_alias="${1:-hm3d_large}"
    samples="${2:-24}"
fi
out="/opt/ws/logs/calibration/${scene_alias}"

if ! docker ps --format '{{.Names}}' | grep -Fxq "${COGNAV_CONTAINER_NAME}"; then
    echo "Container is not running: ${COGNAV_CONTAINER_NAME}" >&2
    echo "Start it with: sudo ./scripts/run_container.sh -d" >&2
    exit 1
fi

# Scenes named after a manifest are resolved here; habitat aliases in the container.
scene_override="$(scene_path_for "${scene_alias}")"

echo "Calibrating scene     : ${scene_alias}"
[[ -n "${scene_override}" ]] && echo "Resolved from manifest: ${scene_override}"
echo "Samples               : ${samples}"
echo "Sample directory      : ${out}  (host: logs/calibration/${scene_alias})"

if [[ -n "${hypersim_root}" ]]; then
    echo "Mode                  : hypersim control (expect a ratio near 1.0)"
    echo "Hypersim scene        : ${hypersim_root}"
    docker exec -it "${COGNAV_CONTAINER_NAME}" bash -lc "
        set -euo pipefail
        source /opt/cognav/setup_env.sh
        perc=\$(ros2 pkg prefix cognav_perception)/share/cognav_perception

        conda run --no-capture-output -n perception python -m cognav_perception.calibration.import_samples \
            --source hypersim --root '${hypersim_root}' --out '${out}' --limit ${samples}

        conda run --no-capture-output -n perception python -m cognav_perception.calibration.estimate_scale \
            --samples '${out}' \
            --weights \"\$perc/model_weights/depth_anything_v2_metric_hypersim_vits.pth\"
    "
    exit 0
fi

docker exec -it "${COGNAV_CONTAINER_NAME}" bash -lc "
    set -euo pipefail
    source /opt/cognav/setup_env.sh
    share=\$(ros2 pkg prefix cognav_platform_adapter)/share/cognav_platform_adapter
    perc=\$(ros2 pkg prefix cognav_perception)/share/cognav_perception
    scene='${scene_override}'
    if [ -z \"\$scene\" ]; then
      scene=\$(/usr/bin/python3 - <<'PY'
import sys
sys.path.insert(0, '/opt/ws/install/cognav_platform_adapter/share/cognav_platform_adapter/launch')
from platform_adapter_launch import HABITAT_SCENE_ALIASES, _habitat_scene_path
from ament_index_python.packages import get_package_share_directory
print(_habitat_scene_path(get_package_share_directory('cognav_platform_adapter'), '${scene_alias}'))
PY
)
    fi
    if [ ! -f \"\$scene\" ]; then
      echo \"scene file not found: \$scene\" >&2
      echo \"'${scene_alias}' is neither a habitat alias nor one of episodes/*.json\" >&2
      exit 1
    fi
    echo \"Resolved scene        : \$scene\"

    conda run --no-capture-output -n habitat python -m cognav_perception.calibration.render_samples \
        --scene \"\$scene\" --out '${out}' --samples ${samples}

    conda run --no-capture-output -n perception python -m cognav_perception.calibration.estimate_scale \
        --samples '${out}' \
        --weights \"\$perc/model_weights/depth_anything_v2_metric_hypersim_vits.pth\"
"
