#!/usr/bin/env bash
#
# The batch: every scene, every episode, every arm, run sequentially.
#
#   sudo ./scripts/run_experiments.sh
#   sudo ./scripts/run_experiments.sh --scenes "Adrian Bowlus" --episodes 3
#   ./scripts/run_experiments.sh --dry-run
#
# Every arm replays the same (scene, start pose, seed) triples, so results are
# paired per episode. An episode whose log already ends with `# cognav end` is
# skipped, so an interrupted batch resumes with the same command.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"

SCENES=("${COGNAV_SCENES[@]}")
EPISODES="${COGNAV_EPISODES}"
ARMS=(navigate_only no_retry no_escape tree mesh)
TIMEOUT="${COGNAV_TRIAL_TIMEOUT:-500.0}"
DRY_RUN=0
TAG="$(date +%Y%m%d_%H%M)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenes)   read -r -a SCENES <<< "$2"; shift 2 ;;
        --arms)     read -r -a ARMS <<< "$2"; shift 2 ;;
        --episodes) EPISODES="$2"; shift 2 ;;
        --timeout)  TIMEOUT="$2"; shift 2 ;;
        --tag)      TAG="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ ${DRY_RUN} -eq 0 ]]; then
    require_image
    require_data
fi

OUT="${COGNAV_OUT_DIR}/results/batch_${TAG}"
total=$(( ${#SCENES[@]} * EPISODES * ${#ARMS[@]} ))
echo "batch ${TAG}"
echo "  scenes   ${SCENES[*]}"
echo "  episodes ${EPISODES} per scene, the first ${EPISODES} of each manifest"
echo "  arms     ${ARMS[*]}"
echo "  budget   ${TIMEOUT} s per episode"
echo "  runs     ${total}"
echo "  out      ${OUT}"
echo

# Check every scene before the first container starts.
fail=0
for scene in "${SCENES[@]}"; do
    manifest="${PLANNER_DIR}/episodes/${scene}.json"
    [[ -f "${manifest}" ]] || { echo "!! no manifest: ${manifest}" >&2; fail=1; continue; }
    if ! python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('depth_scale') else 1)" "${manifest}"; then
        echo "!! ${scene}: manifest records no depth_scale" >&2
        echo "   ./scripts/calibrate_depth_scale.sh ${scene}" >&2
        fail=1
    fi
    for arm in "${ARMS[@]}"; do
        if [[ "${arm}" == "mesh" && ! -f "${PLANNER_DIR}/navmesh/${scene}.npz" ]]; then
            echo "!! ${scene}: no navmesh raster for the mesh arm (scripts/export_navmesh.py)" >&2
            fail=1
        fi
    done
done
[[ ${fail} -eq 0 ]] || { echo; echo "preflight failed, nothing was run" >&2; exit 1; }

[[ ${DRY_RUN} -eq 1 ]] || mkdir -p "${OUT}" "${PLANNER_DIR}/navmesh"
mapfile -t GUI_ARGS < <(docker_gui_args)

done_n=0 skipped=0 ran=0
for scene in "${SCENES[@]}"; do
    manifest="${PLANNER_DIR}/episodes/${scene}.json"
    for (( ep=0; ep<EPISODES; ep++ )); do
        read -r ep_id start_hab depth_scale seed scene_rel < <(EPISODE="${ep}" python3 - "${manifest}" <<'PY'
import json, os, sys
m = json.load(open(sys.argv[1]))
n = int(os.environ["EPISODE"])
if n >= len(m["episodes"]):
    sys.exit(f"manifest has {len(m['episodes'])} episodes, asked for index {n}")
e = m["episodes"][n]
print(e["id"], ",".join(str(v) for v in e["start_hab"]),
      m.get("depth_scale", ""), e.get("seed", 0), m["scene_rel"])
PY
)
        for arm in "${ARMS[@]}"; do
            done_n=$(( done_n + 1 ))
            # One directory per arm, one log per episode, named by manifest id.
            arm_dir="${OUT}/${arm}"
            log="${arm_dir}/${ep_id}.log"
            if [[ -f "${log}" ]] && grep -q '^# cognav end' "${log}"; then
                skipped=$(( skipped + 1 ))
                continue
            fi

            ARM="${arm}" source "${HERE}/arm_dispatch.sh"
            args=("${ARM_ARGS[@]}"
                  "scene:=${CONTAINER_HABITAT_DATA}/${scene_rel}"
                  "start_hab:=${start_hab}"
                  "depth_scale:=${depth_scale}"
                  "seed:=${seed}"
                  "enable_cmd_vel:=true"
                  "trial_timeout_sec:=${TIMEOUT}"
                  "tree_log_ranges:=full"
                  "tree_log_path:=/opt/ws/logs/${ep_id}.log")
            [[ "${arm}" == "mesh" ]] && args+=("mesh_path:=/opt/ws/navmesh/${scene}.npz")

            printf '[%3d/%3d] %-16s %-14s ' "${done_n}" "${total}" "${ep_id}" "${arm}"
            if [[ ${DRY_RUN} -eq 1 ]]; then
                echo "(dry run)"
                printf '             %s\n' "${args[@]}"
                continue
            fi

            mapfile -t MOUNTS < <(docker_mount_args "${arm_dir}")
            start=$(date +%s)
            # Hard deadline: the trial budget plus time to load and shut down.
            if timeout --signal=INT "$(python3 -c "print(int(float('${TIMEOUT}') + 180))")" \
                docker run --rm --gpus all \
                    --name "${COGNAV_CONTAINER_NAME}_batch" \
                    "${GUI_ARGS[@]}" "${MOUNTS[@]}" \
                    -v "${PLANNER_DIR}/navmesh:/opt/ws/navmesh:ro" \
                    -e COGNAV_PLATFORM=habitat \
                    "${COGNAV_IMAGE}" \
                    bash -lc "ros2 launch ${LAUNCH_PKG} ${LAUNCH_FILE} $(printf '%q ' "${args[@]}")" \
                    > "${log}.console" 2>&1
            then :; else
                echo -n "(exit $?) "
            fi
            elapsed=$(( $(date +%s) - start ))
            ran=$(( ran + 1 ))

            if grep -q '^# cognav end' "${log}" 2>/dev/null; then
                echo "ok ${elapsed}s  $(grep '^# cognav end' "${log}" | tail -1 | cut -c15-)"
            else
                echo "NO END MARKER after ${elapsed}s, see ${log}.console"
            fi
        done
    done
done

echo
echo "batch ${TAG}: ${ran} run, ${skipped} already complete, of ${total}"
echo "score with:  conda run -n habitat python scripts/score_batch.py --batch ${OUT}"
