#!/usr/bin/env bash
#
# Build the Docker image.
#
#   sudo ./scripts/build_image.sh
#   sudo ./scripts/build_image.sh --no-cache
#
# The first build is slow, mostly the two conda environments. They sit in
# layers before the workspace copy, so source edits rebuild only the last ones.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"

cd "${PLANNER_DIR}"

if [[ ! -f src/cognav_perception/model_weights/depth_anything_v2_metric_hypersim_vits.pth ]]; then
    echo "note: DA2 checkpoint not in the checkout, the build will download it." >&2
    echo "      Pass --build-arg DOWNLOAD_DA2_WEIGHTS=0 to refuse instead." >&2
fi

docker build -t "${COGNAV_IMAGE}" -f docker/Dockerfile "$@" .

echo
echo "built ${COGNAV_IMAGE}"
docker image inspect "${COGNAV_IMAGE}" --format '  size {{.Size}} bytes' 2>/dev/null || true
echo
echo "next:"
echo "  ./scripts/link_data.sh                       scene meshes"
echo "  ./scripts/calibrate_depth_scale.sh <scene>   before any batch"
echo "  ./scripts/run_episode.sh --arm tree scene:=hm3d_large"
