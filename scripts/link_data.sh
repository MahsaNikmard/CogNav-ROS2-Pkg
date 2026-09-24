#!/usr/bin/env bash
#
# Make the Habitat scene data available at habitat_data/ in the checkout.
#
#   COGNAV_HABITAT_SRC=/path/to/habitat_data ./scripts/link_data.sh          # symlink
#   COGNAV_HABITAT_SRC=/path/to/habitat_data ./scripts/link_data.sh --copy   # copy
#
# The source directory holds the scene_datasets/ and versioned_data/ trees the
# episode manifests refer to.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLANNER_DIR="$(cd "${HERE}/.." && pwd)"

SRC="${COGNAV_HABITAT_SRC:-}"
DEST="${PLANNER_DIR}/habitat_data"
MODE="link"
[[ "${1:-}" == "--copy" ]] && MODE="copy"

if [[ -z "${SRC}" || ! -d "${SRC}" ]]; then
    echo "!! set COGNAV_HABITAT_SRC to the directory holding the Habitat scenes" >&2
    exit 1
fi

if [[ -e "${DEST}" ]]; then
    echo "${DEST} already exists, nothing to do"
    exit 0
fi

if [[ "${MODE}" == "copy" ]]; then
    rsync -a "${SRC}/" "${DEST}/"
else
    ln -s "${SRC}" "${DEST}"
    echo "symlinked ${DEST} -> ${SRC}"
fi
du -sh "${DEST}/" 2>/dev/null | sed 's/^/  /'
