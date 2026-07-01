#!/bin/bash

set -euo pipefail

# Install the self-contained repo_overlay into a target StarVLA/VLA-JEPA repo.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OVERLAY_DIR="${SCRIPT_DIR}/repo_overlay"
TARGET_REPO="${TARGET_REPO:-${1:-}}"

if [[ -z "${TARGET_REPO}" ]]; then
    echo "Usage: TARGET_REPO=/path/to/repo bash install_into_repo.sh" >&2
    echo "   or: bash install_into_repo.sh /path/to/repo" >&2
    exit 2
fi

if [[ ! -d "${TARGET_REPO}" ]]; then
    echo "TARGET_REPO does not exist or is not a directory: ${TARGET_REPO}" >&2
    exit 2
fi

if [[ ! -d "${OVERLAY_DIR}" ]]; then
    echo "Cannot find repo_overlay directory: ${OVERLAY_DIR}" >&2
    exit 1
fi

stamp="$(date +%Y%m%d_%H%M%S)"

while IFS= read -r -d '' source_path; do
    relative_path="${source_path#${OVERLAY_DIR}/}"
    target_path="${TARGET_REPO}/${relative_path}"
    target_dir="$(dirname -- "${target_path}")"

    mkdir -p "${target_dir}"
    if [[ -e "${target_path}" ]]; then
        cp -a "${target_path}" "${target_path}.bak.${stamp}"
    fi
    cp -a "${source_path}" "${target_path}"
    echo "installed ${relative_path}"
done < <(find "${OVERLAY_DIR}" -type f -print0 | sort -z)

echo "Installed LIBERO-Plus fast evaluation files into ${TARGET_REPO}"
echo "Backups, if any, use suffix .bak.${stamp}"
