#!/bin/bash

set -euo pipefail

# This wrapper is distributed from examples/LIBERO-Plus/fast_eval_package.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
TARGET_REPO="${TARGET_REPO:-${DEFAULT_REPO_ROOT}}"
REPO_ROOT="$(cd -- "${TARGET_REPO}" && pwd)"
TARGET_SCRIPT="${REPO_ROOT}/examples/LIBERO-Plus/eval_libero_plus_sharded_batched.sh"

if [[ ! -f "${TARGET_SCRIPT}" ]]; then
    echo "Cannot find shared LIBERO-Plus fast evaluation script: ${TARGET_SCRIPT}" >&2
    echo "If this package is outside the repo, install it first:" >&2
    echo "  TARGET_REPO=/path/to/repo bash ${SCRIPT_DIR}/install_into_repo.sh" >&2
    exit 1
fi

cd "${REPO_ROOT}"
exec bash examples/LIBERO-Plus/eval_libero_plus_sharded_batched.sh "$@"
