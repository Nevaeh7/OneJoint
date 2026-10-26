#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [gpu-index]" >&2
  exit 2
fi

GPU_INDEX="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for dataset in beauty instruments yelp; do
  echo "[OneJoint] starting ${dataset}"
  "${SCRIPT_DIR}/run_onejoint.sh" "${dataset}" "${GPU_INDEX}"
done
