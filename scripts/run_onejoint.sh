#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <beauty|instruments|yelp> [gpu-index]" >&2
  exit 2
fi

DATASET="$1"
GPU_INDEX="${2:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${ONEJOINT_DATA_ROOT:-${ROOT}/dataset}"
OUTPUT_ROOT="${ONEJOINT_OUTPUT_ROOT:-${ROOT}/outputs}"
CONFIG="${ONEJOINT_CONFIG:-${ROOT}/config/onejoint.yaml}"
PYTHON_BIN="${ONEJOINT_PYTHON:-$(command -v python)}"

case "${DATASET}" in
  beauty)
    EMBEDDING="Beauty.emb-llama.npy"
    EPOCHS=120
    EARLY_STOP=15
    BEAMS=20
    EVAL_BATCH=32
    ;;
  instruments)
    EMBEDDING="Instruments.emb-llama.npy"
    EPOCHS=120
    EARLY_STOP=15
    BEAMS=20
    EVAL_BATCH=32
    ;;
  yelp)
    EMBEDDING="Yelp.emb-llama.npy"
    EPOCHS=200
    EARLY_STOP=20
    BEAMS=80
    EVAL_BATCH=8
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${DATA_ROOT}/${DATASET}/${EMBEDDING}" ]]; then
  echo "Missing semantic embedding: ${DATA_ROOT}/${DATASET}/${EMBEDDING}" >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing config: ${CONFIG}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "${PYTHON_BIN}" train_onejoint.py \
  --config="${CONFIG}" \
  --dataset="${DATASET}" \
  --data_path="${DATA_ROOT}" \
  --semantic_emb_path="${EMBEDDING}" \
  --log_dir="${OUTPUT_ROOT}/logs/${DATASET}" \
  --checkpoint_root="${OUTPUT_ROOT}/checkpoints" \
  --training_mode=onejoint \
  --rqvae_path=None \
  --epochs="${EPOCHS}" \
  --early_stop="${EARLY_STOP}" \
  --eval_batch_size="${EVAL_BATCH}" \
  --num_beams="${BEAMS}" \
  --evaluate_test_each_epoch=false \
  --evaluate_test_at_end=true
