#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1
# V1 Base 768 strict-OOF teachers. Default is a command preview; --execute starts training.
# Set PYTHON_BIN to an environment with the training requirements already installed.
PYTHON_BIN="${PYTHON_BIN:-/home/featurize/.conda/envs/cxr-kd-v1/bin/python}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/featurize/data/cxr_first}"
RUN_NAME="${RUN_NAME:-021_convnext_v1_base768_teacher_cv5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
KD_RUN_COMMAND=("$PYTHON_BIN" -u run_teachers.py \
  --run-dir "../runs/$RUN_NAME" --image-root "$IMAGE_ROOT" \
  --batch-size "$BATCH_SIZE" --effective-batch-size "$EFFECTIVE_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" --prefetch-factor "$PREFETCH_FACTOR" "$@")
for KD_ARGUMENT in "$@"; do
  if [[ "$KD_ARGUMENT" == "--execute" ]]; then
    exec "$PYTHON_BIN" -u training_evidence_monitor.py run \
      --evidence-dir "../runs/$RUN_NAME/evidence" -- "${KD_RUN_COMMAND[@]}"
  fi
done
exec "${KD_RUN_COMMAND[@]}"
