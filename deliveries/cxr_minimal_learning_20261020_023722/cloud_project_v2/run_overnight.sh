#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1
# Official weights were cached and verified offline during the 4090 preflight.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
PYTHON_BIN="${PYTHON_BIN:-/environment/miniconda3/envs/cxr4090/bin/python}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/featurize/data/cxr_first}"
RUN_NAME="${RUN_NAME:-020_convnextv2_base768_posbce_cv5_4090}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
# Measured 768 BF16/checkpoint + resident EMA/Adam/gradient probe:
# batch 2: 2.69 GiB allocated, 3.18 GiB reserved, 3.66 GiB sampled device use.
# Steady synthetic throughput: batch 2=15.97 img/s, batch 4=15.53 img/s.
# Batch 4 also passes; batch 2 is faster in the tested 1/2/4 range.
BATCH_SIZE="${BATCH_SIZE:-2}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-32}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python does not exist: $PYTHON_BIN" >&2
  exit 1
fi
if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ && "$EFFECTIVE_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "Batch sizes must be positive integers." >&2
  exit 1
fi
if (( EFFECTIVE_BATCH_SIZE % BATCH_SIZE != 0 )); then
  echo "BATCH_SIZE must divide EFFECTIVE_BATCH_SIZE." >&2
  exit 1
fi
ACCUMULATION_STEPS=$(( EFFECTIVE_BATCH_SIZE / BATCH_SIZE ))
mkdir -p ../logs
LOG_FILE="../logs/${RUN_NAME}_$(TZ=Asia/Hong_Kong date +%Y%m%d_%H%M%S).log"
echo "Pipeline log: $LOG_FILE"
echo "768x768; batch=$BATCH_SIZE; accumulation_steps=$ACCUMULATION_STEPS; effective_batch=$EFFECTIVE_BATCH_SIZE; workers=$NUM_WORKERS; prefetch=$PREFETCH_FACTOR"
"$PYTHON_BIN" -u run_all_folds.py --run-name "$RUN_NAME" --image-root "$IMAGE_ROOT" \
  --num-workers "$NUM_WORKERS" --prefetch-factor "$PREFETCH_FACTOR" \
  --batch-size "$BATCH_SIZE" --effective-batch-size "$EFFECTIVE_BATCH_SIZE" \
  "$@" 2>&1 | tee -a "$LOG_FILE"
