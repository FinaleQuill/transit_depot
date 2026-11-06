#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1
KD_MIXED_PYTHON="${PYTHON_BIN:-/home/featurize/.conda/envs/cxr-kd-v1/bin/python}"
exec "$KD_MIXED_PYTHON" -u run_mixed_teachers.py "$@"
