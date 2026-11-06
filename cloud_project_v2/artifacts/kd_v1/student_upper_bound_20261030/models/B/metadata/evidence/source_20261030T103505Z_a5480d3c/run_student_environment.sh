#!/usr/bin/env bash
# Isolated training dependencies; the pinned inference image remains unchanged.
# Arguments are passed to Python. This wrapper never starts a job by itself.
set -euo pipefail

if [[ $# -eq 0 ]]; then
  printf 'Usage: %s PYTHON_ARGUMENTS...\n' "$0"
  exit 0
fi

STUDENT_IMAGE='crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete@sha256:5e333ae965f02b4c80e4a5c89a9d33cae9ba54ffaea2f73217ffcd4b4a7fabc6'
STUDENT_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
STUDENT_WORK_ROOT="$(dirname -- "$STUDENT_REPO_ROOT")"
STUDENT_DATA_ROOT='/home/featurize/data'
STUDENT_INFERENCE_ADDONS='/tmp/dual768_competition_addons'
STUDENT_TRAINING_ADDONS='/tmp/cloud_student_training_addons_cp312'
STUDENT_LOCAL_ROOT='/tmp/cloud_student_runs_20261030'
STUDENT_BASELINE="$STUDENT_WORK_ROOT/deliveries/038_dual768_optimized_project_20261030"

for STUDENT_PATH in "$STUDENT_DATA_ROOT" "$STUDENT_INFERENCE_ADDONS/timm-1.0.30.dist-info" \
                    "$STUDENT_TRAINING_ADDONS/albumentations-2.0.8.dist-info" "$STUDENT_BASELINE"; do
  if [[ ! -d "$STUDENT_PATH" ]]; then
    printf 'Restore the documented environment first; missing directory: %s\n' "$STUDENT_PATH" >&2
    exit 2
  fi
done
for STUDENT_CORE in torch torchvision numpy PIL nvidia triton; do
  for STUDENT_TARGET in "$STUDENT_TRAINING_ADDONS" "$STUDENT_INFERENCE_ADDONS"; do
    if [[ -e "$STUDENT_TARGET/$STUDENT_CORE" || -e "$STUDENT_TARGET/$STUDENT_CORE.py" ]]; then
      printf 'Addon target shadows an image core module: %s/%s\n' "$STUDENT_TARGET" "$STUDENT_CORE" >&2
      exit 2
    fi
  done
done

STUDENT_GPU_ARGS=()
if command -v nvidia-container-runtime >/dev/null 2>&1; then
  STUDENT_GPU_ARGS+=(--runtime=nvidia --gpus all)
else
  STUDENT_GPU_ARGS+=(--runtime=runc)
  for STUDENT_DEVICE in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
    [[ -c "$STUDENT_DEVICE" ]] || { printf 'Missing GPU device: %s\n' "$STUDENT_DEVICE" >&2; exit 2; }
    STUDENT_GPU_ARGS+=(--device "$STUDENT_DEVICE")
  done
  STUDENT_CUDA_LIBRARY="$(readlink -f /lib/x86_64-linux-gnu/libcuda.so.1)"
  STUDENT_DRIVER_VERSION="${STUDENT_CUDA_LIBRARY##*/libcuda.so.}"
  for STUDENT_LIBRARY in /lib/x86_64-linux-gnu/libcuda.so.1 \
                         /lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
                         /lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1 \
                         "/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.$STUDENT_DRIVER_VERSION"; do
    [[ -f "$STUDENT_LIBRARY" ]] || { printf 'Missing driver library: %s\n' "$STUDENT_LIBRARY" >&2; exit 2; }
    STUDENT_GPU_ARGS+=(--volume "$STUDENT_LIBRARY:/usr/local/nvidia/lib64/$(basename -- "$STUDENT_LIBRARY"):ro")
  done
fi

mkdir -p "$STUDENT_LOCAL_ROOT"
exec docker run --rm --pull=never --network none --shm-size=4g -i \
  "${STUDENT_GPU_ARGS[@]}" \
  --user "$(id -u):$(id -g)" \
  --volume "$STUDENT_WORK_ROOT:$STUDENT_WORK_ROOT" \
  --volume "$STUDENT_REPO_ROOT/data:$STUDENT_REPO_ROOT/data:ro" \
  --volume "$STUDENT_WORK_ROOT/runs:$STUDENT_WORK_ROOT/runs:ro" \
  --volume "$STUDENT_WORK_ROOT/deliveries:$STUDENT_WORK_ROOT/deliveries:ro" \
  --volume "$STUDENT_BASELINE:$STUDENT_BASELINE:ro" \
  --volume "$STUDENT_DATA_ROOT:$STUDENT_DATA_ROOT:ro" \
  --volume "$STUDENT_LOCAL_ROOT:$STUDENT_LOCAL_ROOT" \
  --volume "$STUDENT_INFERENCE_ADDONS:/opt/dual768_addons:ro" \
  --volume "$STUDENT_TRAINING_ADDONS:/opt/student_training_addons:ro" \
  --workdir "$STUDENT_REPO_ROOT" \
  --env PYTHONPATH=/opt/student_training_addons:/opt/dual768_addons \
  --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONHASHSEED=44 --env HF_HUB_OFFLINE=1 \
  --env NO_ALBUMENTATIONS_UPDATE=1 --env TZ=Asia/Hong_Kong \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 --env NUMEXPR_NUM_THREADS=1 \
  --entrypoint python "$STUDENT_IMAGE" -B "$@"
