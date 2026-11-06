#!/usr/bin/env bash
# Isolated organizer-image runtime for new tight-crop A/B runs.
set -euo pipefail

if [[ $# -eq 0 ]]; then
  printf 'Usage: %s PYTHON_ARGUMENTS...\n' "$0"
  exit 0
fi

IMAGE='crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete@sha256:5e333ae965f02b4c80e4a5c89a9d33cae9ba54ffaea2f73217ffcd4b4a7fabc6'
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
WORK="$(dirname -- "$REPO")"
DATA='/home/featurize/data'
INFERENCE_ADDONS='/tmp/dual768_competition_addons'
TRAINING_ADDONS='/tmp/cloud_student_training_addons_cp312'
LOCAL='/tmp/cloud_student_runs_20261030'

for required in "$DATA/cxr_first/train" "$DATA/cxr_first/valid" \
                "$INFERENCE_ADDONS/timm-1.0.30.dist-info" \
                "$TRAINING_ADDONS/albumentations-2.0.8.dist-info"; do
  [[ -d "$required" ]] || { printf 'Missing runtime input: %s\n' "$required" >&2; exit 2; }
done
for core in torch torchvision numpy PIL nvidia triton; do
  for target in "$INFERENCE_ADDONS" "$TRAINING_ADDONS"; do
    [[ ! -e "$target/$core" && ! -e "$target/$core.py" ]] || {
      printf 'Addon shadows image core: %s/%s\n' "$target" "$core" >&2; exit 2;
    }
  done
done

gpu_args=()
if command -v nvidia-container-runtime >/dev/null 2>&1; then
  gpu_args+=(--runtime=nvidia --gpus all)
else
  gpu_args+=(--runtime=runc)
  for device in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
    [[ -c "$device" ]] || { printf 'Missing GPU device: %s\n' "$device" >&2; exit 2; }
    gpu_args+=(--device "$device")
  done
  driver_version="$(readlink -f /lib/x86_64-linux-gnu/libcuda.so.1)"
  driver_version="${driver_version##*libcuda.so.}"
  for library in /lib/x86_64-linux-gnu/libcuda.so.1 \
                 /lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
                 /lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1 \
                 "/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.$driver_version"; do
    [[ -f "$library" ]] || { printf 'Missing driver library: %s\n' "$library" >&2; exit 2; }
    gpu_args+=(--volume "$library:/usr/local/nvidia/lib64/$(basename -- "$library"):ro")
  done
fi

mkdir -p "$LOCAL"
exec docker run --rm --pull=never --network none --shm-size=4g -i \
  "${gpu_args[@]}" --user "$(id -u):$(id -g)" \
  --volume "$WORK:$WORK:ro" \
  --volume "$DATA:$DATA:ro" \
  --volume "$LOCAL:$LOCAL" \
  --volume "$INFERENCE_ADDONS:/opt/dual768_addons:ro" \
  --volume "$TRAINING_ADDONS:/opt/student_training_addons:ro" \
  --workdir "$REPO" \
  --env PYTHONPATH=/opt/student_training_addons:/opt/dual768_addons \
  --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONHASHSEED=44 --env HF_HUB_OFFLINE=1 \
  --env NO_ALBUMENTATIONS_UPDATE=1 --env TZ=Asia/Hong_Kong \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 --env NUMEXPR_NUM_THREADS=1 \
  --entrypoint python "$IMAGE" -B "$@"
