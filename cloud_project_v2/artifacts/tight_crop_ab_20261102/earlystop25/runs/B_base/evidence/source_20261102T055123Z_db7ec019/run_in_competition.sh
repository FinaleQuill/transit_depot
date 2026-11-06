#!/usr/bin/env bash
# Run Python in the pinned organizer image on the current 2080Ti host.
# All arguments are Python arguments. With no arguments, only usage is shown.
set -euo pipefail

if [[ $# -eq 0 ]]; then
  printf 'Usage: %s PYTHON_ARGUMENTS...\n' "$0"
  printf 'Example: %s --version\n' "$0"
  exit 0
fi

COMPETITION_IMAGE='crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete@sha256:5e333ae965f02b4c80e4a5c89a9d33cae9ba54ffaea2f73217ffcd4b4a7fabc6'
COMPETITION_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
COMPETITION_WORK_ROOT='/home/featurize/work'
COMPETITION_DATA_ROOT='/home/featurize/data'
COMPETITION_ADDONS_ROOT='/tmp/dual768_competition_addons'

if [[ ! -d "$COMPETITION_ADDONS_ROOT/timm-1.0.30.dist-info" ]]; then
  printf 'Missing pinned timm addons at %s; prepare the documented isolated target first.\n' "$COMPETITION_ADDONS_ROOT" >&2
  exit 2
fi
# The target may add timm and its missing dependencies; core image packages
# must continue to resolve from the organizer image.
for COMPETITION_CORE_MODULE in torch torchvision numpy PIL scipy sklearn nvidia triton; do
  if [[ -e "$COMPETITION_ADDONS_ROOT/$COMPETITION_CORE_MODULE" || -e "$COMPETITION_ADDONS_ROOT/$COMPETITION_CORE_MODULE.py" ]]; then
    printf 'The addons target shadows an organizer core module: %s\n' "$COMPETITION_CORE_MODULE" >&2
    exit 2
  fi
done

# This host has a stale NVIDIA-runtime entry but no runtime executable.
# Explicit devices and read-only driver files avoid daemon/driver changes.
# These are the already-installed driver 610.57.04 files, not CUDA wheels.
exec docker run --rm --pull=never --runtime=runc --network none -i \
  --user "$(id -u):$(id -g)" \
  --device /dev/nvidia0 --device /dev/nvidiactl \
  --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  --volume /lib/x86_64-linux-gnu/libcuda.so.1:/usr/local/nvidia/lib64/libcuda.so.1:ro \
  --volume /lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/local/nvidia/lib64/libnvidia-ml.so.1:ro \
  --volume /lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1:/usr/local/nvidia/lib64/libnvidia-ptxjitcompiler.so.1:ro \
  --volume /lib/x86_64-linux-gnu/libnvidia-gpucomp.so.610.57.04:/usr/local/nvidia/lib64/libnvidia-gpucomp.so.610.57.04:ro \
  --volume /usr/bin/nvidia-smi:/usr/local/nvidia/bin/nvidia-smi:ro \
  --volume "$COMPETITION_WORK_ROOT:$COMPETITION_WORK_ROOT" \
  --volume "$COMPETITION_DATA_ROOT:$COMPETITION_DATA_ROOT:ro" \
  --volume "$COMPETITION_ADDONS_ROOT:/opt/dual768_addons:ro" \
  --workdir "$COMPETITION_REPO_ROOT" \
  --env PYTHONPATH=/opt/dual768_addons \
  --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONHASHSEED=43 --env HF_HUB_OFFLINE=1 \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 --env NUMEXPR_NUM_THREADS=1 \
  --env TZ=Asia/Hong_Kong \
  --entrypoint python "$COMPETITION_IMAGE" -B "$@"
