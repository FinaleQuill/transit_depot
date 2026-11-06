#!/usr/bin/env bash
set -euo pipefail
candidate_root='/tmp/student_candidate_20261031'
candidate_repo='/home/featurize/work/cloud_project_v2'
candidate_image='crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete@sha256:5e333ae965f02b4c80e4a5c89a9d33cae9ba54ffaea2f73217ffcd4b4a7fabc6'
exec docker run --rm --pull=never --runtime=runc --network none -i --shm-size=4g \
 --user "$(id -u):$(id -g)" \
 --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
 --volume /lib/x86_64-linux-gnu/libcuda.so.1:/usr/local/nvidia/lib64/libcuda.so.1:ro \
 --volume /lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/local/nvidia/lib64/libnvidia-ml.so.1:ro \
 --volume /lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1:/usr/local/nvidia/lib64/libnvidia-ptxjitcompiler.so.1:ro \
 --volume /lib/x86_64-linux-gnu/libnvidia-gpucomp.so.610.57.04:/usr/local/nvidia/lib64/libnvidia-gpucomp.so.610.57.04:ro \
 --volume /usr/bin/nvidia-smi:/usr/local/nvidia/bin/nvidia-smi:ro \
 --volume /home/featurize/work:/home/featurize/work \
 --volume "$candidate_root:$candidate_root" \
 --volume "$candidate_root/data:$candidate_root/data:ro" \
 --volume /tmp/dual768_competition_addons:/opt/dual768_addons:ro \
 --workdir "$candidate_repo" \
 --env PYTHONPATH=/opt/dual768_addons --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 \
 --env PYTHONHASHSEED=43 --env HF_HUB_OFFLINE=1 --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
 --env OPENBLAS_NUM_THREADS=1 --env NUMEXPR_NUM_THREADS=1 --env TZ=Asia/Hong_Kong \
 --entrypoint python "$candidate_image" -B "$@"
