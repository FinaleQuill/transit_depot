#!/usr/bin/env bash
set -u
base=/home/featurize/work/cloud_project_v2/artifacts/compliance_closeout_20261102
image='crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete@sha256:125d8bcb4ce5a2a9acec6d9838b37e2ae1e2014c09b848700f9530f2f7856a0b'
docker run --rm --name compliance-041-cost100 --gpus all --network none --shm-size=1g \
  -v "$base/package:/audit:ro" -v "$base/cost_inputs:/inputs:ro" \
  -v "$base/cost_execution:/out" -v /tmp/dual768_competition_addons:/addons:ro \
  -e PYTHONPATH=/addons -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=1 \
  -w /audit "$image" \
  python cost.py submission.csv /inputs/pictures_100 /inputs/pictures_5 \
  --model-dir model/historical_041 --manifest /inputs/original_test_identity_manifest.csv \
  --image-submission evidence/runtime/041_historical/submission.image_predictions.csv \
  --output-dir /out/cost100 \
  > "$base/cost_execution/run.log" 2>&1
result=$?
printf '%s\n' "$result" > "$base/cost_execution/exit_code.txt"
exit "$result"
