"""Run only the three explicitly authorized 033 inference stages, with receipts."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from training_evidence_monitor import snapshot_source

stage = sys.argv[1]
model = "../deliveries/033_dual_tiny768_seed43_25_seed44_75/fp16_hflip2"
base = "../deliveries/033_dual_tiny768_seed43_25_seed44_75"
inputs = "../deliveries/031_resource_stress_20261027/inputs"
arguments = {
    "compose": ["compose_students.py", "--model-dirs",
        "../deliveries/029_tiny768_kd_seed43/fp16_hflip2",
        "../deliveries/030_tiny768_kd_seed44/fp16_hflip2",
        "--weights", "1", "3", "--tta", "hflip2", "--output", model, "--execute"],
    "validate": ["validate_student.py", "--model-dir", model,
        "--image-root", "/home/featurize/data/cxr_first",
        "--manifest", "data/clean_v1_reviewed/valid_manifest.csv",
        "--output-dir", base + "/valid_fp16_hflip2"],
    "benchmark": ["benchmark_local_resources.py", "--model-dir", model,
        "--pictures-100", inputs + "/pictures_100", "--pictures-5", inputs + "/pictures_5",
        "--selection-json", inputs + "/selection.json", "--output-dir", base + "/latency4090",
        "--profile", "dual768_seed43_25_seed44_75_native4090",
        "--memory-limit-gib", "16", "--execute"],
}
if stage not in arguments:
    raise ValueError("Only compose, validate or benchmark is authorized")
receipt_path = EVIDENCE / f"execution_{stage}.json"
if receipt_path.exists():
    raise FileExistsError("Execution receipts are immutable; inspect the existing stage")
source = EVIDENCE / "source"
if not source.exists():
    snapshot_source(ROOT, source)
manifest = json.loads((source / "source_manifest.json").read_text())
for item in manifest["files"]:
    if hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
        raise RuntimeError("Execution source differs from the captured version")
environment = os.environ.copy()
environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   PYTHONDONTWRITEBYTECODE="1", NO_ALBUMENTATIONS_UPDATE="1")
if stage == "compose":
    environment["CUDA_VISIBLE_DEVICES"] = ""
elif any(environment.get(key) for key in ["CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"]):
    raise RuntimeError("Native GPU stages require no inherited MPS resource profile")
command = [sys.executable, "-u", *arguments[stage]]
receipt = {"stage": stage, "command": command, "cwd": str(ROOT),
    "source_archive": str(source), "started_utc": datetime.now(timezone.utc).isoformat(),
    "environment": {k: environment.get(k) for k in ["CUDA_VISIBLE_DEVICES", "CUDA_MPS_PIPE_DIRECTORY",
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]},
    "training_started": False, "test_data_used": False, "exit_code": None}
receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
print(f"Starting authorized {stage}; output is captured in {stage}.log", flush=True)
started = time.monotonic()
try:
    with (EVIDENCE / f"{stage}.log").open("x") as log:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    receipt["exit_code"] = result.returncode
finally:
    receipt.update(ended_utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic() - started)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k: receipt[k] for k in ["stage", "exit_code", "seconds"]}), flush=True)
sys.exit(receipt["exit_code"] if receipt["exit_code"] is not None else 1)
