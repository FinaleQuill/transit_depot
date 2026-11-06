"""Run the one uninstrumented main benchmark and retain its exact exit code."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

save = Path("/home/featurize/work/dual768_proxy_timing_results")
bundle = Path("/home/featurize/work/deliveries/034_dual768_proxy_timing_20261029/bundle")
python = "/tmp/dual768_timing_env/bin/python"
run_id = "dual768_033_native_" + datetime.datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y%m%d_%H%M%S")
output = save / run_id
assert not output.exists()
env = os.environ.copy()
overrides = dict(PYTHONHASHSEED="43", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
env.update(overrides)
command = [python, "-B", str(save / "preparation/native_seed43.py"),
           "--model-dir", "model/033", "--pictures-100", "inputs/pictures_100",
           "--pictures-5", "inputs/pictures_5", "--selection-json", "inputs/selection.portable.json",
           "--output-dir", str(output), "--profile", run_id, "--memory-limit-gib", "8", "--execute"]
receipt = dict(run_id=run_id, output_dir=str(output), argv=command, cwd=str(bundle),
               environment_overrides=overrides, inherited_cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES"),
               started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               seed=43, allocator_limit_gib=8, instrumented=False)
(save / (run_id + ".command.json")).write_text(json.dumps(receipt, indent=2) + "\n")
(save / "main_run.json").open("x").write(json.dumps(receipt, indent=2) + "\n")
started = time.perf_counter()
with (save / (run_id + ".console.log")).open("x") as console:
    result = subprocess.run(command, cwd=bundle, env=env, stdout=console, stderr=subprocess.STDOUT)
receipt.update(ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               wall_seconds=time.perf_counter() - started, exit_code=result.returncode)
(save / (run_id + ".exit_code.txt")).write_text(str(result.returncode) + "\n")
(save / (run_id + ".execution_receipt.json")).write_text(json.dumps(receipt, indent=2) + "\n")
(save / (run_id + ".execution_receipt.txt")).write_text("\n".join(f"{key}={receipt[key]}" for key in ("started_utc", "ended_utc", "wall_seconds", "exit_code")) + "\n")
print(json.dumps(receipt, indent=2))
raise SystemExit(result.returncode)
