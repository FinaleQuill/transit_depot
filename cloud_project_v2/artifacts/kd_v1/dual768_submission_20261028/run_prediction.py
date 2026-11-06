"""Execute the explicitly requested frozen 033 test prediction; never train or fit."""
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
SEED = 43

if len(sys.argv) > 1 and sys.argv[1] == "--invoke":
    import random
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    from predict import main
    raise SystemExit(main(sys.argv[2:]))

from training_evidence_monitor import snapshot_source

model = ROOT.parent / "deliveries/033_dual_tiny768_seed43_25_seed44_75/fp16_hflip2"
output = model.parent / "test_predictions/submission.csv"
manifest = ROOT / "data/clean_v1_reviewed/test_manifest.csv"
acceptance = json.loads((ROOT / "docs/DUAL_TINY768_ACCEPTANCE_20261028.json").read_text())
assert hashlib.sha256((model / "config.json").read_bytes()).hexdigest() == acceptance["deployment_config_sha256"]
if output.parent.exists() or (EVIDENCE / "execution.json").exists():
    raise FileExistsError("Do not overwrite an existing test prediction or execution record")

sources = sorted(set(list(ROOT.glob("*.py")) + list(ROOT.glob("*.sh"))
                     + list(ROOT.glob("requirements*.txt")) + [ROOT / "AGENTS.md"]))
protected = sources + list(model.glob("*.pt")) + [model / "config.json", manifest,
    ROOT / "data/clean_v1_reviewed/train_clean.csv", ROOT / "data/clean_v1_reviewed/valid_manifest.csv",
    ROOT / "data/cv5_overnight/folds.json", ROOT / "docs/DUAL_TINY768_ACCEPTANCE_20261028.json",
    ROOT / "docs/DUAL_TINY768_SELECTION_20261028.json", ROOT.parent / "rules.pdf",
    Path("/home/featurize/data/cxr_first/cxr_first/test.csv"), Path(__file__).resolve()]
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
before = {"captured_at_utc": datetime.now(timezone.utc).isoformat(),
          "files": {str(p.resolve()): digest(p) for p in protected}}
(EVIDENCE / "protected_before.json").write_text(json.dumps(before, indent=2) + "\n")
snapshot_source(ROOT, EVIDENCE / "source")
arguments = ["--model-dir", str(model), "--image-root", "/home/featurize/data/cxr_first",
             "--manifest", str(manifest), "--output", str(output)]
command = [sys.executable, "-u", str(Path(__file__).resolve()), "--invoke", *arguments]
environment = os.environ.copy()
environment.update(PYTHONHASHSEED=str(SEED), PYTHONDONTWRITEBYTECODE="1", NO_ALBUMENTATIONS_UPDATE="1",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
receipt = {"command": command, "application_entrypoint": "predict.py:main",
    "cwd": str(ROOT), "source_archive": str(EVIDENCE / "source"), "seed": SEED,
    "deployment_config_sha256": acceptance["deployment_config_sha256"],
    "test_manifest_sha256": digest(manifest), "training_started": False,
    "test_used_for_fitting_or_selection": False, "platform_uploaded": False,
    "started_utc": datetime.now(timezone.utc).isoformat(), "exit_code": None}
receipt_path = EVIDENCE / "execution.json"
receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
print("Generating authorized frozen-model submission.csv; log captured in prediction.log", flush=True)
started = time.monotonic()
try:
    with (EVIDENCE / "prediction.log").open("x") as stream:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    receipt["exit_code"] = result.returncode
finally:
    receipt.update(ended_utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic() - started)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k: receipt[k] for k in ["exit_code", "seconds"]}), flush=True)
raise SystemExit(receipt["exit_code"] if receipt["exit_code"] is not None else 1)
