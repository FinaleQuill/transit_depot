"""Diagnostic launcher; keeps the existing inference source and pinned image intact."""
from pathlib import Path
import os
import subprocess
import sys

IMAGE = "crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete@sha256:5e333ae965f02b4c80e4a5c89a9d33cae9ba54ffaea2f73217ffcd4b4a7fabc6"
REPO = "/home/featurize/work/cloud_project_v2"
LOCAL = "/tmp/dual768_2080ti_diagnosis"
ADDONS = "/tmp/dual768_competition_addons"

def main():
    assert sys.argv[1:], "Pass Python arguments"
    assert Path(ADDONS, "timm-1.0.30.dist-info").is_dir()
    Path(LOCAL, "profile_tmp").mkdir(parents=True, exist_ok=True)
    cmd = ["docker", "run", "--rm", "--pull=never", "--runtime=runc", "--network", "none", "-i",
           "--user", f"{os.getuid()}:{os.getgid()}"]
    for dev in ("nvidia0", "nvidiactl", "nvidia-uvm", "nvidia-uvm-tools"):
        assert Path("/dev", dev).exists()
        cmd += ["--device", "/dev/" + dev]
    for lib in ("libcuda.so.1", "libnvidia-ml.so.1", "libnvidia-ptxjitcompiler.so.1", "libnvidia-gpucomp.so.610.57.04"):
        assert Path("/lib/x86_64-linux-gnu", lib).is_file()
        cmd += ["--volume", f"/lib/x86_64-linux-gnu/{lib}:/usr/local/nvidia/lib64/{lib}:ro"]
    for mount in ("/usr/bin/nvidia-smi:/usr/local/nvidia/bin/nvidia-smi:ro",
                  "/home/featurize/work:/home/featurize/work",
                  "/home/featurize/data:/home/featurize/data:ro",
                  f"{ADDONS}:/opt/dual768_addons:ro", f"{LOCAL}:{LOCAL}"):
        cmd += ["--volume", mount]
    env = {"PYTHONPATH": "/opt/dual768_addons", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONHASHSEED": "43", "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "TZ": "Asia/Hong_Kong",
           "TMPDIR": LOCAL + "/profile_tmp"}
    for key, value in env.items():
        cmd += ["--env", f"{key}={value}"]
    cmd += ["--workdir", REPO, "--entrypoint", "python", IMAGE, "-B", *sys.argv[1:]]
    return subprocess.run(cmd).returncode

if __name__ == "__main__":
    raise SystemExit(main())
