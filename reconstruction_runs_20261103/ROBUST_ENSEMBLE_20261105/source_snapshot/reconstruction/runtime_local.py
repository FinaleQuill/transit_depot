"""Pinned E2 runtime with durable, verified generic-library addons.

The frozen E0/E1 runtime remains unchanged. Containers have no network and
only the isolated research outputs and explicit mechanics-smoke root are
writable. This module neither launches training on import nor loads models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from reconstruction.runtime import IMAGE, ORIGINAL, OUTPUT, ROOT

ADDON_ROOT = OUTPUT / "assets/e2_runtime_addons_20261104"
ADDON_MANIFEST = ADDON_ROOT / "manifest.json"
ADDON_MANIFEST_SHA256 = "9d31553bb8f063802b80052ec4948b6ffc608f8c13eba1fcf1a3fd5c8283c21f"
SMOKE_ROOT = Path("/home/featurize/data/cxr_reconstruction_smokes_20261104")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_addons():
    """Refuse altered or incomplete libraries before constructing a command."""
    if sha(ADDON_MANIFEST) != ADDON_MANIFEST_SHA256:
        raise ValueError("E2 addon manifest differs from the registered SHA256")
    manifest = json.loads(ADDON_MANIFEST.read_text())
    if manifest["schema"] != "cxr_e2_runtime_addons_v1" or manifest["base_image"] != IMAGE:
        raise ValueError("E2 addon provenance refers to another runtime")
    registered = manifest["files"]
    actual = {str(p.relative_to(ADDON_ROOT)) for p in ADDON_ROOT.rglob("*") if p.is_file()}
    if actual != set(registered) | {"manifest.json"}:
        raise ValueError("E2 addon inventory contains missing or unregistered files")
    for relative, expected in registered.items():
        path = ADDON_ROOT / relative
        if (path.is_symlink() or not path.resolve().is_relative_to(ADDON_ROOT.resolve())
                or path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]):
            raise ValueError(f"E2 addon content differs: {relative}")
    return {"manifest": str(ADDON_MANIFEST), "sha256": ADDON_MANIFEST_SHA256,
            "packages": manifest["packages"], "total_bytes": manifest["total_bytes"]}


def command(arguments, *, gpu=True, name=None):
    verify_addons()
    mounts = [
        (ROOT, ROOT, "ro"), (ORIGINAL, ORIGINAL, "ro"),
        (OUTPUT, OUTPUT, "rw"), (OUTPUT / "assets", OUTPUT / "assets", "ro"),
        (OUTPUT / "splits", OUTPUT / "splits", "ro"),
        (Path("/home/featurize/data"), Path("/home/featurize/data"), "ro"),
        (SMOKE_ROOT, SMOKE_ROOT, "rw"),
        (ADDON_ROOT / "site", Path("/opt/cxr_e2_addons"), "ro"),
    ]
    for source, _, _ in mounts:
        if not source.is_dir():
            raise FileNotFoundError(f"Required E2 mount is absent: {source}")
    args = ["docker", "run", "--rm", "--pull=never", "--network=none", "--shm-size=4g"]
    if gpu:
        args += ["--gpus", "all"]
    if name:
        args += ["--name", name]
    args += ["--user", f"{os.getuid()}:{os.getgid()}"]
    for source, target, mode in mounts:
        args += ["--volume", f"{source}:{target}:{mode}"]
    for key, value in {
        "PYTHONPATH": f"/opt/cxr_e2_addons:{ROOT}", "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_XET": "1",
        "NO_ALBUMENTATIONS_UPDATE": "1", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "TZ": "Asia/Hong_Kong",
        "PYTHONHASHSEED": "20261104", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }.items():
        args += ["--env", f"{key}={value}"]
    return args + ["--workdir", str(ROOT), "--entrypoint", "python3", IMAGE, "-u", *arguments]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if not argv:
        parser.error("Python arguments required after --")
    raise SystemExit(subprocess.call(command(argv, gpu=not args.cpu, name=args.name)))


if __name__ == "__main__":
    main()
