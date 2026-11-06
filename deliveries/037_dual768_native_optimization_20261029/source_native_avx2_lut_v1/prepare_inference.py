"""Prepare the submitted project before predict/cost, without model inference.

No core package is installed or replaced. The optional verified public-addon
archive supplies timm and its missing dependencies. Native CPU code is built in
the organizer image; no CUDA compiler, training data, or pretrained download is
needed. All preparation finishes before any measured image is read.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

from inference_runtime import ADDONS_ARCHIVE_SHA256, CORE_MODULES, activate_dependencies


def restore_addons(archive: Path, manifest_path: Path, root: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != ADDONS_ARCHIVE_SHA256 or manifest.get("archive_sha256") != digest:
        raise ValueError("Public-addon archive SHA-256 mismatch")
    target = root / "_runtime_addons"
    if target.exists():
        activate_dependencies()
        restored = target
    else:
        temporary = Path(tempfile.mkdtemp(prefix=".prepare-addons-", dir=root))
        try:
            expected = {"dual768_competition_addons/" + entry["path"] for entry in manifest["files"]}
            with tarfile.open(archive) as bundle:
                members = bundle.getmembers()
                files = {item.name for item in members if item.isfile()}
                if files != expected or any(not (item.isfile() or item.isdir()) for item in members):
                    raise ValueError("Unexpected public-addon archive members")
                bundle.extractall(temporary, filter="data")
            restored = temporary / "dual768_competition_addons"
            for name in CORE_MODULES:
                if (restored / name).exists() or (restored / (name + ".py")).exists():
                    raise ValueError("Addon archive shadows an image core module")
            verify_addons(restored, manifest)
            (restored / "RESTORED_ADDONS.json").write_text(json.dumps({"archive_sha256": digest,
                "verified_files": len(manifest["files"]), "packages": manifest["packages"]}, indent=2) + "\n")
            restored.rename(target)
            restored = target
        finally:
            shutil.rmtree(temporary)
    verify_addons(restored, manifest)
    activate_dependencies()


def verify_addons(target: Path, manifest: dict) -> None:
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid addon manifest path")
        path = target / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError("Addon file missing or altered: " + str(relative))
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Addon file SHA mismatch: " + str(relative))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addons-archive", type=Path)
    parser.add_argument("--addons-manifest", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Prepare inside the organizer Python 3.12 environment")
    expected = {"torch": "2.10.0", "torchvision": "0.25.0", "numpy": "1.26.4", "Pillow": "10.4.0"}
    for package, version in expected.items():
        if importlib.metadata.version(package).split("+")[0] != version:
            raise RuntimeError("Unexpected core package version: " + package)
    activate_dependencies()
    archive = args.addons_archive or root / "vendor/competition_addons_cp312_linux_x86_64.tar.gz"
    manifest = args.addons_manifest or root / "vendor/competition_addons_archive_manifest.json"
    if args.check_only:
        pass
    elif archive.is_file() and manifest.is_file():
        restore_addons(archive, manifest, root)
    elif args.addons_archive is not None or args.addons_manifest is not None:
        raise FileNotFoundError("Both the verified addon archive and manifest are required")
    if importlib.metadata.version("timm") != "1.0.30":
        raise RuntimeError("The prepared project requires pinned timm 1.0.30")
    import fast_lanczos
    if not args.check_only and (args.force_build or not fast_lanczos.available()):
        with tempfile.TemporaryDirectory(prefix="native-lanczos-build-") as temporary:
            subprocess.run([sys.executable, "native_lanczos/setup.py", "build_ext", "--inplace", "--force",
                            "--build-temp", temporary], cwd=root, check=True)
        # A native module loaded before a forced rebuild remains mapped for this
        # process. A fresh child verifies the artifact used by predict/cost.
        subprocess.run([sys.executable, "-c", "import fast_lanczos; assert fast_lanczos.available()"], cwd=root, check=True)
    else:
        if not fast_lanczos.available():
            raise RuntimeError("Native backend missing; run preparation before timing")
    import torch
    report = {"python": sys.version, "core_versions": {name: importlib.metadata.version(name) for name in expected},
              "timm": importlib.metadata.version("timm"), "cuda_build": torch.version.cuda,
              "native_prepared": True, "native_threads_max": 8, "model_inference_executed": False,
              "training_executed": False, "core_packages_replaced": False,
              "project_local_addons": str(root / "_runtime_addons") if (root / "_runtime_addons").exists() else None}
    if torch.version.cuda != "12.8":
        raise RuntimeError("Expected organizer CUDA 12.8 torch build")
    if args.output is not None:
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
