"""Activate project-local public addons without shadowing image core packages."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ADDONS_ARCHIVE_SHA256 = "eb7ecd612df5568f9f87b9f3bc3ec71cf775ed3d00c4b072467f308e2f32fe10"
CORE_MODULES = ("torch", "torchvision", "numpy", "PIL", "scipy", "sklearn", "nvidia", "triton")


def activate_dependencies() -> Path | None:
    target = Path(__file__).resolve().parent / "_runtime_addons"
    if not target.exists():
        return None
    marker = target / "RESTORED_ADDONS.json"
    if not marker.is_file() or json.loads(marker.read_text()).get("archive_sha256") != ADDONS_ARCHIVE_SHA256:
        raise RuntimeError("Project-local addons have no verified preparation receipt")
    if any((target / name).exists() or (target / (name + ".py")).exists() for name in CORE_MODULES):
        raise RuntimeError("Project-local addons must not shadow competition image core modules")
    if str(target) not in sys.path:
        sys.path.insert(0, str(target))
    return target
