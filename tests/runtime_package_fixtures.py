"""Small synthetic packaging fixtures, never executable native/model artifacts."""
from pathlib import Path
import hashlib
import json
from unittest.mock import patch

import package_kd_submission as package


def add_runtime_support(source: Path, case) -> None:
    for name in package.NATIVE_SOURCE_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"Synthetic native source/license fixture: {name}\n")
    (source / "native_lanczos/_core.cpython-312-x86_64-linux-gnu.so").write_bytes(b"synthetic, non-executable native artifact")
    vendor = source / "vendor"
    vendor.mkdir()
    raw = b"synthetic public dependency archive; never installed"
    digest = hashlib.sha256(raw).hexdigest()
    (vendor / package.ADDONS_FILES[0]).write_bytes(raw)
    (vendor / package.ADDONS_FILES[1]).write_text(json.dumps({"archive_sha256": digest}))
    override = patch.object(package, "ADDONS_ARCHIVE_SHA256", digest)
    override.start()
    case.addCleanup(override.stop)
    for relative in ("native_lanczos/build/ignore.o", "native_lanczos/dist/ignore.whl",
                     "native_lanczos/__pycache__/ignore.pyc"):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"must not be packaged")


def passing_cost(inventory, submission_sha, sidecar_sha="d" * 64, config_sha=None):
    semantic = {"match": True, "semantic_match": True, "identity_coverage_match": True,
                "mismatched_study_count": 0, "semantic_mismatched_study_count": 0,
                "max_absolute_error": 0.0}
    images = {**semantic, "compared_images": 100, "expected_images": 100, "mismatched_image_count": 0}
    return {"inference_files": inventory, "submission_sha256": submission_sha,
            "deployment_config_sha256": config_sha, "official_compliance_pass": True,
            "reproducibility": {"seed": 43}, "mean_ms": 1.0,
            "environment": {"gpu_name": "Synthetic T4 fixture", "official_environment_compatible": True},
            "protocol": {"measured_images": 100, "warmup_images": 5, "rounds": 3,
                         "batch_size": 1, "execution": "serial"},
            "comparison": {"status": "matched", "coverage": {"complete_studies_proven": True},
                           "rounds": [dict(semantic) for _ in range(3)],
                           "image_rounds": [dict(images) for _ in range(3)],
                           "image_submission_sha256": sidecar_sha}}
