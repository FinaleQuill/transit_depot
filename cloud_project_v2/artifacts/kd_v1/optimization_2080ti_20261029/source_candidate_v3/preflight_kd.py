"""只读训练准备检查：不构建网络、不下载权重、不读取图片像素。"""
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import sys

from kd_common import BASE_BACKBONE, TINY_BACKBONE, sha256_file, write_json


def inspect_environment() -> dict:
    versions, problems = {}, []
    for package in ("torch", "torchvision", "timm", "numpy", "Pillow", "opencv-python-headless",
                    "albumentations", "scikit-learn", "safetensors", "huggingface-hub"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            problems.append(f"Missing dependency: {package}")
    if sys.version_info[:2] != (3, 12):
        problems.append("Use the documented Python 3.12 training environment")
    if not versions.get("torch", "").startswith("2.10.0"):
        problems.append("Use torch 2.10.0 + CUDA 12.8 from requirements_kd.txt")
    if not versions.get("torchvision", "").startswith("0.25.0"):
        problems.append("Use torchvision 0.25.0 paired with torch 2.10.0")
    return {"python": platform.python_version(), "packages": versions,
            "training_environment_pass": not problems, "problems": problems,
            "cuda_execution_checked": False}


def inspect_data(fold_plan: Path, image_root: Path, check_files: bool) -> dict:
    from cv_splits import load_fold_records, read_fold_plan
    from dataset import read_manifest, resolve_native_image_root

    plan = read_fold_plan(fold_plan)
    summaries, missing = {}, []
    # 不扫描 test 或使用其分布；训练前只核查 train/valid。
    for split, source in (("train", plan["source_train_manifest"]), ("valid", plan["official_holdout"])):
        records = read_manifest(source["path"], split=split)
        root = resolve_native_image_root(image_root, split=split)
        if check_files:
            for row in records:
                path = root / row["relative_path"]
                if not path.is_file() or path.stat().st_size == 0:
                    missing.append(row["image_id"])
        summaries[split] = {"images": len(records), "studies": len({r["study_key"] for r in records}),
                            "patients": len({r["subject_id"] for r in records}),
                            "manifest_sha256": sha256_file(source["path"]), "resolved_image_root": str(root)}
    folds = []
    for fold in range(5):
        fitting = load_fold_records(fold_plan, fold, "train")
        held_out = load_fold_records(fold_plan, fold, "valid")
        overlap = {r["subject_id"] for r in fitting} & {r["subject_id"] for r in held_out}
        if overlap:
            raise ValueError("Patient leakage in teacher fold")
        folds.append({"fold": fold, "training_images": len(fitting), "oof_images": len(held_out),
                      "patient_overlap": len(overlap)})
    return {"data_pass": not missing, "splits": summaries, "folds": folds,
            "missing_or_empty_count": len(missing), "missing_image_ids_first20": missing[:20],
            "image_existence_checked": check_files, "image_bytes_sha256_checked": False,
            "note": "Training Dataset verifies each accepted image SHA256 when it is read."}


def main(argv=None) -> int:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-plan", type=Path, default=project / "data/cv5_overnight/folds.json")
    parser.add_argument("--image-root", type=Path, default=Path("/home/featurize/data/cxr_first"))
    parser.add_argument("--check-files", action="store_true", help="Stat every train/valid file, without decoding")
    parser.add_argument("--metadata-only", action="store_true", help="Audit data even when the future training environment is not installed")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = {"training_started": False, "models_built": 0, "pretrained_downloads": 0,
              "approved_backbones": [BASE_BACKBONE, TINY_BACKBONE],
              "environment": inspect_environment(),
              "data": inspect_data(args.fold_plan, args.image_root, args.check_files)}
    report["ready_for_training_preflight"] = (report["environment"]["training_environment_pass"]
                                               and report["data"]["data_pass"] and args.check_files)
    if args.output:
        if args.output.exists():
            raise FileExistsError("Choose a new preflight report path")
        if args.output.resolve().is_relative_to(args.image_root.resolve()):
            raise ValueError("Do not write into the original image directory")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    passed = report["data"]["data_pass"] if args.metadata_only else report["ready_for_training_preflight"]
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
