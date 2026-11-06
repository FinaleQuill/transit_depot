"""Registered paired Tiny/Base complete-Study training; dry-run unless --execute.

The final E2 EMA is last.pt. best.pt remains a diagnostic validation-selected
checkpoint and must not be exported as the precommitted treatment result.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import json
import os
from pathlib import Path
import sys

from audit_trail import training_output_has_artifacts
from kd_common import BASE_BACKBONE, TINY_BACKBONE, sha256_file
import study_student
from study_student import StudyConfig
from train_distill import _configuration, validate_output_paths
import dataset_overnight
from tight_crop_ab_data import make_ab_study_loader, make_ab_valid_loader


ROOT = Path(__file__).resolve().parent
PLAN = ROOT / "docs/TIGHT_CROP_AB_PLAN_20261102.json"
SOFT_LABELS = ROOT.parent / "runs/024_convnext_v1_base768_mixed20_12_cv5/oof_logits_tta_v2"
FOLD_PLAN = ROOT / "data/cv5_overnight/folds.json"
SNAPSHOTS = ROOT / "artifacts/kd_v1/student_upper_bound_20261030/models"
INIT_SHA = {
    "tiny": "aa3c632e7764b57a522920369e16d9cb4768e43e008c0588439b4438aa88fd6f",
    "base": "9cb733eb65b71eeb0fe7633122dcb085b47fc4c13b4f46f139f5b773a5e805a3",
}
SOURCE_FILES = ("tight_crop_ab_training.py", "tight_crop_ab_data.py",
                "tight_crop_ab_preprocess.py", "tight_crop.py")


@dataclass(frozen=True)
class ABConfig(StudyConfig):
    arm: str = "A"
    member: str = "tiny"
    ab_plan_sha256: str = ""
    ab_preprocess_sha256: str = ""
    ab_crop_sha256: str = ""
    erase_probability: float = 0.0

    def validate(self) -> None:
        super().validate()
        if self.arm not in ("A", "B") or self.member not in ("tiny", "base"):
            raise ValueError("A/B run requires a registered arm and member")
        backbone = TINY_BACKBONE if self.member == "tiny" else BASE_BACKBONE
        if self.backbone_name != backbone:
            raise ValueError("A/B member and backbone differ")
        if any(len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
               for v in (self.ab_plan_sha256, self.ab_preprocess_sha256, self.ab_crop_sha256)):
            raise ValueError("A/B plan and source hashes are required")
        if (self.seed != 48 or self.epochs != 2 or self.patience != 0
                or self.pack_images != 8 or self.warmup_epochs != 1
                or self.kd_warmup_epochs != 0 or self.alpha != 0.2
                or self.temperature != 2 or self.ema_decay != 0.999
                or self.erase_probability != (0.5 if self.arm == "B" else 0.0)):
            raise ValueError("A/B fixed training schedule or erasing rule changed")
        expected_lr = (1e-5, 1e-4, 0.1) if self.member == "tiny" else (5e-6, 5e-5, 0.15)
        if (self.backbone_lr, self.head_lr, self.drop_path) != expected_lr:
            raise ValueError("A/B member learning rates or drop path changed")
        if self.resume is None and (self.init_student_ema is None
                                    or self.init_student_ema_sha256 != INIT_SHA[self.member]):
            raise ValueError("Fresh A/B training requires the registered FP32 EMA initialization")


def _preprocessing(config: ABConfig):
    prior = study_student.student_preprocessing_config_original(config)
    return {**prior, "resize": "registered_tight_crop_ab_canvas",
            "eval_transform": "full_decode_exact_reference_v1+shared_AB_canvas+ImageNet_Normalize",
            "decoder_candidate": "full_decode_exact_reference_v1",
            "arm": config.arm, "member": config.member,
            "ab_arm": config.arm, "ab_member": config.member,
            "crop_policy": "none" if config.arm == "A" else "tight_crop.detect_tight_crop",
            "ab_plan_sha256": config.ab_plan_sha256,
            "ab_preprocess_sha256": config.ab_preprocess_sha256,
            "ab_crop_sha256": config.ab_crop_sha256,
            "training_random_erasing": {
                "p": config.erase_probability, "area_fraction_of_content": [0.005, 0.02],
                "before_normalization": True, "valid_or_inference": False,
                "triggered_location_mix": {"outer_20pct_content_edge_band": 0.5,
                                           "uniform_position_in_content": 0.5},
                "intensity_targeting": False},
            "paired_training": {"seed": 48, "fixed_epochs": 2,
                                "checkpoint": "last.pt E2 EMA"}}


def _source_version(paths):
    originals = list(paths)
    extras = [ROOT / name for name in SOURCE_FILES] + [PLAN]
    return study_student.capture_source_version_original([*originals, *extras])


@contextmanager
def _ab_loaders(arm: str):
    old_study = study_student.make_study_loader
    old_valid = dataset_overnight.make_overnight_loader
    old_preprocessing = study_student.student_preprocessing_config
    old_source = study_student.capture_source_version
    study_student.student_preprocessing_config_original = old_preprocessing
    study_student.capture_source_version_original = old_source
    study_student.make_study_loader = partial(make_ab_study_loader, arm=arm)
    dataset_overnight.make_overnight_loader = partial(make_ab_valid_loader, arm=arm)
    study_student.student_preprocessing_config = _preprocessing
    study_student.capture_source_version = _source_version
    try:
        yield
    finally:
        study_student.make_study_loader = old_study
        dataset_overnight.make_overnight_loader = old_valid
        study_student.student_preprocessing_config = old_preprocessing
        study_student.capture_source_version = old_source
        del study_student.student_preprocessing_config_original
        del study_student.capture_source_version_original


def build_config(*, arm: str, member: str, output: Path, resume: Path | None = None,
                 image_root: Path = Path("/home/featurize/data/cxr_first"),
                 num_workers: int = 4, valid_batch_size: int = 4) -> ABConfig:
    if arm not in ("A", "B") or member not in ("tiny", "base"):
        raise ValueError("Expected --arm A/B and --member tiny/base")
    if not PLAN.is_file():
        raise FileNotFoundError("The frozen A/B plan must exist before preview or execution")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("The registered A/B plan must be a JSON object")
    source = SNAPSHOTS / ("A" if member == "tiny" else "B") / "best_ema_fp32.pt"
    if resume is None and sha256_file(source) != INIT_SHA[member]:
        raise ValueError("Registered initial EMA SHA256 changed")
    if not (ROOT / "tight_crop_ab_preprocess.py").is_file():
        raise FileNotFoundError("Shared A/B preprocessing module is required")
    backbone = TINY_BACKBONE if member == "tiny" else BASE_BACKBONE
    lr = (1e-5, 1e-4, 0.1) if member == "tiny" else (5e-6, 5e-5, 0.15)
    config = ABConfig(fold_plan=FOLD_PLAN, soft_labels=SOFT_LABELS, output=output,
        image_root=image_root, resume=resume, init_student_ema=source if resume is None else None,
        init_student_ema_sha256=INIT_SHA[member] if resume is None else None,
        backbone_name=backbone, backbone_lr=lr[0], head_lr=lr[1], drop_path=lr[2],
        seed=48, epochs=2, patience=0, warmup_epochs=1, kd_warmup_epochs=0,
        alpha=0.2, temperature=2.0, ema_decay=0.999, pack_images=8,
        num_workers=num_workers, valid_batch_size=valid_batch_size,
        arm=arm, member=member, ab_plan_sha256=sha256_file(PLAN),
        ab_preprocess_sha256=sha256_file(ROOT / "tight_crop_ab_preprocess.py"),
        ab_crop_sha256=sha256_file(ROOT / "tight_crop.py"),
        erase_probability=0.5 if arm == "B" else 0.0)
    config.validate()
    validate_output_paths(config)
    return config


def load_final_ema(path: Path, *, arm: str, member: str):
    """Load only the registered E2 EMA for downstream export or evaluation."""
    import torch

    path = Path(path).resolve(strict=True)
    if path.name != "last.pt":
        raise ValueError("The registered A/B result is E2 last.pt, never best.pt")
    config = build_config(arm=arm, member=member, output=path.parent,
                          resume=path)
    digest = sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if sha256_file(path) != digest:
        raise RuntimeError("A/B checkpoint changed while loading")
    training = checkpoint.get("training_config", {})
    preprocessing = checkpoint.get("preprocessing", {})
    if (checkpoint.get("epoch") != 2
            or checkpoint.get("progress", {}).get("completed_epochs") != 2
            or checkpoint.get("weights_source") != "ema"
            or checkpoint.get("validation_complete") is not True
            or training.get("arm") != arm or training.get("member") != member
            or training.get("ab_plan_sha256") != config.ab_plan_sha256
            or training.get("ab_preprocess_sha256") != config.ab_preprocess_sha256
            or training.get("ab_crop_sha256") != config.ab_crop_sha256
            or preprocessing.get("ab_arm") != arm
            or preprocessing.get("decoder_candidate") != "full_decode_exact_reference_v1"
            or checkpoint.get("student_warm_start", {}).get("source_snapshot_sha256")
                != INIT_SHA[member]
            or not isinstance(checkpoint.get("model_state_dict"), dict)):
        raise ValueError("A/B final checkpoint identity or fixed E2 EMA is invalid")
    return checkpoint


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument("--member", choices=("tiny", "base"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--image-root", type=Path,
                        default=Path("/home/featurize/data/cxr_first"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--valid-batch-size", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config = build_config(arm=args.arm, member=args.member, output=args.output,
        resume=args.resume, image_root=args.image_root, num_workers=args.num_workers,
        valid_batch_size=args.valid_batch_size)
    if not args.execute:
        print(json.dumps({"dry_run": True, "config": _configuration(config),
            "checkpoint_selection": "fixed_final_E2_EMA_from_last.pt",
            "preprocessing": {"arm": args.arm, "plan_sha256": config.ab_plan_sha256,
                "shared_module_sha256": config.ab_preprocess_sha256,
                "crop_module_sha256": config.ab_crop_sha256}}, indent=2))
        return 0
    output = config.output.resolve()
    if config.resume is None and training_output_has_artifacts(output):
        raise FileExistsError("Fresh A/B run requires an empty output directory")
    if config.resume is not None and (config.resume.resolve().parent != output
                                       or config.resume.name != "last.pt"):
        raise ValueError("A/B resume must use this run directory's last.pt")
    if os.environ.get("KD_TRAINING_LOG_ACTIVE") != "1":
        from training_evidence_monitor import run_logged
        return run_logged(argparse.Namespace(evidence_dir=output / "evidence",
            command=[sys.executable, "-u", str(Path(__file__).resolve()),
                     *(argv if argv is not None else sys.argv[1:])]))
    with _ab_loaders(args.arm):
        print(json.dumps(study_student.run_training(config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
