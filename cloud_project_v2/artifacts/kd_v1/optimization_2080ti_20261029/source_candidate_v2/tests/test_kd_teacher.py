"""教师协议的CPU元数据测试；不构建模型、不读取原图、不训练。"""
from __future__ import annotations

import contextlib
from dataclasses import asdict, replace
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from model_overnight import (
    BASE_BACKBONE_NAME, MODEL_NAME, TINY_BACKBONE_NAME, validate_model_config, build_overnight_model,
)
import run_teachers
from train_overnight import CLASS_IDS, _jsonable, preprocessing_config
from train_teacher import TeacherConfig, SELECTION_PROTOCOL, _final_payload, validate_teacher_resume, parse_args


class TeacherProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="kd_teacher_metadata_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plan = self.root / "folds.json"
        self.plan.write_text("{}")
        self.config = TeacherConfig(fold_plan=self.plan, fold=0, output=self.root / "fold_0",
                                    image_root=self.root / "readonly", epochs=3)
        self.binding = {"source_manifest_sha256": "a" * 64, "fold_plan_sha256": "b" * 64,
            "fold_index": 0, "training_subject_ids": ["p1"], "oof_subject_ids": ["p2"],
            "selection_protocol": SELECTION_PROTOCOL, "outer_fold_used_for_selection": False}
        self.validation = types.SimpleNamespace(validate_pretrained_provenance=lambda *_args: None)

    def checkpoint(self, epoch: int = 1) -> dict:
        complete = epoch == self.config.epochs
        return {"schema_version": 6, "artifact_type": "kd_teacher", "model_name": MODEL_NAME,
            "model_config": self.config.model_config(), "class_ids": CLASS_IDS, "image_size": 768,
            "preprocessing": preprocessing_config(image_size=768), "inference_precision": "bfloat16",
            **self.binding, "pretrained_provenance": {}, "resume_supported": True,
            "model_state_dict": {}, "raw_model_state_dict": {}, "optimizer_state_dict": {},
            "scaler_state_dict": {}, "rng_state": {}, "epoch": epoch,
            "progress": {"completed_epochs": epoch, "optimizer_updates": 4, "ema_updates": 4},
            "training_config": _jsonable(asdict(self.config)), "training_complete": complete,
            "inference_ready": complete}

    def validate(self, checkpoint: dict, config: TeacherConfig | None = None) -> None:
        with patch.dict("sys.modules", {"kd_common": self.validation}):
            validate_teacher_resume(checkpoint, config or self.config, self.binding, "bfloat16")

    def test_direct_teacher_defaults_to_metadata_preview(self) -> None:
        _config, dry_run = parse_args(["--fold-plan", str(self.plan), "--fold", "0",
                                       "--output", str(self.root / "teacher")])
        self.assertTrue(dry_run)

    def test_exact_approved_backbones_and_no_v2(self) -> None:
        for backbone in (BASE_BACKBONE_NAME, TINY_BACKBONE_NAME):
            self.assertEqual(validate_model_config({"backbone_name": backbone})["backbone_name"], backbone)
        with self.assertRaises(ValueError):
            validate_model_config({"backbone_name": "convnextv2_base.fcmae_ft_in1k"})

    def test_pinned_local_pretrained_file_reaches_timm_overlay(self) -> None:
        weights = self.root / "fixture.safetensors"
        weights.write_bytes(b"metadata fixture, not real weights")
        fake_model = types.SimpleNamespace(set_grad_checkpointing=lambda _enabled: None)
        fake_timm = types.SimpleNamespace(create_model=lambda *args, **kwargs: fake_model)
        with patch.dict("sys.modules", {"timm": fake_timm}), patch.object(fake_timm, "create_model", return_value=fake_model) as build:
            result = build_overnight_model(pretrained=True, pretrained_weights_path=str(weights))
        self.assertIs(result, fake_model)
        self.assertEqual(build.call_args.kwargs["pretrained_cfg_overlay"], {"file": str(weights)})
        with self.assertRaises(ValueError):
            build_overnight_model(pretrained=True, pretrained_weights_path="")

    def test_resume_uses_real_material_hash_validation(self) -> None:
        from kd_common import APPROVED_PRETRAINS, text_sha256
        revision = "d" * 40
        materials = {
            "model_card": "---\nlicense: apache-2.0\n---\nImageNet-22k",
            "apache_license": "Apache License\nunit test material",
            "upstream_license": "MIT License\nunit test material",
        }
        value = self.checkpoint()
        value["pretrained_provenance"] = {
            **APPROVED_PRETRAINS[BASE_BACKBONE_NAME], "backbone_name": BASE_BACKBONE_NAME,
            "weights_sha256": "e" * 64, "revision": revision,
            "weights_url": f"https://huggingface.co/timm/{BASE_BACKBONE_NAME}/resolve/{revision}/model.safetensors",
            "materials": {name: {"text": text, "sha256": text_sha256(text)} for name, text in materials.items()},
        }
        validate_teacher_resume(value, self.config, self.binding, "bfloat16")
        value["pretrained_provenance"]["materials"]["model_card"]["text"] += " altered"
        with self.assertRaises(ValueError):
            validate_teacher_resume(value, self.config, self.binding, "bfloat16")

    def test_teacher_cannot_enable_outer_early_stopping(self) -> None:
        self.config.validate()
        with self.assertRaises(ValueError):
            replace(self.config, patience=3).validate()

    def test_resume_requires_same_source_and_selection_protocol(self) -> None:
        self.validate(self.checkpoint())
        for field, value in (("schema_version", 5), ("model_name", "convnextv2_base_gap"),
                             ("fold_index", 1), ("selection_protocol", "best_outer_auc"),
                             ("source_manifest_sha256", "c" * 64)):
            broken = self.checkpoint()
            broken[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(broken)

    def test_resume_rejects_changed_horizon_and_allows_equivalent_microbatch(self) -> None:
        with self.assertRaises(ValueError):
            self.validate(self.checkpoint(), replace(self.config, epochs=4))
        self.validate(self.checkpoint(), replace(self.config, batch_size=4, accumulation_steps=8))
        with self.assertRaises(ValueError):
            self.validate(self.checkpoint(), replace(self.config, accumulation_steps=8))

    def test_unfinished_or_inconsistent_teacher_cannot_export(self) -> None:
        with self.assertRaises(ValueError):
            _final_payload(self.checkpoint())
        broken = self.checkpoint()
        broken["training_complete"] = True
        with self.assertRaises(ValueError):
            self.validate(broken)
        broken = self.checkpoint()
        broken["progress"]["ema_updates"] = 3
        with self.assertRaises(ValueError):
            self.validate(broken)

    def test_final_teacher_contains_only_ema_weights_and_no_optimizer(self) -> None:
        checkpoint = self.checkpoint(epoch=3)
        self.validate(checkpoint)
        final = _final_payload(checkpoint)
        self.assertIn("model_state_dict", final)
        for field in ("raw_model_state_dict", "optimizer_state_dict", "scaler_state_dict", "rng_state"):
            self.assertNotIn(field, final)
        self.assertFalse(final["resume_supported"])
        self.assertTrue(checkpoint["resume_supported"])

    def test_runner_default_never_spawns_or_creates_run(self) -> None:
        args = run_teachers.parse_args(["--fold-plan", str(self.plan), "--run-dir", str(self.root / "new_run")])
        self.assertFalse(args.execute)
        with patch("run_teachers.subprocess.run") as process, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_teachers.run(args), 0)
        process.assert_not_called()
        self.assertFalse(args.run_dir.exists())

    def test_runner_resumes_only_same_fold_last(self) -> None:
        args = run_teachers.parse_args(["--fold-plan", str(self.plan), "--run-dir", str(self.root / "run")])
        directory = args.run_dir / "fold_2"
        directory.mkdir(parents=True)
        (directory / "last.pt").touch()
        command = run_teachers.training_command(args, 2)
        self.assertIn("--resume", command)
        self.assertEqual(command[command.index("--resume") + 1], str(directory / "last.pt"))
        self.assertNotIn("--resume", run_teachers.training_command(args, 1))
        self.assertNotIn("inference", " ".join(command))


if __name__ == "__main__":
    unittest.main()
