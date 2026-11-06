"""Independent BCE ancestry, export and mixed ensembles using tiny CPU fixtures."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from compose_students import compose_students
from deploy_kd import (BASE_BACKBONE, TINY_BACKBONE, CLASS_IDS, MEAN, STD,
                       deployment_inventory, sha256_file, validate_deployment_config,
                       validate_provenance, validate_student_checkpoint)
from export_student import export_student, make_deployment_config


def provenance(backbone):
    from kd_common import APPROVED_PRETRAINS, text_sha256
    approved = APPROVED_PRETRAINS[backbone]
    revision = "a" * 40
    texts = {"model_card": "license: apache-2.0\nSynthetic test fixture",
             "upstream_license": "MIT License\nSynthetic test fixture",
             "apache_license": "Apache License\nSynthetic test fixture"}
    return {**approved, "backbone_name": backbone, "revision": revision,
            "weights_sha256": "b" * 64,
            "weights_url": f"https://huggingface.co/{approved['repo_id']}/resolve/{revision}/model.safetensors",
            "materials": {k: {"text": v, "sha256": text_sha256(v)} for k, v in texts.items()}}


def supervised_checkpoint(size=512, tta="hflip2"):
    return {"schema_version": 9, "artifact_type": "kd_student", "model_name": "convnext_v1_tiny_gap",
        "training_mode": "supervised_bce", "model_config": {"backbone_name": TINY_BACKBONE,
                                                               "gradient_checkpointing": True},
        "class_ids": CLASS_IDS, "image_size": size, "weights_source": "ema",
        "validation_complete": True, "inference_ready": True,
        "validation_tta": tta, "deployment_tta": tta, "deployment_ensemble": False,
        "preprocessing": {"image_size": size, "validation_size": size, "resize": "letterboxLANCZOS",
            "exif_transpose": True, "color_mode": "RGB", "mean": list(MEAN), "std": list(STD),
            "training_scales": [size], "validation_tta": tta},
        "pretrained_provenance": provenance(TINY_BACKBONE), "teacher_provenance": [],
        "source_manifest_sha256": "d" * 64, "fold_plan_sha256": "e" * 64,
        "official_valid_manifest_sha256": "f" * 64,
        "training_config": {"training_mode": "supervised_bce", "alpha": 0., "soft_labels": None,
            "image_size": size, "validation_tta": tta, "seed": 43},
        "distillation": {"enabled": False, "alpha": 0., "loss": "weighted_bce",
            "class_reduction": "mean", "study_equal_weighting": True, "student_training_size": size,
            "validation_tta": tta, "deployment_tta": tta, "deployment_ensemble": False}}


def kd_member(index=1, size=768):
    ck = supervised_checkpoint(size)
    teachers = [{"fold_index": fold, "checkpoint_sha256": "c" * 64,
        "selection_protocol": "fixed_epoch_final_ema", "training_complete": True,
        "schema_version": 6, "artifact_type": "kd_teacher", "weights_source": "ema",
        "model_name": "convnext_v1_base_gap", "outer_fold_used_for_selection": False,
        "official_valid_used_for_training": False, "test_used_for_training": False,
        "epoch": 12, "training_config": {"epochs": 12}, "training_subject_ids": ["train_patient"],
        "oof_subject_ids": ["heldout_patient"], "pretrained_provenance": provenance(BASE_BACKBONE),
        "source_manifest_sha256": "d" * 64, "fold_plan_sha256": "e" * 64} for fold in range(5)]
    return {"weights_file": f"student_{index:02d}_fp16.pt", "weights_sha256": str(index + 1) * 64,
        "source_checkpoint_sha256": str(index + 5) * 64, "model_config": {"backbone_name": TINY_BACKBONE,
            "gradient_checkpointing": False}, "image_size": size, "tta": "hflip2", "seed": 44,
        "weights_source": "ema", "pretrained_provenance": ck["pretrained_provenance"],
        "teacher_provenance": teachers, "preprocessing": ck["preprocessing"]}


def supervised_member(index=0, size=512):
    ck = supervised_checkpoint(size)
    return {"weights_file": f"student_{index:02d}_fp16.pt", "weights_sha256": str(index + 1) * 64,
        "source_checkpoint_sha256": str(index + 5) * 64, "model_config": {"backbone_name": TINY_BACKBONE,
            "gradient_checkpointing": False}, "image_size": size, "tta": "hflip2", "seed": 43,
        "weights_source": "ema", "training_mode": "supervised_bce",
        "pretrained_provenance": ck["pretrained_provenance"], "teacher_provenance": [],
        "distillation": ck["distillation"], "preprocessing": ck["preprocessing"]}


def configuration(members):
    return make_deployment_config(members, source_manifest_sha256="d" * 64,
        fold_plan_sha256="e" * 64, official_valid_manifest_sha256="f" * 64)


class SupervisedDeploymentTests(unittest.TestCase):
    def test_schema9_accepts_explicit_independent_bce_for_supported_sizes_and_views(self):
        for size in (512, 768):
            for tta in ("identity", "hflip2"):
                with self.subTest(size=size, tta=tta):
                    validate_student_checkpoint(supervised_checkpoint(size, tta))

    def test_independent_bce_cannot_hide_teachers_cache_or_kl(self):
        changes = [("training_mode", "distillation"), ("schema_version", 8),
                   ("teacher_provenance", [{}]), ("teacher_provenance", None)]
        for field, value in changes:
            bad = supervised_checkpoint()
            bad[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_student_checkpoint(bad)
        for field, value in (("enabled", True), ("alpha", .2), ("alpha", False),
                             ("loss", "bernoulli_kl_T2_plus_weighted_bce"),
                             ("soft_labels_sha256", "a" * 64), ("teacher_views", ["identity"]),
                             ("study_equal_weighting", False), ("student_training_size", 768)):
            bad = supervised_checkpoint()
            bad["distillation"][field] = value
            with self.subTest(protocol_field=field), self.assertRaises(ValueError):
                validate_student_checkpoint(bad)
        for field, value in (("training_mode", "distillation"), ("alpha", .2),
                             ("soft_labels", "unexpected-cache"), ("image_size", 768)):
            bad = supervised_checkpoint()
            bad["training_config"][field] = value
            with self.subTest(training_field=field), self.assertRaises(ValueError):
                validate_student_checkpoint(bad)

    def test_no_teacher_permission_does_not_apply_to_legacy_kd_or_unlicensed_student(self):
        p = provenance(TINY_BACKBONE)
        validate_provenance(p, [], training_mode="supervised_bce")
        for kwargs in ({}, {"training_mode": "distillation"}, {"training_mode": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                validate_provenance(p, [], **kwargs)
        with self.assertRaises(ValueError):
            validate_provenance({**p, "license": "cc-by-nc-4.0"}, [], training_mode="supervised_bce")

    def test_fp16_export_keeps_supervised_ancestry_without_loading_actual_backbone(self):
        import torch
        class FakeShapeVerifier:
            def load_state_dict(self, state, strict):
                if strict is not True or state["synthetic"].dtype != torch.float16:
                    raise ValueError("Unexpected synthetic export")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ck = supervised_checkpoint(768)
            ck["model_state_dict"] = {"synthetic": torch.tensor([1., 2.])}
            source = root / "checkpoint.pt"
            torch.save(ck, source)
            with patch("model_overnight.build_overnight_model", return_value=FakeShapeVerifier()) as builder:
                export_student(source, root / "export", tta="identity")
            self.assertFalse(builder.call_args.kwargs["pretrained"])
            cfg = json.loads((root / "export/config.json").read_text())
            validate_deployment_config(cfg)
            self.assertEqual(cfg["schema_version"], 2)
            self.assertEqual(cfg["training_mode"], "supervised_bce")
            self.assertEqual(cfg["teacher_provenance"], [])
            self.assertEqual(cfg["members"][0]["teacher_provenance"], [])
            self.assertEqual(cfg["members"][0]["source_checkpoint_schema_version"], 9)
            self.assertEqual(cfg["tta"], "identity")
            self.assertEqual(cfg["members"][0]["distillation"]["validation_tta"], "hflip2")
            state = torch.load(root / "export/student_fp16.pt", weights_only=True)
            self.assertEqual(state["synthetic"].dtype, torch.float16)
            self.assertTrue(deployment_inventory(root / "export", cfg)["size_pass"])

    def test_composition_preserves_each_member_mode_and_handles_empty_teacher_ancestry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dirs = []
            for index, member in enumerate((supervised_member(), kd_member())):
                source = root / f"source{index}"
                source.mkdir()
                weights = source / member["weights_file"]
                weights.write_bytes(f"synthetic weight {index}".encode())
                member["weights_sha256"] = sha256_file(weights)
                (source / "config.json").write_text(json.dumps(configuration([member])))
                dirs.append(source)
            preview = compose_students(dirs, output=root / "composed")
            self.assertFalse(preview["execute"])
            self.assertFalse((root / "composed").exists())
            compose_students(dirs, output=root / "composed", execute=True)
            cfg = json.loads((root / "composed/config.json").read_text())
            self.assertEqual(cfg["members"][0]["training_mode"], "supervised_bce")
            self.assertEqual(cfg["members"][0]["teacher_provenance"], [])
            self.assertEqual(len(cfg["members"][1]["teacher_provenance"]), 5)
            self.assertEqual(cfg["members"][1].get("training_mode", "distillation"), "distillation")
            self.assertEqual(cfg["training_mode"], "supervised_bce")
            validate_deployment_config(cfg)
            self.assertTrue(deployment_inventory(root / "composed", cfg)["size_pass"])

    def test_supervised_member_requires_explicit_mode_and_consistent_representative(self):
        cfg = configuration([supervised_member(), kd_member()])
        bad = copy.deepcopy(cfg)
        bad["members"][0].pop("training_mode")
        with self.assertRaises(ValueError):
            validate_deployment_config(bad)
        bad = copy.deepcopy(cfg)
        bad["training_mode"] = "distillation"
        with self.assertRaises(ValueError):
            validate_deployment_config(bad)
        bad = copy.deepcopy(cfg)
        bad["members"][1]["teacher_provenance"] = []
        with self.assertRaises(ValueError):
            validate_deployment_config(bad)
        bad = copy.deepcopy(cfg)
        bad["members"][0]["source_checkpoint_schema_version"] = 8
        with self.assertRaises(ValueError):
            validate_deployment_config(bad)
        bad = copy.deepcopy(cfg)
        bad["members"][0]["source_manifest_sha256"] = "a" * 64
        with self.assertRaises(ValueError):
            validate_deployment_config(bad)


if __name__ == "__main__":
    unittest.main()
