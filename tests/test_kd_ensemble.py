"""CPU metadata/math and fake-model tests: no backbone, GPU or data inference."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from compose_students import compose_students, composition_plan
from deploy_kd import (BASE_BACKBONE, TINY_BACKBONE, CLASS_IDS, MEAN, STD, Predictor,
                       aggregate_member_probabilities, deployment_inventory,
                       normalized_model_weights, sha256_file, validate_deployment_config,
                       validate_student_checkpoint)
from export_student import make_deployment_config


def provenance(backbone):
    from kd_common import APPROVED_PRETRAINS, text_sha256
    repo = APPROVED_PRETRAINS[backbone]["repo_id"]
    rev = "a" * 40
    texts = {"model_card": "license: apache-2.0\nSynthetic test fixture",
             "upstream_license": "MIT License\nSynthetic test fixture",
             "apache_license": "Apache License\nSynthetic test fixture"}
    return {**APPROVED_PRETRAINS[backbone], "backbone_name": backbone,
            "revision": rev, "weights_sha256": "b" * 64,
            "weights_url": f"https://huggingface.co/{repo}/resolve/{rev}/model.safetensors",
            "materials": {k: {"text": v, "sha256": text_sha256(v)} for k, v in texts.items()}}


def member(index=0, size=512, tta="hflip2"):
    teachers = [{"fold_index": fold, "checkpoint_sha256": "c" * 64,
                 "selection_protocol": "fixed_epoch_final_ema", "training_complete": True,
                 "schema_version": 6, "artifact_type": "kd_teacher", "weights_source": "ema",
                 "model_name": "convnext_v1_base_gap", "outer_fold_used_for_selection": False,
                 "official_valid_used_for_training": False, "test_used_for_training": False,
                 "epoch": 12, "training_config": {"epochs": 12},
                 "training_subject_ids": ["train_patient"], "oof_subject_ids": ["heldout_patient"],
                 "pretrained_provenance": provenance(BASE_BACKBONE),
                 "source_manifest_sha256": "d" * 64, "fold_plan_sha256": "e" * 64} for fold in range(5)]
    return {"weights_file": f"student_{index:02d}_fp16.pt", "weights_sha256": str(index + 1) * 64,
            "source_checkpoint_sha256": str(index + 5) * 64,
            "model_config": {"backbone_name": TINY_BACKBONE, "gradient_checkpointing": False},
            "image_size": size, "tta": tta, "seed": 43 + index,
            "weights_source": "ema", "pretrained_provenance": provenance(TINY_BACKBONE),
            "teacher_provenance": teachers,
            "preprocessing": {"image_size": size, "validation_size": size,
                              "resize": "letterboxLANCZOS", "color_mode": "RGB",
                              "exif_transpose": True, "mean": list(MEAN), "std": list(STD),
                              "training_scales": [size], "validation_tta": "hflip2"}}


def config(members, weights=None):
    return make_deployment_config(members, source_manifest_sha256="d" * 64,
                                  fold_plan_sha256="e" * 64,
                                  official_valid_manifest_sha256="f" * 64,
                                  model_weights=weights)


class EnsembleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kd-ensemble-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_multires_member_summary_and_probability_weights(self):
        cfg = config([member(0), member(1, 768)], [1.0, 3.0])
        validate_deployment_config(cfg)
        self.assertIsNone(cfg["image_size"])
        self.assertEqual(cfg["image_sizes"], [512, 768])
        np.testing.assert_allclose(cfg["model_weights"], np.array([[0.25] * 10, [0.75] * 10]))
        result = aggregate_member_probabilities([np.full(10, .2), np.full(10, .8)], cfg["model_weights"])
        np.testing.assert_allclose(result, .65)

    def test_classwise_weights_are_frozen_normalized_and_not_votes(self):
        weights = [[1.0] * 5 + [0.0] * 5, [0.0] * 5 + [1.0] * 5]
        cfg = config([member(0), member(1, 768)], weights)
        result = aggregate_member_probabilities([np.full(10, .2), np.full(10, .8)], cfg["model_weights"])
        np.testing.assert_allclose(result, [.2] * 5 + [.8] * 5)
        for bad in ([0., 0.], [-1., 1.], [float("nan"), 1.], [True, 1.], [[True] * 10, [1.] * 10]):
            with self.subTest(weights=bad), self.assertRaises(ValueError):
                normalized_model_weights(bad, 2)

    def test_duplicate_model_and_falsified_summary_are_rejected(self):
        cfg = config([member(0), member(1, 768)])
        for change in ("weights_sha256", "source_checkpoint_sha256", "weights_file"):
            bad = copy.deepcopy(cfg)
            bad["members"][1][change] = bad["members"][0][change]
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_deployment_config(bad)
        for field, value in (("ensemble", False), ("tta", "identity"), ("image_sizes", [512]),
                             ("model_weights", [[1.] * 10, [1.] * 10])):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_deployment_config({**cfg, field: value})

    def test_inventory_rejects_escaping_or_symlink_files(self):
        cfg = config([member(0)])
        (self.root / "student_00_fp16.pt").write_bytes(b"synthetic weights")
        cfg["members"][0]["weights_sha256"] = sha256_file(self.root / "student_00_fp16.pt")
        (self.root / "config.json").write_text(json.dumps(cfg))
        inventory = deployment_inventory(self.root, cfg)
        self.assertEqual(len(inventory["files"]), 2)
        (self.root / "student_00_fp16.pt").unlink()
        outside = self.root.parent / (self.root.name + "_outside")
        outside.write_bytes(b"synthetic")
        self.addCleanup(lambda: outside.unlink(missing_ok=True))
        (self.root / "student_00_fp16.pt").symlink_to(outside)
        with self.assertRaises(ValueError):
            deployment_inventory(self.root, cfg)

    def test_composer_preview_then_new_verified_directory_without_model_load(self):
        sources = []
        for index, size in enumerate((512, 768)):
            root = self.root / f"source{index}"
            root.mkdir()
            m = member(index, size)
            (root / m["weights_file"]).write_bytes(f"synthetic fake fp16 {index}".encode())
            m["weights_sha256"] = sha256_file(root / m["weights_file"])
            (root / "config.json").write_text(json.dumps(config([m])))
            sources.append(root)
        output = self.root / "composed"
        preview = compose_students(sources, output=output)
        self.assertFalse(preview["execute"])
        self.assertFalse(output.exists())
        report = compose_students(sources, output=output, execute=True)
        self.assertEqual(report["member_count"], 2)
        self.assertTrue(report["size_pass"])
        loaded = json.loads((output / "config.json").read_text())
        validate_deployment_config(loaded)
        with self.assertRaises(ValueError):
            composition_plan([sources[0], sources[0]], output=self.root / "duplicate")
        with self.assertRaises(FileExistsError):
            compose_students(sources, output=output, execute=True)

    def test_fake_predictor_decodes_once_and_averages_sigmoid_per_view(self):
        import torch
        image_path = self.root / "image.png"
        Image.new("RGB", (4, 2), (255, 100, 20)).save(image_path)
        calls = []
        class FakeModel:
            def __init__(self, size, values):
                self.size, self.values, self.index = size, values, 0
            def __call__(self, tensor):
                calls.append((self.size, tuple(tensor.shape)))
                value = self.values[self.index]
                self.index += 1
                return torch.full((1, 10), value, dtype=torch.float32)
        predictor = Predictor.__new__(Predictor)
        predictor.torch = torch
        predictor.device = torch.device("cpu")
        predictor.members = [member(0, 512), member(1, 768, "identity")]
        predictor.models = [FakeModel(512, [0., 2.]), FakeModel(768, [-1.])]
        predictor.model_weights = [[.25] * 10, [.75] * 10]
        from inference_inputs import ExactU8Input
        predictor.input_lookup = ExactU8Input(torch, predictor.device, MEAN, STD)
        with patch("deploy_kd.Image.open", wraps=Image.open) as opened:
            probabilities = predictor.predict_image(image_path)
        expected = .25 * (.5 + 1 / (1 + np.exp(-2))) / 2 + .75 / (1 + np.exp(1))
        np.testing.assert_allclose(probabilities, expected, rtol=0, atol=1e-7)
        self.assertEqual(opened.call_count, 1)
        self.assertEqual(calls, [(512, (1, 3, 512, 512)), (512, (1, 3, 512, 512)), (768, (1, 3, 768, 768))])

    def test_fp16_export_legacy_and_new_tta_are_declared_without_real_model(self):
        import torch
        from export_student import export_student
        class FakeShapeVerifier:
            def load_state_dict(self, state, strict):
                self.assertion = (strict and state["synthetic"].dtype == torch.float16)
                if not self.assertion:
                    raise ValueError("Synthetic serialization verifier failed")
        m = member(0, 512)
        checkpoint = {"schema_version": 7, "artifact_type": "kd_student",
                      "model_name": "convnext_v1_tiny_gap", "model_config": m["model_config"],
                      "class_ids": CLASS_IDS, "image_size": 512, "weights_source": "ema",
                      "validation_complete": True, "inference_ready": True,
                      "preprocessing": {**m["preprocessing"], "validation_tta": "identity"},
                      "pretrained_provenance": m["pretrained_provenance"],
                      "teacher_provenance": m["teacher_provenance"],
                      "source_manifest_sha256": "d" * 64, "fold_plan_sha256": "e" * 64,
                      "official_valid_manifest_sha256": "f" * 64,
                      "training_config": {"seed": 43}, "model_state_dict": {"synthetic": torch.ones(2)},
                      "distillation": {"deployment_tta": "identity", "deployment_ensemble": False}}
        source = self.root / "synthetic_checkpoint.pt"
        torch.save(checkpoint, source)
        with patch("model_overnight.build_overnight_model", return_value=FakeShapeVerifier()):
            export_student(source, self.root / "legacy_export")
            export_student(source, self.root / "tta_export", tta="hflip2")
        legacy = json.loads((self.root / "legacy_export/config.json").read_text())
        enhanced = json.loads((self.root / "tta_export/config.json").read_text())
        self.assertEqual((legacy["schema_version"], legacy["tta"]), (1, "identity"))
        self.assertEqual((enhanced["schema_version"], enhanced["tta"]), (2, "hflip2"))
        validate_deployment_config(legacy)
        validate_deployment_config(enhanced)
        self.assertEqual(torch.load(self.root / "tta_export/student_fp16.pt", weights_only=True)["synthetic"].dtype, torch.float16)
        (self.root / "tta_export/student_fp16.pt").write_bytes(b"modified synthetic file")
        with self.assertRaises(ValueError):
            deployment_inventory(self.root / "tta_export", enhanced)

    def test_schema8_export_protocol_binds_resolution_and_view_distillation(self):
        m = member(0, 768)
        ck = {"schema_version": 8, "artifact_type": "kd_student", "model_name": "convnext_v1_tiny_gap",
              "model_config": m["model_config"], "class_ids": CLASS_IDS, "image_size": 768,
              "weights_source": "ema", "validation_complete": True, "inference_ready": True,
              "preprocessing": m["preprocessing"], "pretrained_provenance": m["pretrained_provenance"],
              "teacher_provenance": m["teacher_provenance"], "source_manifest_sha256": "d" * 64,
              "fold_plan_sha256": "e" * 64, "validation_tta": "hflip2",
              "training_config": {"image_size": 768, "validation_tta": "hflip2"},
              "distillation": {"student_training_size": 768, "validation_tta": "hflip2",
                               "deployment_tta": "hflip2", "deployment_ensemble": False,
                               "teacher_tta": "identity_hflip", "teacher_views": ["identity", "hflip"],
                               "soft_target_aggregation": "mean_sigmoid_after_pos_weight_alignment_and_temperature"}}
        validate_student_checkpoint(ck)
        for field, value in (("image_size", 512), ("validation_tta", "identity")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_student_checkpoint({**ck, field: value})


if __name__ == "__main__":
    unittest.main()
