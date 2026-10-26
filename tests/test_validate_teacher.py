"""Tiny metadata and mocked evaluation fixtures; no real checkpoint/model/GPU."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_teacher as diagnostic
from kd_common import APPROVED_PRETRAINS, text_sha256


def fixture_provenance():
    revision = "d" * 40
    materials = {"model_card":"license: apache-2.0\nSynthetic unit fixture",
                 "apache_license":"Apache License\nSynthetic unit fixture",
                 "upstream_license":"MIT License\nSynthetic unit fixture"}
    return {**APPROVED_PRETRAINS[diagnostic.BASE_BACKBONE],
            "backbone_name":diagnostic.BASE_BACKBONE, "revision":revision,
            "weights_sha256":"c" * 64,
            "weights_url":f"https://huggingface.co/timm/{diagnostic.BASE_BACKBONE}/resolve/{revision}/model.safetensors",
            "materials":{k:{"text":v,"sha256":text_sha256(v)} for k,v in materials.items()}}


class TeacherDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="teacher-diagnostic-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = [{"image_id":"a", "subject_id":"p0", "study_key":"p0/s0"},
                        {"image_id":"b", "subject_id":"p1", "study_key":"p1/s1"}]
        self.plan = {"source_manifest_sha256":"a" * 64,
                     "source_train_manifest":{"path":str(self.root/"train.csv")},
                     "study_to_fold":{"p0/s0":0,"p1/s1":1}}

    def checkpoint(self, epoch=13):
        complete = epoch == 20
        return {"schema_version":6, "artifact_type":"kd_teacher", "model_name":diagnostic.MODEL_NAME,
                "model_config":{"backbone_name":diagnostic.BASE_BACKBONE,"gradient_checkpointing":True},
                "class_ids":list(range(1,11)), "image_size":768, "fold_index":0,
                "epoch":epoch, "training_config":{"epochs":20},
                "fold_plan_sha256":"b" * 64, "source_manifest_sha256":"a" * 64,
                "selection_protocol":"fixed_epoch_final_ema", "weights_source":"ema",
                "training_complete":complete, "inference_ready":complete,
                "outer_fold_used_for_selection":False, "official_valid_used_for_training":False,
                "test_used_for_training":False,
                "progress":{"completed_epochs":epoch,"optimizer_updates":100,"ema_updates":100},
                "training_subject_ids":["p1"], "oof_subject_ids":["p0"],
                "training_image_ids":["b"], "oof_image_ids":["a"],
                "preprocessing":{"image_size":768,"validation_size":768,"resize":"letterboxLANCZOS",
                    "validation_tta":"identity","color_mode":"RGB","exif_transpose":True,
                    "data_source":"raw_readonly","mean":[0.485,0.456,0.406],"std":[0.229,0.224,0.225]},
                "pretrained_provenance":fixture_provenance(), "model_state_dict":{"fixture":"EMA"}}

    def validate(self, value):
        return diagnostic.validate_teacher_metadata(value, plan=self.plan, plan_sha256="b" * 64,
                                                    train_records=self.records)

    def test_intermediate_checkpoint_is_diagnostic_without_promoting_completion(self):
        source = self.checkpoint()
        before = copy.deepcopy(source)
        report = self.validate(source)
        self.assertEqual(source,before)
        self.assertEqual(report["epoch"],13)
        self.assertFalse(report["inference_ready"])
        self.assertFalse(report["training_complete"])

    def test_complete_teacher_is_accepted_without_optimizer_state(self):
        source = self.checkpoint(20)
        self.assertTrue(self.validate(source)["training_complete"])

    def test_inconsistent_completion_or_other_source_is_rejected(self):
        changes = [("training_complete",True),("inference_ready",True),("schema_version",5),
                   ("fold_index",1),("source_manifest_sha256","f"*64),
                   ("official_valid_used_for_training",True),("outer_fold_used_for_selection",True),
                   ("test_used_for_training",True)]
        for key,value in changes:
            with self.subTest(key=key),self.assertRaises(ValueError):
                checkpoint=self.checkpoint();checkpoint[key]=value;self.validate(checkpoint)

    def test_fold_membership_and_preprocessing_cannot_be_substituted(self):
        checkpoint=self.checkpoint();checkpoint["training_subject_ids"]=["p0","p1"]
        with self.assertRaises(ValueError):self.validate(checkpoint)
        checkpoint=self.checkpoint();checkpoint["preprocessing"]["validation_size"]=512
        with self.assertRaises(ValueError):self.validate(checkpoint)
        checkpoint=self.checkpoint();checkpoint["pretrained_provenance"]["license"]="cc-by-nc-4.0"
        with self.assertRaises(ValueError):self.validate(checkpoint)

    def test_official_binding_uses_exact_valid_and_registered_training_bytes(self):
        manifest=self.root/"valid.csv";manifest.write_text("official valid fixture")
        train=self.root/"train.csv";train.write_text("approved train fixture")
        valid_sha,train_sha=diagnostic.sha256_file(manifest),diagnostic.sha256_file(train)
        plan=self.root/"folds.json"
        plan.write_text(json.dumps({"n_splits":5,"class_ids":list(range(1,11)),
            "official_holdout":{"path":str(manifest),"sha256":valid_sha},
            "source_train_manifest":{"path":str(train),"sha256":train_sha},
            "source_manifest_sha256":train_sha}))
        with patch.object(diagnostic,"APPROVED_VALID_SHA256",valid_sha),patch.object(
                diagnostic,"APPROVED_TRAIN_SHA256",train_sha):
            self.assertEqual(diagnostic.official_binding(manifest,plan)["manifest_sha256"],valid_sha)
            manifest.write_text("a train/test/outer-fold substitute")
            with self.assertRaises(ValueError):diagnostic.official_binding(manifest,plan)

    def test_preview_does_not_open_checkpoint_or_create_output(self):
        checkpoint=self.root/"epoch13.pt";checkpoint.write_bytes(b"tiny fixture")
        output=self.root/"report"
        binding={"manifest_sha256":"v"*64,"fold_plan_sha256":"p"*64}
        original=Path.open
        def guard(path,*args,**kwargs):
            if path==checkpoint:raise AssertionError("Preview opened a checkpoint")
            return original(path,*args,**kwargs)
        with patch.object(diagnostic,"official_binding",return_value=binding),patch.object(
                Path,"open",guard),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(diagnostic.main(["--checkpoint",str(checkpoint),"--output-dir",str(output)]),0)
        self.assertFalse(output.exists())

    def test_execute_is_refused_before_gpu_or_checkpoint_when_training_active(self):
        with patch.object(diagnostic,"active_training_processes",return_value=[{"pid":123}]),patch.dict(
                sys.modules,{"torch":None}),self.assertRaisesRegex(RuntimeError,"active training"):
            diagnostic.evaluate_checkpoints(checkpoints=["never-read.pt"],manifest="none",fold_plan="none",
                image_root="none",output_dir="none")

    def test_proc_guard_matches_script_arguments_without_reading_weights(self):
        proc=self.root/"proc";proc.mkdir()
        for pid,command in (("987001",b"python\0/project/train_teacher.py\0--execute\0"),
                            ("987002",b"python\0/project/validate_teacher.py\0--execute\0")):
            p=proc/pid;p.mkdir();(p/"cmdline").write_bytes(command)
        self.assertEqual(diagnostic.active_training_processes(proc),[{"pid":987001,"scripts":["train_teacher.py"]}])

    def report(self,epoch,auc):
        return {"complete":True,"checkpoint":f"epoch{epoch}.pt","checkpoint_sha256":str(epoch)*32,
                "epoch":epoch,"macro_auc":auc,"fold":0,"manifest_sha256":"v"*64,
                "fold_plan_sha256":"p"*64,"image_size":768,"precision":"bfloat16",
                "tta":"identity","aggregation":"study_mean_probability","num_images":4,"num_studies":2,
                "training_complete":epoch==20,"inference_ready":epoch==20}

    def test_comparison_is_descriptive_and_rejects_different_protocol(self):
        reports=[self.report(13,.8),self.report(20,.802)]
        result=diagnostic.compare_reports(reports)
        self.assertAlmostEqual(result["results"][1]["macro_auc_delta_from_reference"],.002)
        self.assertFalse(result["checkpoint_selection_performed"])
        self.assertFalse(result["thresholds_fitted"])
        reports[1]["manifest_sha256"]="other"
        with self.assertRaises(ValueError):diagnostic.compare_reports(reports)

    def test_execute_cli_keeps_arguments_and_uses_complete_console_wrapper(self):
        arguments=["--checkpoint","epoch13.pt","epoch20.pt","--output-dir",str(self.root/"report"),"--execute"]
        fake=types.SimpleNamespace(run_logged=lambda args:9)
        with patch.dict(os.environ,{},clear=True),patch.dict(sys.modules,{"training_evidence_monitor":fake}),patch.object(
                fake,"run_logged",return_value=9) as logged:
            self.assertEqual(diagnostic.main(arguments),9)
        self.assertEqual(logged.call_args.args[0].command[3:],arguments)
        self.assertEqual(logged.call_args.args[0].evidence_dir,self.root/"report/evidence")

    def test_complete_mocked_evaluation_writes_report_and_original_flags(self):
        import numpy as np
        image_root=self.root/"images";image_root.mkdir()
        source_root=self.root/"source";source_root.mkdir()
        (source_root/"validate_teacher.py").write_text("# source fixture\n")
        checkpoint_path=self.root/"epoch13.pt";checkpoint_path.write_bytes(b"tiny fixture")
        manifest=self.root/"valid.csv";manifest.write_text("fixture")
        plan_path=self.root/"folds.json";plan_path.write_text("fixture")
        checkpoint=self.checkpoint()
        binding={"manifest":str(manifest),"manifest_sha256":diagnostic.sha256_file(manifest),
                 "fold_plan_sha256":diagnostic.sha256_file(plan_path)}
        checkpoint["fold_plan_sha256"]=binding["fold_plan_sha256"]
        records=[{"image_id":"v0"},{"image_id":"v1"}]
        values={"study_keys":["v0/s0","v1/s1"],"subject_ids":["v0","v1"],"study_ids":["s0","s1"],
                "targets":np.asarray([[0]*10,[1]*10]),"probabilities":np.asarray([[.1]*10,[.9]*10]),
                "metrics":{"macro_auc":1.,"per_class_auc":[1.]*10,"tta":"identity768"}}
        model=types.SimpleNamespace(load_state_dict=lambda *a,**k:None)
        model.eval=lambda:model;model.requires_grad_=lambda *a:model;model.to=lambda *a,**k:model
        cuda=types.SimpleNamespace(is_available=lambda:True,set_device=lambda *a:None,
            is_bf16_supported=lambda:True,manual_seed_all=lambda *a:None,empty_cache=lambda:None,
            get_device_name=lambda *a:"fake GPU, no hardware calls")
        torch=types.SimpleNamespace(cuda=cuda,device=lambda *a:"fake",bfloat16="bf16",float16="fp16",
            set_num_threads=lambda *a:None,manual_seed=lambda *a:None,channels_last="fake",__version__="fixture",
            backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=True)),
            load=lambda *a,**k:copy.deepcopy(checkpoint))
        modules={"torch":torch,"cv_splits":types.SimpleNamespace(read_fold_plan=lambda *a:self.plan),
            "dataset":types.SimpleNamespace(read_manifest=lambda path,split:self.records if split=="train" else records),
            "dataset_overnight":types.SimpleNamespace(make_overnight_loader=lambda *a,**k:[]),
            "model_overnight":types.SimpleNamespace(build_overnight_model=lambda *a,**k:model),
            "train_overnight":types.SimpleNamespace(evaluate=lambda *a:values)}
        with patch.dict(sys.modules,modules),patch.object(diagnostic,"active_training_processes",return_value=[]),patch.object(
                diagnostic,"official_binding",return_value=binding),patch.object(diagnostic,"PROJECT",source_root),contextlib.redirect_stdout(io.StringIO()):
            result=diagnostic.evaluate_checkpoints(checkpoints=[checkpoint_path],manifest=manifest,fold_plan=plan_path,
                image_root=image_root,output_dir=self.root/"result",num_workers=0)
        self.assertTrue(result["complete"])
        self.assertFalse(result["results"][0]["inference_ready"])
        self.assertEqual(result["reports"][0]["tta"],"identity")
        self.assertEqual(len((self.root/"result/checkpoint_0_epoch_13.csv").read_text().splitlines()),21)
        self.assertTrue((self.root/"result/comparison_report.json").is_file())


if __name__=="__main__":unittest.main()
