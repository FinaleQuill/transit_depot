"""CPU-only synthetic/mocked protocol tests: no torch import, GPU or model work."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_local_resources as bench
import cost
from deploy_kd import sha256_file


class LocalResourceBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="local-resource-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "student_fp16.pt").write_bytes(b"synthetic weights, not a real model")
        config = {"inference_files": ["student_fp16.pt", "config.json"],
                  "image_size": 512, "tta": "hflip2", "ensemble": False,
                  "training_mode": "distillation", "weights_source": "ema"}
        (self.model / "config.json").write_text(json.dumps(config))
        self.manifest = self.root / "manifest.csv"
        rows, groups, full_counts = [], {}, {}
        for name, count, subject in (("measured", 100, "subject_1"), ("warmup", 5, "subject_2")):
            directory = self.root / ("pictures_100" if name == "measured" else "pictures_5")
            entries = []
            for index in range(count):
                study = f"study_{index // 5:03d}"
                relative = f"valid/{subject}/{study}/image_{index:03d}.jpg"
                output = directory / subject / study / f"image_{index:03d}.jpg"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(f"synthetic {name} {index}".encode())
                digest = sha256_file(output)
                record = {"image_id": f"valid:{subject}:{index}", "split": "valid",
                          "subject_id": subject, "study_id": study,
                          "study_key": f"{subject}/{study}", "relative_path": relative,
                          "file_sha256": digest}
                rows.append(record)
                full_counts[record["study_key"]] = full_counts.get(record["study_key"], 0) + 1
                entries.append({**record, "sha256": digest})
            groups[name] = entries
        with self.manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.selection = self.root / "selection.json"
        self.selection_data = {"schema_version": 1, "origin_split": "valid",
                               "manifest_path": str(self.manifest),
                               "manifest_sha256": sha256_file(self.manifest),
                               "full_study_image_counts": full_counts, **groups,
                               "counts": {"measured_images": 100, "warmup_images": 5,
                                          "measured_studies": 20, "warmup_studies": 1,
                                          "measured_patients": 1, "warmup_patients": 1}}
        self.save_selection()
        self.arguments = dict(model_dir=self.model, pictures_100=self.root / "pictures_100",
                              pictures_5=self.root / "pictures_5", selection_json=self.selection,
                              output_dir=self.root / "output", profile="synthetic_native")
        self.events = []
        self.cuda = SimpleNamespace(
            is_available=lambda: True, init=lambda: self.events.append("cuda_init"),
            get_device_properties=lambda device: SimpleNamespace(
                multi_processor_count=128, total_memory=24 * 1024 ** 3, name="Synthetic T4"),
            set_per_process_memory_fraction=lambda fraction, device: self.events.append(("cap", fraction, device)),
            reset_peak_memory_stats=lambda device: self.events.append("reset_peak"),
            max_memory_allocated=lambda device: 123456,
            max_memory_reserved=lambda device: 234567)
        self.torch = SimpleNamespace(cuda=self.cuda)

    def save_selection(self):
        self.selection.write_text(json.dumps(self.selection_data))

    def patches(self):
        events, torch = self.events, self.torch
        class FakePredictor:
            config = {}
            inventory = {}
            device = "cuda:0"
            def __init__(self, model_dir):
                events.append("model_init")
                self.torch = torch
            def synchronize(self):
                events.append("sync")
            def predict_image(self, path):
                events.append(("predict", Path(path).name))
                return np.linspace(0.1, 0.9, 10)
        return patch.object(bench, "Predictor", FakePredictor)

    def run_mock(self, *, requested=None, time_round=None):
        real_time_round = cost.time_round
        calls = []
        def fake_round(predictor, records, output):
            index = len(calls)
            calls.append(len(records))
            # Exercise shared CSV writer, Study aggregation and prediction path;
            # deterministic clock supplies non-GPU timings of i + 1 + 10*round ms.
            ticks = []
            elapsed = 0.0
            for image in range(len(records)):
                ticks.extend((elapsed, elapsed + (image + 1 + index * 10) / 1000))
                elapsed += 1.0
            iterator = iter(ticks)
            return real_time_round(predictor, records, output, clock=lambda: next(iterator))
        with patch.dict(sys.modules, {"torch": self.torch}), self.patches(), \
                patch.dict("os.environ", {}, clear=True), \
                patch.object(cost, "timing_environment", return_value={"gpu_name": "Synthetic T4", "official_environment_compatible": True}), \
                patch.object(cost, "time_round", side_effect=time_round or fake_round), \
                patch("sys.stdout", new_callable=io.StringIO):
            report = bench.run_benchmark(**self.arguments, execute=True,
                                         requested_thread_percentage=requested)
        return report, calls

    def test_metadata_preview_never_imports_torch_or_initializes_predictor_or_writes(self):
        with patch.object(bench, "Predictor", side_effect=AssertionError("No model allowed")), \
                patch.dict(sys.modules, {"torch": None}):
            report = bench.run_benchmark(**self.arguments)
        self.assertFalse(report["executed"])
        self.assertEqual(report["status"], "metadata_preview")
        self.assertFalse(self.arguments["output_dir"].exists())
        self.assertTrue(report["coverage"]["hashes_verified"])
        self.assertEqual(report["inference_files"]["total_bytes"], sum(p.stat().st_size for p in self.model.iterdir()))

    def test_three_round_aggregation_and_explicit_noncertification_even_on_t4(self):
        report, calls = self.run_mock(requested=100)
        self.assertEqual(calls, [100, 100, 100])
        np.testing.assert_allclose(report["per_image_mean_ms"], np.arange(1, 101) + 10, atol=1e-9)
        self.assertAlmostEqual(report["mean_ms"], 60.5)
        self.assertAlmostEqual(report["median_ms"], 60.5)
        self.assertAlmostEqual(report["p95_ms"], 105.05)
        self.assertAlmostEqual(report["max_ms"], 110)
        np.testing.assert_allclose(report["round_mean_ms"], [50.5, 60.5, 70.5])
        self.assertEqual(np.asarray(report["per_image_round_ms"]).shape, (100, 3))
        self.assertTrue(report["local_stress_test"])
        self.assertFalse(report["T4_latency_certified"])
        self.assertFalse(report["official_compliance_pass"])
        self.assertIsNone(report["T4_elapsed_ms"])
        self.assertEqual(report["comparison"]["status"], "matched")
        self.assertTrue(all(r["absolute_tolerance"] == 1e-7 for r in report["comparison"]["rounds"]))
        self.assertEqual(len([event for event in self.events if isinstance(event, tuple) and event[0] == "predict"]), 305)
        self.assertEqual(report["peak_allocated_bytes"], 123456)
        self.assertEqual(report["peak_reserved_bytes"], 234567)
        written = json.loads((self.arguments["output_dir"] / "local_resource_report.json").read_text())
        self.assertEqual(written, report)

    def test_allocator_cap_precedes_predictor_and_native_sm_recorded(self):
        report, _ = self.run_mock()
        cap_event = ("cap", 16 / 24, 0)
        self.assertIn(cap_event, self.events)
        self.assertLess(self.events.index(cap_event), self.events.index("model_init"))
        self.assertEqual(report["resources"]["cuda_visible_sm_count"], 128)
        self.assertEqual(report["resources"]["native_baseline_sm_count"], 128)
        self.assertEqual(report["resources"]["mode"], "native")
        self.assertEqual(report["resources"]["allocator"]["requested_limit_bytes"], 16 * 1024 ** 3)

    def test_ineffective_mps_quota_fails_before_cap_model_or_output(self):
        with patch.dict(sys.modules, {"torch": self.torch}), self.patches(), \
                patch.dict("os.environ", {"CUDA_MPS_PIPE_DIRECTORY": "/synthetic/pipe",
                                          "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "25"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "ineffective"):
                bench.run_benchmark(**self.arguments, execute=True, requested_thread_percentage=25)
        self.assertEqual(self.events, ["cuda_init"])
        self.assertFalse(self.arguments["output_dir"].exists())

    def test_restricted_quota_requires_matching_mps_environment(self):
        for environment in ({}, {"CUDA_MPS_PIPE_DIRECTORY": "/pipe", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "50"}):
            with self.subTest(environment=environment), patch.dict("os.environ", environment, clear=True):
                with self.assertRaises(RuntimeError):
                    bench.prepare_cuda(self.torch, requested_thread_percentage=25,
                                       baseline_sm_count=128, memory_limit_gib=16)
        self.assertEqual(self.events, [])

    def test_effective_quota_records_mps_and_device_sm_count(self):
        self.cuda.get_device_properties = lambda device: SimpleNamespace(
            multi_processor_count=32, total_memory=24 * 1024 ** 3, name="Synthetic 4090")
        with patch.dict("os.environ", {"CUDA_MPS_PIPE_DIRECTORY": "/pipe",
                                       "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "25"}, clear=True), \
                patch("sys.stdout", new_callable=io.StringIO):
            report = bench.prepare_cuda(self.torch, requested_thread_percentage=25,
                                        baseline_sm_count=128, memory_limit_gib=16)
        self.assertEqual(report["mode"], "mps_client")
        self.assertEqual(report["cuda_visible_sm_count"], 32)
        self.assertEqual(report["requested_thread_percentage"], 25)

    def test_existing_output_rejected_before_cuda_or_input_changes(self):
        self.arguments["output_dir"].mkdir()
        marker = self.arguments["output_dir"] / "keep"
        marker.write_text("original")
        with patch.object(bench, "validate_inputs", side_effect=AssertionError("must reject first")):
            with self.assertRaises(FileExistsError):
                bench.run_benchmark(**self.arguments, execute=True)
        self.assertEqual(marker.read_text(), "original")
        self.assertEqual(self.events, [])

    def test_manifest_hash_identity_and_image_hash_are_independently_checked(self):
        for mutation in ("manifest", "image", "identity", "counts", "selected_counts"):
            with self.subTest(mutation=mutation):
                original = self.selection.read_text()
                if mutation == "manifest":
                    self.selection_data["manifest_sha256"] = "0" * 64
                elif mutation == "image":
                    self.selection_data["measured"][0]["sha256"] = "0" * 64
                elif mutation == "identity":
                    self.selection_data["measured"][0]["image_id"] = "wrong"
                elif mutation == "selected_counts":
                    self.selection_data["counts"]["measured_images"] = 99
                else:
                    self.selection_data["full_study_image_counts"]["subject_1/study_000"] += 1
                self.save_selection()
                with self.assertRaises(ValueError):
                    bench.run_benchmark(**self.arguments)
                self.selection_data = json.loads(original)
                self.save_selection()

    def test_incomplete_study_fails_even_when_selection_and_files_agree(self):
        # Declare a sixth original view, absent from the selected Study.
        with self.manifest.open("a", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["valid:extra", "valid", "subject_1", "study_000", "subject_1/study_000",
                             "valid/subject_1/study_000/extra.jpg", "0" * 64])
        self.selection_data["manifest_sha256"] = sha256_file(self.manifest)
        self.selection_data["full_study_image_counts"]["subject_1/study_000"] = 6
        self.save_selection()
        with self.assertRaisesRegex(ValueError, "omits"):
            bench.run_benchmark(**self.arguments)

    def test_prediction_drift_is_reported_without_certification(self):
        actual = cost.time_round
        calls = []
        def drifting_round(predictor, records, output):
            durations, results = actual(predictor, records, output)
            if calls:
                results = {key: values + 1e-5 for key, values in results.items()}
                from deploy_kd import CSV_HEADER, write_prediction
                with output.open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(CSV_HEADER)
                    for (subject, study), values in results.items():
                        write_prediction(writer, subject, study, values)
            calls.append(True)
            return durations, results
        report, _ = self.run_mock(time_round=drifting_round)
        self.assertEqual(report["comparison"]["status"], "failed")
        self.assertTrue(report["comparison"]["rounds"][0]["match"])
        self.assertFalse(report["comparison"]["rounds"][1]["match"])
        self.assertFalse(report["comparison"]["rounds"][2]["match"])
        self.assertFalse(report["T4_latency_certified"])
        self.assertFalse(report["official_compliance_pass"])

    def test_csv_missing_class_is_rejected(self):
        actual = cost.time_round
        def broken_round(predictor, records, output):
            result = actual(predictor, records, output)
            lines = output.read_text().splitlines()
            output.write_text("\n".join(lines[:-1]) + "\n")
            return result
        with self.assertRaisesRegex(ValueError, "ten classes"):
            self.run_mock(time_round=broken_round)


if __name__ == "__main__":
    unittest.main()
