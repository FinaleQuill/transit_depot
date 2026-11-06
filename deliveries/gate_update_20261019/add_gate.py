#!/usr/bin/env python3
"""Append the saved epoch-5 gates to the existing CSV; no model inference."""
import argparse
import csv
import hashlib
import io
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SOURCE = "c985cf8736ebb342af6c8662acbbb277042fa8ef779524b18751220333969650"
EXPECTED_MODEL = "2c3f88b25e04ea86af20eaf2e41d42db241cb5013657283afa3eabdf54893b4b"
HEADER = ["Subject_id", "Study_id", "Predict_class", "Probability"]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        return next(reader), list(reader)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    out = parser.parse_args().output.resolve()
    source = ROOT / "deliveries/best_auc_20261019/submission.csv"
    inference_dir = ROOT / "deliveries/016_a4000_tiny_v2_inference"
    run = ROOT / "runs/016_a4000_tiny_v2"
    data = ROOT / "cloud_project_v2/data/clean_v1_reviewed"
    selection_path = source.with_name("provenance.json")
    inference_path = inference_dir / "provenance.json"
    thresholds_path = run / "thresholds.json"
    metrics_path = run / "best_metrics.json"
    checkpoint = run / "best.pt"
    inputs = [source, selection_path, inference_path, thresholds_path, metrics_path,
              checkpoint, inference_dir / "submission.csv",
              inference_dir / "study_probabilities.csv", inference_dir / "study_binary.csv",
              ROOT / "submission.csv", data / "train_clean.csv",
              data / "valid_manifest.csv", data / "test_manifest.csv"]
    before = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    selection, proof = read_json(selection_path), read_json(inference_path)
    binding, metrics = read_json(thresholds_path), read_json(metrics_path)
    assert sha(source) == selection["submission_sha256"] == EXPECTED_SOURCE
    assert sha(checkpoint) == binding["checkpoint_sha256"] == metrics["checkpoint_sha256"] == proof["checkpoint_sha256"] == EXPECTED_MODEL
    assert selection["selected"]["run"] == run.name
    assert binding["epoch"] == metrics["epoch"] == proof["checkpoint_epoch"] == 5
    assert binding["class_ids"] == list(range(1, 11))
    gates = binding["thresholds"]
    assert len(gates) == 10 and all(math.isfinite(g) and 0 <= g <= 1 for g in gates)
    assert gates == metrics["thresholds"] == proof["thresholds"]
    assert binding["weights_source"] == metrics["weights_source"] == proof["weights_source"] == proof["validation_weights_source"] == proof["threshold_weights_source"] == "ema"
    assert binding["preprocessing_sha256"] == metrics["preprocessing_sha256"] == proof["preprocessing_sha256"]
    assert binding["inference_precision"] == metrics["inference_precision"] == proof["inference_precision"] == "bfloat16"
    assert binding["aggregation"] == proof["aggregation"] == "study_mean_probability"
    assert metrics["validation_complete"] is True
    for split, name in [("train", "train_clean.csv"), ("valid", "valid_manifest.csv"), ("test", "test_manifest.csv")]:
        assert sha(data / name) == proof[f"{split}_manifest_sha256"]
    for name, digest in proof["files_sha256"].items():
        assert sha(inference_dir / name) == digest

    header, rows = read_csv(source)
    assert header == HEADER and len(rows) == 14140 and all(len(r) == 4 for r in rows)
    sample_header, sample_rows = read_csv(ROOT / "submission.csv")
    assert sample_header == HEADER
    assert [r[:3] for r in rows] == [r[:3] for r in sample_rows]
    raw_header, raw_rows = read_csv(inference_dir / "submission.csv")
    assert raw_header == HEADER[1:]
    assert [r[1:] for r in rows] == raw_rows
    wide_header, wide_rows = read_csv(inference_dir / "study_probabilities.csv")
    assert wide_header == ["study_id"] + [f"class_{c}" for c in range(1, 11)]
    assert len(wide_rows) == 1414
    probabilities = {r[0]: r[1:] for r in wide_rows}
    binary_header, binary_rows = read_csv(inference_dir / "study_binary.csv")
    assert binary_header == wide_header and len(binary_rows) == 1414
    binaries = {r[0]: r[1:] for r in binary_rows}
    with (data / "test_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        manifest_rows = list(csv.DictReader(stream))
    subjects = {}
    for row in manifest_rows:
        study, subject = row["study_id"], row["subject_id"]
        assert subjects.setdefault(study, subject) == subject
    assert len(manifest_rows) == 2218 and len(subjects) == 1414
    seen, study_order = set(), []
    for index, row in enumerate(rows):
        subject, study, class_id, probability = row
        c = int(class_id)
        assert class_id == str(index % 10 + 1)
        assert subject == subjects[study]
        key = study, c
        assert key not in seen
        seen.add(key)
        if index % 10 == 0:
            study_order.append(study)
        p = float(probability)
        assert math.isfinite(p) and 0 <= p <= 1
        assert probability == probabilities[study][c - 1]
        assert int(p >= gates[c - 1]) == int(binaries[study][c - 1])
    assert study_order == list(subjects)
    assert seen == {(study, c) for study in subjects for c in range(1, 11)}

    out.mkdir(parents=True, exist_ok=True)
    targets = [out / name for name in ("submission.csv", "thresholds.json", "validation_report.json", "provenance.json")]
    assert not any(p.exists() for p in targets), "Use a fresh --output directory; existing deliveries are preserved."
    output = out / "submission.csv"
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\r\n")
        writer.writerow(HEADER + ["Gate"])
        writer.writerows(row + [repr(gates[int(row[2]) - 1])] for row in rows)
    result_header, result_rows = read_csv(output)
    assert result_header == HEADER + ["Gate"] and len(result_rows) == len(rows)
    assert [r[:-1] for r in result_rows] == rows
    assert all(len(r) == 5 and float(r[-1]) == gates[int(r[2]) - 1] for r in result_rows)
    original = io.StringIO(newline="")
    writer = csv.writer(original, lineterminator="\r\n")
    writer.writerow(HEADER)
    writer.writerows(r[:-1] for r in result_rows)
    assert hashlib.sha256(original.getvalue().encode("utf-8")).hexdigest() == EXPECTED_SOURCE
    assert not output.read_bytes().startswith(b"\xef\xbb\xbf")
    after = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    assert before == after
    now = datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(timespec="seconds")
    thresholds_copy = {**binding, "original_thresholds_file": str(thresholds_path),
                       "original_thresholds_sha256": sha(thresholds_path)}
    write_json(out / "thresholds.json", thresholds_copy)
    report = {
        "created_at": now, "status": "passed", "submission": str(output),
        "header": result_header, "data_rows": len(result_rows), "studies": len(subjects),
        "classes_per_study": 10, "source_test_images": len(manifest_rows),
        "duplicates": 0, "missing_study_class_pairs": 0,
        "invalid_probabilities": 0, "invalid_gates": 0,
        "original_four_columns_and_row_order_unchanged": True,
        "probability_text_preserved_exactly": True,
        "probabilities_match_original_inference_long_and_wide_tables": True,
        "identity_and_order_match_test_manifest_and_sample": True,
        "gates_match_saved_epoch5_ema_thresholds": True,
        "binary_predictions_match_original_saved_binary_table": True,
        "all_input_files_unchanged": before == after,
        "source_four_columns_reconstructed_sha256": EXPECTED_SOURCE,
        "submission_sha256": sha(output), "encoding": "UTF-8 without BOM", "line_endings": "CRLF",
        "validation_macro_auc": metrics["val"]["macro_auc"],
        "validation_macro_f1_at_default_gate": metrics["val_at_05"]["macro_f1"],
        "validation_macro_f1_at_saved_gates": metrics["val"]["macro_f1"],
        "threshold_evaluation_scope": "same official validation set used for threshold tuning; not independent test performance",
        "additional_training_runs": 0, "additional_model_inference_runs": 0,
        "threshold_refitting_performed": False, "platform_upload_performed": False,
    }
    write_json(out / "validation_report.json", report)
    write_json(out / "provenance.json", {
        "created_at": now, "operation": "append saved per-class Gate; preserve every original cell and row",
        "selected_run": run.name, "checkpoint_epoch": 5, "weights_source": "ema",
        "checkpoint_sha256": EXPECTED_MODEL, "inference_precision": "bfloat16",
        "aggregation": proof["aggregation"], "preprocessing_sha256": proof["preprocessing_sha256"],
        "source_submission": str(source), "source_selection_provenance": str(selection_path),
        "source_inference_provenance": str(inference_path),
        "threshold_source": "saved same-epoch official valid Study-level F1 thresholds; no test fitting",
        "gates_by_class": {str(c): gates[c - 1] for c in range(1, 11)},
        "input_files_sha256": before, "conversion_script_sha256": sha(Path(__file__).resolve()),
        "output_submission_sha256": sha(output),
        "official_rules_url": "https://www.ncccu.org.cn/index/Paper/case1.html",
        "official_example_url": "https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261019/4dc233268b61702b4343f5a2d6385a78.png",
        "official_rules_checked_on": "2026-10-19 Asia/Hong_Kong",
        "gate_effect": "binary decisions for Macro-F1; continuous probabilities and their Macro-AUC unchanged",
        "platform_upload_performed": False,
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
