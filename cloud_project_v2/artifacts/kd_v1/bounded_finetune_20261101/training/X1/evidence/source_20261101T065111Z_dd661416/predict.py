"""Generate complete Study probabilities using the fixed exported FP16 student."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from deploy_kd import (CSV_HEADER, IMAGE_CSV_HEADER, Predictor, StudyAccumulator, prediction_records,
                       write_prediction, write_image_prediction, write_json, sha256_file, deployment_members)
from inference_reproducibility import configure_inference


def predict_submission(*, model_dir: str | Path, image_root: str | Path,
                       output: str | Path, manifest: str | Path | None = None,
                       predictor_factory=None) -> dict:
    output = Path(output).resolve()
    if output.name != "submission.csv":
        raise ValueError("Official output filename must be submission.csv")
    if output.exists():
        raise FileExistsError("Choose a new submission output file")
    if output.is_relative_to(Path(image_root).resolve()):
        raise ValueError("Prediction output must not be inside the read-only image directory")
    records = prediction_records(image_root, manifest)
    reproducibility = configure_inference()
    predictor = (predictor_factory or Predictor)(model_dir)
    accumulator = StudyAccumulator(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    image_output = output.with_suffix(".image_predictions.csv")
    image_temporary = image_output.with_name(image_output.name + ".tmp")
    if temporary.exists() or image_output.exists() or image_temporary.exists():
        raise FileExistsError(temporary)
    count = 0
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream, \
                image_temporary.open("x", encoding="utf-8", newline="") as image_stream:
            writer = csv.writer(stream)
            image_writer = csv.writer(image_stream)
            gates = predictor.config.get("gates")
            writer.writerow(CSV_HEADER + (["Gate"] if gates is not None else []))
            image_writer.writerow(IMAGE_CSV_HEADER)
            for record in records:
                probability = predictor.predict_image(record["path"])
                write_image_prediction(image_writer, record, probability)
                image_stream.flush()
                result = accumulator.add(record, probability)
                if result is not None:
                    write_prediction(writer, *result, gates=gates)
                    stream.flush()
                    count += 1
        if not accumulator.complete() or count != len(accumulator.expected):
            raise RuntimeError("Incomplete Study coverage; no final submission will be published")
        temporary.rename(output)
        image_temporary.rename(image_output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        image_temporary.unlink(missing_ok=True)
        raise
    members = getattr(predictor, "members", None)
    if members is None:
        members = deployment_members(predictor.config)
    report = {"submission": str(output), "submission_sha256": sha256_file(output),
              "studies": count, "images": len(records), "data_rows": count * 10,
              "batch_size": 1, "precision": "float16", "tta": predictor.config["tta"],
              "ensemble": predictor.config["ensemble"],
              "member_count": len(members),
              "image_sizes": sorted({m["image_size"] for m in members}),
              "deployment_config_sha256": sha256_file(Path(model_dir) / "config.json"),
              "manifest_sha256": sha256_file(manifest) if manifest is not None else None,
              "deployment_inventory": predictor.inventory,
              "image_predictions": str(image_output),
              "image_predictions_sha256": sha256_file(image_output),
              "latency_status": "pending_cost_py_T4_measurement",
              "reproducibility": reproducibility}
    write_json(output.with_suffix(output.suffix + ".report.json"), report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent /
                        "data/clean_v1_reviewed/test_manifest.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import json
    print(json.dumps(predict_submission(**vars(args)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
