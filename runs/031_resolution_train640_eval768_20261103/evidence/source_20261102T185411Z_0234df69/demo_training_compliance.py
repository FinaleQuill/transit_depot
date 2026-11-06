"""Create a clearly labelled lifecycle example using bytes only, never training."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from training_compliance import ComplianceRecorder, read_events


def create_example(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    (output / "README.md").write_text(
        "# Compliance lifecycle example — NOT A TRAINING RUN\n\n"
        "Only synthetic metadata and small byte fixtures. No model, image, GPU, "
        "training, inference or real AUC. Both checkpoint versions are text bytes.\n",
        encoding="utf-8")
    recorder = ComplianceRecorder(output, trainer="metadata_example_only",
        configuration={"demonstration_only": True, "actual_training_executions": 0,
                       "actual_inference_executions": 0, "final_members_frozen": False},
        source_files=[Path(__file__)], demonstration=True)
    try:
        for epoch in (1, 2):
            temporary = output / "fixture_checkpoint.txt.tmp"
            checkpoint = output / "fixture_checkpoint.txt"
            data = f"NOT MODEL WEIGHTS. Synthetic checkpoint version {epoch}.\n".encode()
            with temporary.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(checkpoint)
            fd = os.open(str(output), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            recorder.checkpoint_saved(checkpoint, hashlib.sha256(data).hexdigest(), epoch=epoch, role="fixture")
            recorder.epoch_completed(epoch, {"synthetic_metadata": True, "real_metrics": None})
        recorder.finish("completed", {"demonstration_only": True, "actual_training_executions": 0})
    finally:
        recorder.close()
    rows = read_events(output / "compliance/events.jsonl")
    receipt = {"demonstration_only": True, "actual_training_executions": 0,
        "actual_inference_executions": 0, "event_types": [r["event_type"] for r in rows],
        "events": len(rows), "last_record_sha256": rows[-1]["record_sha256"]}
    (output / "DEMO_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(create_example(parser.parse_args().output), indent=2))
