"""Prepare CPU-only latency inputs from complete original validation Studies.

Preview is the default. Pass --execute to copy exactly 100 measured and five
warmup images, preserving bytes and the original valid identities. No labels,
quality fields, model outputs, or test data participate in selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _image_root(path: Path) -> Path:
    choices = [candidate for candidate in (path, path / "cxr_first")
               if (candidate / "valid").is_dir()]
    if len(choices) != 1:
        raise ValueError("image root must contain exactly one valid/ or cxr_first/valid/")
    return choices[0].resolve()


def _read_manifest(data: bytes, root: Path) -> dict[str, list[dict]]:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    required = {"image_id", "split", "subject_id", "study_id", "relative_path", "file_sha256"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("manifest is missing required identity/SHA columns")
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError("duplicate manifest columns")
    studies: dict[str, list[dict]] = {}
    seen_ids, seen_paths, seen_sources = set(), set(), set()
    for number, row in enumerate(reader, 2):
        if None in row or any(not isinstance(row[field], str) for field in required):
            raise ValueError(f"malformed manifest row {number}")
        relative = row["relative_path"]
        parts = PurePosixPath(relative).parts
        if (row["split"] != "valid" or len(parts) != 4 or parts[0] != "valid"
                or "\\" in relative or "/".join(parts) != relative
                or not re.fullmatch(r"subject_[0-9]+", parts[1])
                or not re.fullmatch(r"study_[0-9]+", parts[2])
                or parts[3] in {".", ".."} or ".." in parts):
            raise ValueError(f"invalid valid identity/path at manifest row {number}")
        subject, study = parts[1:3]
        key = f"{subject}/{study}"
        if (row["subject_id"] != subject or row["study_id"] != study
                or ("study_key" in row and row["study_key"] != key)
                or ("dicom_id" in row and row["dicom_id"] != Path(parts[3]).stem)):
            raise ValueError(f"mismatched manifest identity at row {number}")
        image_id, digest = row["image_id"], row["file_sha256"]
        if not image_id or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError(f"missing image ID or invalid SHA256 at row {number}")
        source = (root / relative).resolve()
        if not _within(source, root) or not source.is_file() or source != root / relative:
            raise ValueError(f"missing file or source path escapes image root at row {number}")
        if image_id in seen_ids or relative in seen_paths or source in seen_sources:
            raise ValueError(f"duplicate image ID/file at row {number}")
        seen_ids.add(image_id)
        seen_paths.add(relative)
        seen_sources.add(source)
        studies.setdefault(key, []).append({
            "image_id": image_id, "subject_id": subject, "study_id": study,
            "study_key": key, "relative_path": relative,
            "file_sha256": digest.lower(), "sha256": digest.lower(),
            "source_path": str(source),
        })
    if not studies:
        raise ValueError("manifest contains no validation images")
    for rows in studies.values():
        rows.sort(key=lambda row: (row["relative_path"], row["image_id"]))
    return studies


def _exact_studies(studies: dict[str, list[dict]], count: int,
                   rng: random.Random) -> list[str]:
    keys = sorted(studies)
    rng.shuffle(keys)
    subsets: dict[int, tuple[str, ...]] = {0: ()}
    for key in keys:
        size = len(studies[key])
        for subtotal, selected in sorted(list(subsets.items()), reverse=True):
            total = subtotal + size
            if total <= count and total not in subsets:
                subsets[total] = selected + (key,)
        if count in subsets:
            return list(subsets[count])
    raise ValueError(f"cannot select exactly {count} images using complete Studies")


def prepare(manifest: Path, image_root: Path, output: Path, seed: int = 20261027,
            execute: bool = False) -> dict:
    """Validate and plan a selection; copy only when explicitly requested."""
    manifest, provided_root = Path(manifest).resolve(), Path(image_root).resolve()
    root = _image_root(provided_root)
    # lexists also rejects dangling symlinks and must precede resolve().
    if os.path.lexists(output):
        raise ValueError("output already exists; choose a new directory")
    output = Path(output).resolve()
    if (_within(output, provided_root) or _within(provided_root, output)
            or _within(output, root) or _within(root, output)
            or _within(manifest, output)):
        raise ValueError("output overlaps the raw image root or source manifest")
    manifest_data = manifest.read_bytes()
    manifest_digest = hashlib.sha256(manifest_data).hexdigest()
    studies = _read_manifest(manifest_data, root)
    rng = random.Random(seed)
    warm_keys = _exact_studies(studies, 5, rng)
    warm_patients = {studies[key][0]["subject_id"] for key in warm_keys}
    candidates = {key: rows for key, rows in studies.items()
                  if rows[0]["subject_id"] not in warm_patients}
    measured_keys = _exact_studies(candidates, 100, rng)
    groups = {}
    for name, keys, directory in (("warmup", warm_keys, "pictures_5"),
                                  ("measured", measured_keys, "pictures_100")):
        groups[name] = []
        for key in keys:
            expected = {Path(row["source_path"]) for row in studies[key]}
            study_dir = root / "valid" / key
            actual = {path.resolve() for path in study_dir.iterdir() if path.is_file()}
            if actual != expected:
                raise ValueError(f"incomplete original Study in manifest: {key}")
            for row in studies[key]:
                copied = dict(row)
                copied["output_relative_path"] = directory + "/" + "/".join(
                    PurePosixPath(row["relative_path"]).parts[1:])
                groups[name].append(copied)
    all_rows = groups["warmup"] + groups["measured"]
    result = {
        "schema_version": 1, "origin_split": "valid", "manifest_path": str(manifest),
        "manifest_sha256": manifest_digest, "image_root": str(root), "seed": seed,
        "selection_method": "Seeded shuffle of sorted Study IDs; exact-count subset dynamic "
                            "programming; warmup first, then exclude every warmup patient. "
                            "Uses only identities, Study image counts, and seed.",
        "full_study_image_counts": {key: len(studies[key]) for key in sorted(studies)},
        "counts": {"measured_images": 100, "warmup_images": 5,
                   "measured_studies": len(measured_keys), "warmup_studies": len(warm_keys),
                   "measured_patients": len({row["subject_id"] for row in groups["measured"]}),
                   "warmup_patients": len(warm_patients)},
        "complete_study_closure": True, "patient_disjoint": True,
        "labels_rewritten": False, "test_used": False, "gpu_used": False,
        "executed": execute, "hashes_verified": execute, **groups,
    }
    if execute:
        for row in all_rows:
            if sha256(Path(row["source_path"])) != row["file_sha256"]:
                raise ValueError(f"source SHA256 mismatch: {row['image_id']}")
        if sha256(manifest) != manifest_digest:
            raise ValueError("source manifest changed while preparing selection")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()  # Never overwrite an existing output, including a racing creator.
        try:
            for row in all_rows:
                source = Path(row["source_path"])
                target = output / row["output_relative_path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                if (sha256(target) != row["file_sha256"]
                        or sha256(source) != row["file_sha256"]):
                    raise ValueError(f"copied/source SHA256 mismatch: {row['image_id']}")
            if sha256(manifest) != manifest_digest:
                raise ValueError("source manifest changed during copying")
            (output / "selection.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except BaseException:
            shutil.rmtree(output)
            raise
    elif sha256(manifest) != manifest_digest:
        raise ValueError("source manifest changed while preparing selection")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("/home/featurize/data/cxr_first"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--execute", action="store_true", help="copy inputs; default is read-only preview")
    args = parser.parse_args()
    try:
        result = prepare(args.manifest, args.image_root, args.output, args.seed, args.execute)
    except (OSError, ValueError, csv.Error, UnicodeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
