"""Independent, CPU-only CSV audit; does not import project/model code."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path


HEADER = ["Subject_id", "Study_id", "Predict_class", "Probability"]


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        return next(reader, []), list(reader)


def check(submission: Path, manifest: Path, template: Path, wide: Path | None = None) -> dict:
    # 只读测试清单中的身份字段；不使用、猜测或拟合测试标签。
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    expected_studies = list(dict.fromkeys((r["subject_id"], r["study_id"]) for r in records))
    expected_keys = {(subject, study, str(c)) for subject, study in expected_studies for c in range(1, 11)}
    header, rows = read_csv(submission)
    template_header, template_rows = read_csv(template)
    errors: list[str] = []
    if header not in (HEADER, HEADER + ["Gate"]):
        errors.append(f"unexpected header: {header!r}")
    if template_header != HEADER:
        errors.append("reference template header changed")
    malformed = [i + 2 for i, row in enumerate(rows) if len(row) != len(header)]
    if malformed:
        errors.append(f"malformed CSV rows: {malformed[:10]}")
    good_rows = [row for row in rows if len(row) == len(header) and len(row) >= 4]
    keys = [tuple(row[:3]) for row in good_rows]
    duplicate_keys = sum(count - 1 for count in collections.Counter(keys).values())
    missing = expected_keys - set(keys)
    extra = set(keys) - expected_keys
    if duplicate_keys or missing or extra:
        errors.append(f"identity mismatch: duplicates={duplicate_keys}, missing={len(missing)}, extra={len(extra)}")
    if len(expected_studies) != 1414 or len(rows) != 14140:
        errors.append(f"unexpected counts: manifest studies={len(expected_studies)}, output rows={len(rows)}")
    template_order_equal = keys == [tuple(row[:3]) for row in template_rows]
    manifest_order_equal = keys == [(s, t, str(c)) for s, t in expected_studies for c in range(1, 11)]
    if not template_order_equal:
        errors.append("output identity row order differs from reference template")
    probabilities: list[float] = []
    invalid_probability_rows: list[int] = []
    invalid_gate_rows: list[int] = []
    for i, row in enumerate(good_rows, 2):
        try:
            value = float(row[3])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid probability")
            probabilities.append(value)
        except (ValueError, TypeError):
            invalid_probability_rows.append(i)
        if header == HEADER + ["Gate"] and row[4] != "":
            try:
                value = float(row[4])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("invalid Gate")
            except (ValueError, TypeError):
                invalid_gate_rows.append(i)
    if invalid_probability_rows or invalid_gate_rows:
        errors.append(f"invalid numeric rows: probabilities={invalid_probability_rows[:10]}, Gate={invalid_gate_rows[:10]}")
    wide_max_absolute_difference = None
    if wide is not None and len(probabilities) == len(good_rows):
        with wide.open(encoding="utf-8-sig", newline="") as stream:
            wide_rows = list(csv.DictReader(stream))
        wide_values = {(row["Subject_id"], row["Study_id"], str(c)): float(row[f"class_{c}"])
                       for row in wide_rows for c in range(1, 11)}
        if len(wide_rows) != len(expected_studies) or set(wide_values) != expected_keys:
            errors.append("wide output identity coverage mismatch")
        elif any(not math.isfinite(value) or not 0 <= value <= 1 for value in wide_values.values()):
            errors.append("wide output has a nonfinite or out-of-range probability")
        else:
            wide_max_absolute_difference = max(abs(value - wide_values[key]) for key, value in zip(keys, probabilities))
            if not math.isfinite(wide_max_absolute_difference) or wide_max_absolute_difference > 1e-12:
                errors.append(f"wide output probability mismatch: {wide_max_absolute_difference}")
    raw = submission.read_bytes()
    # 严格解码可检测编码破损；BOM/换行形式本身不是平台已证明的拒收理由。
    raw.decode("utf-8")
    report = {
        "passed": not errors,
        "submission": str(submission.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "header": header,
        "encoding": "UTF-8",
        "utf8_bom": raw.startswith(b"\xef\xbb\xbf"),
        "terminal_newline": raw.endswith(b"\n"),
        "data_rows": len(rows),
        "lines_including_header": len(rows) + 1,
        "manifest_images": len(records),
        "studies": len(set(key[:2] for key in keys)),
        "subjects": len(set(key[0] for key in keys)),
        "classes": sorted({key[2] for key in keys}, key=lambda value: int(value) if value.isdigit() else -1),
        "duplicate_keys": duplicate_keys,
        "missing_keys": len(missing),
        "unexpected_keys": len(extra),
        "malformed_rows": len(malformed),
        "template_identity_order_equal": template_order_equal,
        "manifest_identity_order_equal": manifest_order_equal,
        "invalid_probability_rows": len(invalid_probability_rows),
        "invalid_gate_rows": len(invalid_gate_rows),
        "probability_min": min(probabilities) if probabilities else None,
        "probability_max": max(probabilities) if probabilities else None,
        "wide_max_absolute_difference": wide_max_absolute_difference,
        "official_rule_url": "https://www.ncccu.org.cn/index/Paper/case1.html",
        "platform_upload_performed": False,
        "errors": errors,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("/home/featurize/work/cloud_project_v2/data/clean_v1_reviewed/test_manifest.csv"))
    parser.add_argument("--template", type=Path, default=Path("/home/featurize/work/submission.csv"))
    parser.add_argument("--wide", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = check(args.submission, args.manifest, args.template, args.wide)
    body = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.json_out is not None:
        args.json_out.write_text(body, encoding="utf-8")
    print(body, end="")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
