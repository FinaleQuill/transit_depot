"""Prepare a reproducible compliance directory without downloading or running models.

All source files are read-only. Original pretraining weights are copied for
license/source verification, not loaded by inference and not part of model/.
Missing submission/timing/FP16-validation evidence leaves the package pending.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET
import zipfile

from deploy_kd import (deployment_inventory, read_json, read_manifest, read_submission,
                       sha256_file, validate_deployment_config, write_json)

PROJECT_ROOT = Path(__file__).resolve().parent
CODE_FILES = (
    "train_teacher.py", "run_teachers.py", "train_distill.py", "extract_soft_labels.py",
    "kd_common.py", "distill_data.py", "model_overnight.py", "train_overnight.py",
    "dataset.py", "dataset_overnight.py", "cv_splits.py", "runtime_compat.py",
    "deploy_kd.py", "predict.py", "cost.py", "export_student.py", "validate_student.py",
    "package_kd_submission.py", "requirements.txt", "requirements_kd.txt", "requirements_inference.txt",
)
MANIFEST_FILES = (
    "data/clean_v1_reviewed/train_clean.csv",
    "data/clean_v1_reviewed/valid_manifest.csv",
    "data/clean_v1_reviewed/test_manifest.csv",
)
SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _column_name(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def write_xlsx(path: str | Path, rows: Iterable[Sequence[Any]]) -> None:
    """Minimal OOXML workbook using inline strings, preserving IDs and SHA text."""
    workbook = ET.Element("workbook", xmlns=SHEET_NS, attrib={"xmlns:r": REL_NS})
    sheets = ET.SubElement(workbook, "sheets")
    ET.SubElement(sheets, "sheet", name="模型核验材料", sheetId="1", attrib={"r:id": "rId1"})
    worksheet = ET.Element("worksheet", xmlns=SHEET_NS)
    data = ET.SubElement(worksheet, "sheetData")
    for row_number, values in enumerate(rows, start=1):
        row = ET.SubElement(data, "row", r=str(row_number))
        for column, value in enumerate(values, start=1):
            cell = ET.SubElement(row, "c", r=f"{_column_name(column)}{row_number}", t="inlineStr")
            inline = ET.SubElement(cell, "is")
            text = ET.SubElement(inline, "t", attrib={"{http://www.w3.org/XML/1998/namespace}space": "preserve"})
            text.text = "" if value is None else str(value)
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>')
    root_relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>')
    workbook_relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '</Relationships>')
    with zipfile.ZipFile(path, mode="x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_relationships)
        archive.writestr("xl/workbook.xml", ET.tostring(workbook, encoding="utf-8", xml_declaration=True))
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_relationships)
        archive.writestr("xl/worksheets/sheet1.xml", ET.tostring(worksheet, encoding="utf-8", xml_declaration=True))


def unique_pretrains(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Deduplicate the original weights by exact repository and SHA, not filename."""
    from kd_common import validate_pretrained_provenance

    entries = [("student", config["pretrained_provenance"])]
    entries.extend((f"teacher_fold_{t['fold_index']}", t["pretrained_provenance"])
                   for t in config["teacher_provenance"])
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for role, provenance in entries:
        validate_pretrained_provenance(provenance, provenance["backbone_name"])
        key = (provenance["repo_id"], provenance["weights_sha256"])
        if key not in unique:
            unique[key] = {"provenance": provenance, "roles": []}
        unique[key]["roles"].append(role)
    return list(unique.values())


def locate_pretrain(provenance: Mapping[str, Any], cache: str | Path | None = None) -> Path:
    """Search explicit local cache layouts; never download missing source weights."""
    candidates = []
    if cache is not None:
        root = Path(cache).resolve(strict=True)
        repo = provenance["repo_id"]
        revision = provenance["revision"]
        candidates.extend([
            root / f"{provenance['weights_sha256']}.safetensors",
            root / repo.replace("/", "__") / "model.safetensors",
            root / repo / "model.safetensors",
            root / ("models--" + repo.replace("/", "--")) / "snapshots" / revision / "model.safetensors",
            root / provenance["backbone_name"] / "model.safetensors",
        ])
    if provenance.get("weights_path"):
        candidates.append(Path(provenance["weights_path"]))
    mismatches = []
    for candidate in candidates:
        if candidate.is_file():
            if sha256_file(candidate) == provenance["weights_sha256"]:
                return candidate.resolve(strict=True)
            mismatches.append(str(candidate))
    if mismatches:
        raise ValueError(f"Original pretraining SHA-256 mismatch: {mismatches}")
    raise FileNotFoundError(
        f"Original weights not found for {provenance['repo_id']}; provide --pretrain-cache. No download was attempted.")


def copy_verified(source: str | Path, destination: str | Path, expected_sha256: str) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with Path(source).open("rb") as incoming, destination.open("xb") as outgoing:
        for block in iter(lambda: incoming.read(1024 * 1024), b""):
            digest.update(block)
            outgoing.write(block)
    if digest.hexdigest() != expected_sha256:
        destination.unlink()
        raise ValueError(f"Source changed or copied data failed SHA-256 verification: {source}")


def validate_submission_material(path: str | Path, *, test_manifest: str | Path,
                                 expected_studies: int = 1414) -> dict[str, Any]:
    path = Path(path)
    if path.name != "submission.csv":
        raise ValueError("Submitted prediction material must be named submission.csv")
    values = read_submission(path)
    records = read_manifest(test_manifest)
    studies = {(r["subject_id"], r["study_id"]) for r in records}
    if len(studies) != expected_studies or len(values) != expected_studies:
        raise ValueError("Submission or original test manifest Study count differs from the declared dataset")
    if all(subject is None for subject, _ in values):
        if {study for _, study in values} != {study for _, study in studies}:
            raise ValueError("Submission does not cover all original test Studies")
    elif set(values) != studies:
        raise ValueError("Submission patient/Study coverage differs from the original test manifest")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        if "Gate" in header and header[-1] != "Gate":
            raise ValueError("Gate must be the last column")
        for row in reader:
            if row.get("Gate", "").strip():
                gate = float(row["Gate"])
                if not math.isfinite(gate) or not 0 <= gate <= 1:
                    raise ValueError("Submission contains invalid Gate values")
    return {"studies": len(studies), "data_rows": len(studies) * 10,
            "sha256": sha256_file(path), "identity_coverage_complete": True}


def _same_inventory(reported: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    def entries(inventory):
        return {(f["path"], f["bytes"], f["sha256"]) for f in inventory.get("files", [])}
    return entries(reported) == entries(actual) and reported.get("total_bytes") == actual["total_bytes"]


def verify_reports(config: Mapping[str, Any], inventory: Mapping[str, Any], *,
                   submission: Mapping[str, Any] | None, cost: Mapping[str, Any] | None,
                   validation: Mapping[str, Any] | None) -> dict[str, Any]:
    missing = []
    if submission is None:
        missing.append("submission.csv")
    cost_pass = False
    if cost is None:
        missing.append("T4 cost_report.json")
    else:
        if not _same_inventory(cost.get("inference_files", {}), inventory):
            raise ValueError("Timing report was generated for another deployment")
        if submission is not None and cost.get("submission_sha256") != submission["sha256"]:
            raise ValueError("Timing report does not correspond to the packaged submission")
        cost_pass = cost.get("official_compliance_pass") is True
        if not cost_pass:
            missing.append("passing local T4 timing and CSV agreement")
    validation_pass = False
    if validation is None:
        missing.append("FP16 official-valid validation_report.json")
    else:
        if not _same_inventory(validation.get("deployment_inventory", {}), inventory):
            raise ValueError("FP16 validation report was generated for another deployment")
        if validation.get("manifest_sha256") != config["official_valid_manifest_sha256"]:
            raise ValueError("FP16 validation used another validation manifest")
        auc = validation.get("macro_auc")
        validation_pass = (validation.get("complete") is True and validation.get("precision") == "float16"
                           and validation.get("image_size") == 512 and validation.get("tta") == "identity"
                           and validation.get("ensemble") is False and isinstance(auc, (float, int))
                           and math.isfinite(auc) and 0 <= auc <= 1)
        if not validation_pass:
            missing.append("complete FP16 validation evidence")
    ready = submission is not None and cost_pass and validation_pass and inventory["size_pass"]
    return {"status": "ready_for_organizer_review" if ready else "pending",
            "ready_for_organizer_review": ready, "pending_items": missing,
            "final_organizer_retest_required": True}


def package_submission(*, model_dir: str | Path, output: str | Path,
                       submission: str | Path | None = None,
                       cost_report: str | Path | None = None,
                       validation_report: str | Path | None = None,
                       pretrain_cache: str | Path | None = None,
                       project_root: str | Path = PROJECT_ROOT,
                       expected_studies: int = 1414) -> dict[str, Any]:
    model_dir = Path(model_dir).resolve(strict=True)
    source = Path(project_root).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new verification-package directory")
    if output.is_relative_to(model_dir):
        raise ValueError("Package output cannot be inside its source deployment")
    config = read_json(model_dir / "config.json")
    validate_deployment_config(config)
    inventory = deployment_inventory(model_dir, config)
    if not inventory["size_pass"]:
        raise ValueError("Final inference files exceed 500,000,000 bytes")
    if sha256_file(model_dir / "student_fp16.pt") != config["weights_sha256"]:
        raise ValueError("Deployment weights changed after export")
    originals = unique_pretrains(config)
    for item in originals:
        item["source_path"] = locate_pretrain(item["provenance"], pretrain_cache)
    for name in CODE_FILES + MANIFEST_FILES:
        if not (source / name).is_file():
            raise FileNotFoundError(f"Required reproducibility source is missing: {name}")
    if sha256_file(source / MANIFEST_FILES[0]) != config["source_manifest_sha256"]:
        raise ValueError("Training manifest differs from deployment ancestry")
    if sha256_file(source / MANIFEST_FILES[1]) != config["official_valid_manifest_sha256"]:
        raise ValueError("Validation manifest differs from deployment ancestry")
    submission_info = None if submission is None else validate_submission_material(
        submission, test_manifest=source / MANIFEST_FILES[2], expected_studies=expected_studies)
    cost = None if cost_report is None else read_json(cost_report)
    validation = None if validation_report is None else read_json(validation_report)
    readiness = verify_reports(config, inventory, submission=submission_info, cost=cost, validation=validation)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "package_status.json", {"status": "preparing", "ready_for_organizer_review": False})
    copied_sources = []
    sheet_rows: list[list[Any]] = [["序号", "名称", "规则依据", "内容或包内路径", "source_sha256", "code_sha256"]]
    for item in inventory["files"]:
        copy_verified(model_dir / item["path"], output / "model" / item["path"], item["sha256"])
    for name in CODE_FILES + MANIFEST_FILES:
        digest = sha256_file(source / name)
        target = output / "code" / name
        copy_verified(source / name, target, digest)
        copied_sources.append({"source": name, "package_path": str(target.relative_to(output)), "sha256": digest})
        sheet_rows.append([f"CODE-{len(copied_sources)}", "代码或清单", "附录A", str(target.relative_to(output)), digest, digest if name.endswith(".py") else ""])
    original_plan = source / "data/cv5_overnight/folds.json"
    if original_plan.is_file():
        digest = sha256_file(original_plan)
        expected_plans = {teacher["fold_plan_sha256"] for teacher in config["teacher_provenance"]}
        if expected_plans != {digest}:
            raise ValueError("Local fold plan differs from the actual teachers' ancestry")
        copy_verified(original_plan, output / "code/data/cv5_overnight/folds.original.json", digest)
        copied_sources.append({"source": "data/cv5_overnight/folds.json", "package_path": "code/data/cv5_overnight/folds.original.json", "sha256": digest})
    pretrain_materials = []
    for index, item in enumerate(originals, start=1):
        provenance = item["provenance"]
        name = provenance["repo_id"].replace("/", "__") + "_" + provenance["weights_sha256"][:12]
        weight_relative = Path("pretrain_model") / name / "model.safetensors"
        copy_verified(item["source_path"], output / weight_relative, provenance["weights_sha256"])
        materials = {}
        for key, filename in (("apache_license", "Apache-2.0.txt"), ("upstream_license", "MIT.txt"),
                              ("model_card", "README.md")):
            relative = (Path("modelcards") / name / filename if key == "model_card"
                        else Path("licenses") / name / filename)
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8", newline="") as stream:
                stream.write(provenance["materials"][key]["text"])
            if sha256_file(target) != provenance["materials"][key]["sha256"]:
                raise ValueError("Packaged license/model-card content changed")
            materials[key] = {"path": str(relative), "sha256": sha256_file(target),
                              "source_url": provenance["materials"][key].get("url")}
        pretrain_materials.append({"repo_id": provenance["repo_id"], "backbone_name": provenance["backbone_name"],
                                  "roles": item["roles"], "revision": provenance["revision"],
                                  "weights_url": provenance["weights_url"], "weights_sha256": provenance["weights_sha256"],
                                  "package_path": str(weight_relative), "materials": materials,
                                  "counted_as_inference_file": False})
        for key, label, value in (
            ("A1", "所用开源模型名称", provenance["backbone_name"]),
            ("A2", "权重公开链接", provenance["weights_url"]),
            ("A3", "许可证全文", f"{materials['apache_license']['path']}；{materials['upstream_license']['path']}"),
            ("A4", "上游训练数据说明", f"ImageNet-22k→ImageNet-1k；{materials['model_card']['path']}"),
            ("A5", "原预训练模型权重哈希", provenance["weights_sha256"]),
            ("A6", "推理加速精度", "FP16；仅单个Tiny/512，无TTA、无集成"),
        ):
            sheet_rows.append([f"{index}-{key}", label, "第1条/第7条/附录A", value, provenance["weights_sha256"], ""])
    for arg, relative in ((submission, "submission.csv"), (cost_report, "audit/cost_report.json"),
                          (validation_report, "audit/validation_report.json")):
        if arg is not None:
            copy_verified(arg, output / relative, sha256_file(arg))
    if (model_dir / "export_report.json").is_file():
        path = model_dir / "export_report.json"
        copy_verified(path, output / "audit/export_report.json", sha256_file(path))
    root_cost = ('"""Official cost entry; loads the same implementation as code/predict.py."""\n'
                 'from pathlib import Path\nimport sys\n'
                 'sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))\n'
                 'from cost import main\n'
                 'if __name__ == "__main__":\n    raise SystemExit(main())\n')
    (output / "cost.py").write_text(root_cost, encoding="utf-8")
    student_train = ('"""Student training entry; defaults to dry-run."""\n'
                     'from train_distill import main\n'
                     'if __name__ == "__main__":\n    raise SystemExit(main())\n')
    (output / "code/train.py").write_text(student_train, encoding="utf-8")
    (output / "code/model.py").write_text('"""Approved model builder alias."""\nfrom model_overnight import build_overnight_model, validate_model_config\n', encoding="utf-8")
    (output / "code/utils.py").write_text('"""Hash/provenance utilities alias."""\nfrom kd_common import sha256_file, validate_pretrained_provenance\n', encoding="utf-8")
    sheet_rows.extend([
        ["D1", "验证集使用声明", "第3条", "官方valid仅用于学生早停/选checkpoint及独立FP16验证，不加入梯度训练；教师外折不选模", "", ""],
        ["D2", "测试集使用声明", "第5/6条", "测试集仅固定推理及脚本运行验证；不生成训练伪标签、不拟合参数", "", ""],
        ["D3", "推理文件实际大小", "第8条", f"model/实际调用文件合计{inventory['total_bytes']} bytes；上限500000000 bytes", "", ""],
        ["D4", "材料状态", "第10条/附录A", readiness["status"] + "；" + "；".join(readiness["pending_items"]), "", ""],
    ])
    write_xlsx(output / "模型核验材料与数据申明清单.xlsx", sheet_rows)
    readme = f"""# 蒸馏模型核验材料

材料状态：{readiness['status']}。待补项：{'；'.join(readiness['pending_items']) or '无（仍须主办方复测）'}。
缺少最终CSV、同一部署的T4计时或FP16官方valid验证时，本目录只是准备材料，不能宣称可正式提交。

model/中仅有单个V1 Tiny的FP16权重及必要config，实际合计{inventory['total_bytes']} bytes。
pretrain_model/是核验用原始公开预训练权重，不被predict/cost加载；licenses/和modelcards/用于来源核验。
第8条500MB仅统计实际推理文件，本核验目录总大小可以超过500MB。没有打包比赛图像、旧V2权重或教师训练快照。

code/含实际训练、OOF提取、学生蒸馏和共享推理源码，代码与清单SHA见materials_manifest.json和XLSX。
三份清洗清单原样复制在code/data/clean_v1_reviewed/；原始图像须由主办方在本机提供。
code/data/cv5_overnight/folds.original.json如存在，仅为原折表证据，可能含原机器绝对路径，不应直接用于异机续训。
在异机重放时，使用同seed=43重建新折表，并核对Study→fold映射与原表一致：

    python code/cv_splits.py --train-manifest code/data/clean_v1_reviewed/train_clean.csv --valid-manifest code/data/clean_v1_reviewed/valid_manifest.csv --output code/data/cv5_overnight/folds.json --seed 43

固定推理（输出目录需新建且文件不能已存在；路径按实际原图位置填写）：

    python code/predict.py --model-dir model --image-root /path/to/cxr_first --manifest code/data/clean_v1_reviewed/test_manifest.csv --output /new/output/submission.csv

官方口径计时：

    python cost.py submission.csv /path/to/pictures_100 /path/to/pictures_5 --model-dir model --manifest code/data/clean_v1_reviewed/test_manifest.csv --output-dir /new/cost_report

计时调用与predict相同的模型加载和单图预测，包含逐图十类sidecar写出和Study结果写出。100图若缺少所选Study的其他视图，CSV一致性不能认证，会标pending；不得读取目录外预测补齐。
训练与推理依赖分别见code/requirements_kd.txt和code/requirements_inference.txt；code/requirements.txt仅为历史环境记录。推理沿用官方Py3.12、torch2.10.0、CUDA12.8/cuDNN9；不覆盖官方CUDA环境。
本脚本只复制和核验材料，不下载、不启动训练或模型前向。最终合规与评分以主办方复测为准。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    result = {**readiness, "inference_inventory": inventory, "submission": submission_info,
              "pretraining_sources": pretrain_materials, "copied_sources": copied_sources,
              "runtime_weights": ["model/student_fp16.pt"],
              "pretrain_verification_weights_counted_toward_500MB": False,
              "competition_images_included": False, "models_executed": False,
              "downloads_performed": False,
              "files": [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                         "sha256": sha256_file(path)}
                        for path in sorted(output.rglob("*")) if path.is_file()
                        and path.name not in ("package_status.json", "materials_manifest.json")]}
    write_json(output / "materials_manifest.json", result)
    write_json(output / "package_status.json", readiness)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--cost-report", type=Path)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--pretrain-cache", type=Path)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--expected-studies", type=int, default=1414)
    args = parser.parse_args(argv)
    result = package_submission(**vars(args))
    print(json.dumps({k: result[k] for k in ("status", "ready_for_organizer_review", "pending_items", "inference_inventory")},
                     ensure_ascii=False, indent=2))
    return 0 if result["ready_for_organizer_review"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
