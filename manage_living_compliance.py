"""Maintain the review package without changing its immutable closeout ZIP.

This standard-library-only utility performs file operations, never training or
inference. The historical receipt remains authoritative for the snapshot; the
living package has its own manifest and must not be described as a final entry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo
import zipfile

BASE_COMMIT = "c8bd868829c2a9f3445aea08eafbad0eeaee4fd5"
WORKBOOK = "模型核验材料与数据声明清单.xlsx"
SNAPSHOT_NAME = "closeout_c8bd868"
CHECKLIST_NAME = "REVIEW2_FREEZE_CHECKLIST_20261103"
FROZEN_PREFIXES = ("licenses/",)
FROZEN_FILES = {WORKBOOK, "A1_A5_FREEZE_RECORD.md",
                "evidence/XLSX_frozen_cell_comparison.json"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def timestamp() -> str:
    return datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(timespec="seconds")


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def safe_relative(value: str) -> Path:
    p = PurePosixPath(value)
    if p.is_absolute() or not p.parts or ".." in p.parts or "\\" in value:
        raise ValueError(f"Unsafe relative path: {value}")
    return Path(*p.parts)


def file_info(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular file: {path}")
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def archive_manifest(archive_path: Path) -> dict:
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError("Duplicate archive entries")
        for name in names:
            safe_relative(name)
        manifest = json.loads(archive.read("PACKAGE_SHA256SUMS.json"))
        entries = manifest["files"]
        expected = {e["path"] for e in entries} | {"PACKAGE_SHA256SUMS.json"}
        if len(entries) != len(expected) - 1 or set(names) != expected:
            raise ValueError("ZIP membership differs from payload manifest")
        for entry in entries:
            data = archive.read(entry["path"])
            if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError(f"ZIP payload mismatch: {entry['path']}")
        return manifest


def verify_snapshot(snapshot: Path, original_zip: Path, receipt_path: Path) -> dict:
    """Read back the copy, original, checksum sidecar and read-only permissions."""
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ValueError("Snapshot must be a real directory")
    expected_files = {original_zip.name, "DELIVERY_RECEIPT.json", "SNAPSHOT_METADATA.json", "SHA256SUMS.json"}
    if {p.name for p in snapshot.iterdir()} != expected_files:
        raise ValueError("Partial or unexpected snapshot contents; inspect before retrying")
    metadata = json.loads((snapshot / "SNAPSHOT_METADATA.json").read_text())
    sums = json.loads((snapshot / "SHA256SUMS.json").read_text())
    expected_sums = expected_files - {"SHA256SUMS.json"}
    if set(sums["files"]) != expected_sums:
        raise ValueError("Snapshot checksum membership mismatch")
    for name, expected in sums["files"].items():
        if file_info(snapshot / name) != expected:
            raise ValueError(f"Snapshot hash mismatch: {name}")
    source_info = file_info(original_zip)
    if file_info(snapshot / original_zip.name) != source_info:
        raise ValueError("Snapshot and original ZIP no longer match")
    if file_info(snapshot / "DELIVERY_RECEIPT.json") != file_info(receipt_path):
        raise ValueError("Historical receipt changed")
    receipt = json.loads(receipt_path.read_text())
    if source_info != {"bytes": receipt["zip_bytes"], "sha256": receipt["zip_sha256"]}:
        raise ValueError("Original ZIP differs from historical receipt")
    if metadata["archive"] != {"filename": original_zip.name, **source_info}:
        raise ValueError("Snapshot archive metadata mismatch")
    if metadata["base_commit"] != BASE_COMMIT or metadata["receipt"] != file_info(receipt_path):
        raise ValueError("Unexpected baseline commit")
    for p in [snapshot, *snapshot.iterdir()]:
        if stat.S_IMODE(p.stat().st_mode) & 0o222:
            raise ValueError(f"Snapshot is writable by mode bits: {p}")
    if os.path.samestat(original_zip.stat(), (snapshot / original_zip.name).stat()):
        raise ValueError("Snapshot must be an independent copy, not a hard link")
    payload = archive_manifest(snapshot / original_zip.name)
    return {"snapshot": str(snapshot), "archive": source_info,
            "payload_files_verified": len(payload["files"]), "read_only_mode_bits": True,
            "independent_copy": True, "original_unchanged": True}


def create_snapshot(base: Path) -> dict:
    receipt_path = base / "DELIVERY_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    original_zip = base / Path(receipt["zip_path"]).name
    expected = {"bytes": receipt["zip_bytes"], "sha256": receipt["zip_sha256"]}
    if file_info(original_zip) != expected:
        raise ValueError("Original ZIP does not match receipt; refuse snapshot")
    payload = archive_manifest(original_zip)
    snapshots = base / "snapshots"
    snapshots.mkdir(exist_ok=True)
    snapshot = snapshots / SNAPSHOT_NAME
    if snapshot.exists():
        return verify_snapshot(snapshot, original_zip, receipt_path)
    stage = Path(tempfile.mkdtemp(prefix=".snapshot-staging-", dir=snapshots))
    # A failed staging directory is deliberately retained for inspection.
    shutil.copyfile(original_zip, stage / original_zip.name)
    shutil.copyfile(receipt_path, stage / "DELIVERY_RECEIPT.json")
    if file_info(stage / original_zip.name) != expected:
        raise ValueError("Snapshot copy verification failed")
    metadata = {
        "schema_version": 1, "created_at": timestamp(), "timezone": "Asia/Hong_Kong",
        "artifact_type": "historical_review_draft_read_only_snapshot",
        "base_commit": BASE_COMMIT,
        "archive": {"filename": original_zip.name, **expected},
        "receipt": file_info(receipt_path), "source_zip": str(original_zip.resolve()),
        "payload_files_verified": len(payload["files"]),
        "final_members_frozen": False, "A1_A5_frozen": True,
        "protection": "Independent file copy; files 0444 and directory 0555. POSIX read-only, not WORM or an administrator-proof immutable flag.",
    }
    atomic_json(stage / "SNAPSHOT_METADATA.json", metadata)
    atomic_json(stage / "SHA256SUMS.json", {
        "algorithm": "sha256", "scope": "All snapshot files except this checksum sidecar",
        "files": {p.name: file_info(p) for p in sorted(stage.iterdir())},
    })
    for p in stage.iterdir():
        p.chmod(0o444)
    stage.chmod(0o555)
    if snapshot.exists():
        raise ValueError("Snapshot appeared concurrently; refusing replacement")
    stage.rename(snapshot)
    return verify_snapshot(snapshot, original_zip, receipt_path)


def frozen_material_check(package: Path, archive_path: Path) -> dict:
    manifest = archive_manifest(archive_path)
    frozen = [e for e in manifest["files"] if e["path"] in FROZEN_FILES
              or e["path"].startswith(FROZEN_PREFIXES)]
    for entry in frozen:
        if file_info(package / safe_relative(entry["path"])) != {"bytes": entry["bytes"], "sha256": entry["sha256"]}:
            raise ValueError(f"Frozen material changed: {entry['path']}")
    comparison = json.loads((package / "evidence/XLSX_frozen_cell_comparison.json").read_text())
    if comparison["cell_count"] != 62 or not comparison["all_equal"]:
        raise ValueError("Expected the existing 62-cell A1–A5 freeze record")
    return {"frozen_files_verified": len(frozen), "workbook_unchanged": True,
            "A1_A5_cells": 62, "A1_A5_frozen": True, "upstream_reverified": False}


def refresh_manifest(package: Path) -> dict:
    manifest_path = package / "PACKAGE_SHA256SUMS.json"
    files = []
    for path in sorted(package.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Package symlink is not allowed: {path}")
        if path.is_file() and path != manifest_path:
            files.append({"path": path.relative_to(package).as_posix(), **file_info(path)})
    data = {"scope": "All living package files except this manifest itself; historical ZIP has its own unchanged manifest",
            "files": files}
    atomic_json(manifest_path, data)
    return data


def verify_manifest(package: Path) -> int:
    data = json.loads((package / "PACKAGE_SHA256SUMS.json").read_text())
    actual_paths = {p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()}
    expected_paths = {e["path"] for e in data["files"]} | {"PACKAGE_SHA256SUMS.json"}
    if actual_paths != expected_paths:
        raise ValueError("Living package membership differs from manifest; refresh required")
    for e in data["files"]:
        if file_info(package / safe_relative(e["path"])) != {"bytes": e["bytes"], "sha256": e["sha256"]}:
            raise ValueError(f"Living package hash mismatch: {e['path']}")
    return len(data["files"])


def update_package(base: Path, root: Path, sync_sources: list[str]) -> dict:
    snapshot = create_snapshot(base)
    package = base / "package"
    receipt = json.loads((base / "DELIVERY_RECEIPT.json").read_text())
    archive_path = base / "snapshots" / SNAPSHOT_NAME / Path(receipt["zip_path"]).name
    frozen = frozen_material_check(package, archive_path)
    state_path = package / "LIVING_PACKAGE.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "schema_version": 1, "created_at": timestamp(), "base_commit": BASE_COMMIT,
        "package_mode": "living_review_package", "final_members_frozen": False,
        "final_members": None, "official_compliance_pass": False,
        "A1_A5_frozen": True, "A1_A5_frozen_cells": 62,
        "historical_snapshot": "../snapshots/" + SNAPSHOT_NAME,
        "historical_zip_sha256": receipt["zip_sha256"],
        "historical_receipt": "../DELIVERY_RECEIPT.json",
        "historical_matrix": receipt["matrix"],
        "run_evidence_policy": "New authorized runs write run_dir/compliance at start, every epoch, each checkpoint, and termination. Synthetic examples are not training evidence.",
        "freeze_checklist": CHECKLIST_NAME + ".md",
        "source_sync": {},
    }
    if state["final_members_frozen"] is not False or state["final_members"] is not None:
        raise ValueError("This maintenance command cannot freeze or choose final members")
    plans = []
    for name in sync_sources:
        relative = safe_relative(name)
        source = root / relative
        if relative.suffix not in {".py", ".md", ".json"}:
            raise ValueError("Only explicitly named source/document files may be synced")
        if name.startswith("docs/"):
            destination = package / relative.name
        elif len(relative.parts) == 1 and relative.suffix == ".py":
            destination = package / "code" / relative
        else:
            raise ValueError("Sync requires a root Python source or docs/ document")
        if destination.relative_to(package).as_posix() in FROZEN_FILES:
            raise ValueError("Cannot overwrite frozen material")
        info = file_info(source)
        plans.append((name, source, destination, info))
    for name, source, destination, info in plans:
        destination.parent.mkdir(parents=True, exist_ok=True)
        prior = state["source_sync"].get(name, {})
        old_info = file_info(destination) if destination.exists() else None
        if not destination.exists() or file_info(destination) != info:
            shutil.copyfile(source, destination)
        state["source_sync"][name] = {"package_path": destination.relative_to(package).as_posix(),
                                      "pre_living_file": prior.get("pre_living_file", old_info), **info}
    state["last_refreshed_at"] = timestamp()
    atomic_json(state_path, state)
    (package / "LIVING_PACKAGE.md").write_text(
        "# Living package — REVIEW-2 前置维护\n\n"
        "本目录是可持续更新的审查主包，最终成员仍未冻结，不能据此声明正式合规通过。\n"
        "A1–A5 的 62 格及其样式、许可证和来源材料继续冻结；本轮整个 XLSX 字节亦保持不变。\n\n"
        "原交付 ZIP 和 DELIVERY_RECEIPT.json 保留原位原字节；独立只读快照及其 SHA-256 元数据位于 "
        "`../snapshots/closeout_c8bd868/`。文件 0444、目录 0555，只读权限不等同不可被管理员修改的 WORM 存储。\n\n"
        "主包 code/ 可以接入新的合规记录实现；每次明确同步的来源文件与 SHA 记在 LIVING_PACKAGE.json，"
        "历史执行源码仍在 evidence/ 和原 ZIP 中，不能将今天的源码冒称历史运行版本。\n\n"
        "未来经单独授权的新 run 在其 compliance/ 下同步记录启动、逐 epoch、checkpoint 落盘和结束事件；"
        "续训追加新会话，合成演示明确标注 synthetic，不充当真实训练或官方验收。"
        "运行证据与本包每次组装时的索引分开保存，组装后必须刷新并核验 PACKAGE_SHA256SUMS.json。\n\n"
        "七项最终冻结登记见 " + CHECKLIST_NAME + ".md；登记清单不等于通过，最终成员和官方 T4/指定样本/Study 验收仍待。"
        "本轮不自动训练、推理、优化、租机或上传。\n",
        encoding="utf-8",
    )
    for name, title in [("README.md", "## Living package 维护入口"),
                        ("PACKAGE_STRUCTURE.md", "## Living package 增补")]:
        p = package / name
        content = p.read_text()
        if title not in content:
            p.write_text(content.rstrip() + "\n\n" + title + "\n\n"
                         "2026-11-03 起本目录作为持续维护的主包；见 `LIVING_PACKAGE.md`、`LIVING_PACKAGE.json` "
                         "和 `" + CHECKLIST_NAME + ".md`。原 ZIP 为历史只读快照，不再用旧交付回执冒充主包当前哈希。"
                         "上文 2026-11-02 执行记录保留历史时间语境。最终成员尚未冻结。\n", encoding="utf-8")
    frozen_material_check(package, archive_path)
    manifest = refresh_manifest(package)
    verify_manifest(package)
    original_manifest = archive_manifest(archive_path)
    original = {e["path"]: e for e in original_manifest["files"]}
    current = {e["path"]: e for e in manifest["files"]}
    changed = [name for name in sorted(original.keys() & current.keys()) if original[name] != current[name]]
    added = sorted(current.keys() - original.keys())
    removed = sorted(original.keys() - current.keys())
    if removed:
        raise ValueError(f"Historical payload unexpectedly removed: {removed}")
    report = {
        "verified_at": timestamp(), "mode": "living_review_package",
        "snapshot": snapshot, "frozen_material": frozen,
        "living_payload_files": len(manifest["files"]), "historical_payload_files": len(original),
        "changed_paths": changed, "added_paths": added, "removed_paths": removed,
        "final_members_frozen": False, "official_compliance_pass": False,
        "historical_receipt_unchanged": True,
    }
    atomic_json(base / "LIVING_PACKAGE_VERIFICATION.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("artifacts/compliance_closeout_20261102"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--sync-source", action="append", default=[])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        receipt = json.loads((args.base / "DELIVERY_RECEIPT.json").read_text())
        archive = args.base / Path(receipt["zip_path"]).name
        snap = args.base / "snapshots" / SNAPSHOT_NAME
        report = {"snapshot": verify_snapshot(snap, archive, args.base / "DELIVERY_RECEIPT.json"),
                  "frozen_material": frozen_material_check(args.base / "package", snap / archive.name),
                  "living_payload_files": verify_manifest(args.base / "package")}
    else:
        report = update_package(args.base, args.root, args.sync_source)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
