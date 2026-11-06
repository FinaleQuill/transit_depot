"""Derive a new truthful teacher-only index; never edits original training assets."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import hashlib,json

work=Path("/home/featurize/work")
source=work/"runs/024_convnext_v1_base768_mixed20_12_cv5/evidence/evidence_manifest.json"
audit_path=work/"deliveries/037_dual768_native_optimization_20261029/compliance/training_evidence_readonly.json"
raw=source.read_bytes(); audit_raw=audit_path.read_bytes(); audit=json.loads(audit_raw)
assert hashlib.sha256(raw).hexdigest()==audit["source_manifest_sha256"]
assert audit["controller_log_content_audit"]["all_48_teacher_epochs_have_all_expected_markers_and_completion_summaries"]
assert all(r["ledger_chain_valid"] and not r["missing_expected_epochs"] for r in audit["teacher_runs"])
root=work/"cloud_project_v2/artifacts/kd_v1/final_dual768_evidence_20261029/teacher_only"
root.mkdir(parents=True,exist_ok=False)
manifest=deepcopy(json.loads(raw))
teachers={r["id"] for r in manifest["runs"] if r.get("role")=="teacher"}
excluded=[r for r in manifest["runs"] if r.get("role")!="teacher"]
assert teachers=={f"teacher_fold_{i}" for i in range(5)}
manifest["runs"]=[r for r in manifest["runs"] if r["id"] in teachers]
manifest["source_version_timeline"]=[x for x in manifest["source_version_timeline"] if x.get("run_id") in teachers]
frozen_audit=root/"controller_scope_audit.json"
frozen_audit.write_bytes(audit_raw)
by_log={Path(x["path"]).name:x for x in audit["controller_log_content_audit"]["logs"]}
controller_ids={}
for log in manifest["console_logs"]:
 name=Path(log["path"]).name
 if name not in by_log:continue
 entry=by_log[name]; folds=entry["actual_batch_marker_folds"]
 log["original_manifest_coverage_from_start"]=log["coverage_from_start"]
 log["coverage_from_start"]="invocation_only"
 log["run_ids"]=[f"teacher_fold_{f}" for f in folds]
 log["scope_explanation"]="Invocation capture; full per-fold coverage is supported separately by the audited first/100-step/final markers and epoch-completion summaries. Failed and resumed attempts both remain in the original logs."
 log["source_sha256_at_derivation"]=entry["sha256"]
 receipt=Path(log["path"]).with_name(name.replace("console_","command_").replace(".log",".json"))
 assert receipt.is_file()
 log["command_receipt"]=str(receipt)
 manifest["supporting_files"].append({"kind":"native_controller_receipt","path":str(receipt)})
 controller_ids[tuple(folds)]=log["id"]
manifest["console_coverage"]=[x for x in manifest["console_coverage"] if x.get("run_id")=="teacher_fold_0"]
for fold in range(1,5):
 ids=[log["id"] for log in manifest["console_logs"] if f"teacher_fold_{fold}" in log.get("run_ids",[])]
 assert ids
 manifest["console_coverage"].append({"run_id":f"teacher_fold_{fold}","coverage_from_start":"complete","log_ids":ids,
  "coverage_explanation":f"For fold{fold}, all 12 epochs have every expected logged first/100-step/final batch marker and an epoch-completion summary across these original controller logs. The frozen controller_scope_audit.json records each marker count and original log SHA. This declares combined logged coverage, not byte-exact reconstruction of stdout or every batch. The failed partial fold3 epoch5 attempt and restarted attempt remain in separate original logs.",
  "derived_audit":str(frozen_audit)})
manifest["supporting_files"].append({"kind":"controller_log_scope_audit","path":str(frozen_audit)})
manifest["historical_missing_evidence"].append({"blocking":False,"detail":"The command_20261025T133827Z_c1c4fc8e.json controller receipt was not finalized: ended_at, exit_code and console_bytes remain absent/null in the original. Its original log ends with a child RuntimeError. No replacement receipt timestamp or exit code is fabricated. The later command_20261026T051826Z_5864de2d.json is finalized with exit0; original failed/restarted logs and ledger entries are retained."})
manifest["derivation"]={"created_at_utc":datetime.now(timezone.utc).isoformat(),"source_manifest":str(source),"source_manifest_sha256":hashlib.sha256(raw).hexdigest(),"excluded_run_ids":[r["id"] for r in excluded],"exclusion_reason":"Only the five frozen teacher runs belong in this input; the obsolete incomplete 022 student is not a deployed 033 member. The original manifest and all original references are unchanged.","original_assets_modified":False,"models_loaded":False,"checkpoint_binaries_read":False,"collector_validation_pending":True,"audit_sha256":hashlib.sha256(audit_raw).hexdigest()}
(root/"evidence_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
assert source.read_bytes()==raw
print(json.dumps({"teacher_only_index":str(root/"evidence_manifest.json"),"teacher_runs":len(manifest["runs"]),"excluded_run_ids":[r["id"] for r in excluded],"original_manifest_unchanged":True,"collector_validation_pending":True},indent=2))
