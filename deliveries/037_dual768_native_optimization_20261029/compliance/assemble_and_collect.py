"""Real-record evidence assembly and CPU collector; no model load or training."""
from pathlib import Path
import sys,json,datetime
sys.path.insert(0,"/home/featurize/work/cloud_project_v2")
from build_training_evidence import build_evidence
from package_kd_submission import collect_training_evidence
from deploy_kd import read_json
root=Path("/home/featurize/work/cloud_project_v2/artifacts/kd_v1/final_dual768_evidence_20261029")
model=Path("/home/featurize/work/deliveries/033_dual_tiny768_seed43_25_seed44_75/fp16_hflip2")
assembly=build_evidence(model_dir=model,teacher_evidence=root/"teacher_only",student_runs=["/home/featurize/work/runs/029_tiny768_kd_seed43","/home/featurize/work/runs/030_tiny768_kd_seed44"],output=root/"combined",execute=True)
print(json.dumps({"assembly":assembly},indent=2),flush=True)
collected=collect_training_evidence(root/"combined",root/"collected",read_json(model/"config.json"))
report={"cpu_only":True,"models_loaded":False,"training_executed":False,"source_runs_modified":False,"checkpoint_binaries_copied":False,"checkpoint_binaries_hashed_by_builder":True,"assembly":assembly,"collection_complete":collected["complete"],"pending_items":collected["pending_items"],"runs":collected["runs"],"copied_small_file_count":len(collected["files"]),"historical_missing_evidence":collected["historical_missing_evidence"]}
Path("/home/featurize/work/deliveries/037_dual768_native_optimization_20261029/compliance/training_evidence_assembly.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2),flush=True)
