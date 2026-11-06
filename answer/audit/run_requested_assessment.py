"""User-requested 020 assessment; waits for safe stop, then inference only."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,os,shutil,subprocess,sys,time,traceback
PROJECT=Path("/home/featurize/work/cloud_project_v2")
RUN=Path("/home/featurize/work/runs/020_convnextv2_base768_posbce_cv5_4090")
OUTPUT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
 return h.hexdigest()

def main():
 print("Waiting for trainer and runner to exit at the epoch boundary",flush=True)
 while not (OUTPUT/"stop_confirmed.json").is_file():time.sleep(5)
 assert not any(Path(f"/proc/{pid}").exists() for pid in (195922,9010))
 gpu=subprocess.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],capture_output=True,text=True,check=True)
 if gpu.stdout.strip():raise RuntimeError("GPU is still used by another process: "+gpu.stdout)
 config=json.loads((RUN/"run_config.json").read_text())
 checkpoints=[];snapshot=[]
 for fold in range(5):
  directory=RUN/f"fold_{fold}"
  metrics=json.loads((directory/"best_metrics.json").read_text())
  status=json.loads((directory/"status.json").read_text())
  if status.get("completed_epochs",0)<1 or status.get("validated_epochs",0)<1:continue
  path=directory/"best.pt"
  if not path.is_file():continue
  digest=sha(path)
  if digest!=metrics.get("checkpoint_sha256"):raise ValueError(f"fold{fold} best.pt SHA differs from best_metrics")
  if not 1<=metrics["epoch"]<=status["completed_epochs"]:raise ValueError("Invalid best epoch")
  checkpoints.append({"fold":fold,"path":str(path),"sha256":digest,"weight":1.0})
  snapshot.append({"fold":fold,"status":status,"best_metrics":metrics,"checkpoint_sha256":digest})
 if not checkpoints:raise RuntimeError("No validated checkpoints")
 plan={"schema_version":1,"fold_plan":config["fold_plan"],"fold_plan_sha256":sha(config["fold_plan"]),"checkpoints":checkpoints,"tta":{"scales":[768],"hflip":True},"partial_cv_ensemble":len(checkpoints)!=5}
 for key in ("official_valid_manifest_sha256","test_manifest_sha256","sample_submission_sha256"):plan[key]=config[key]
 (OUTPUT/"fold_snapshot_before_inference.json").write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+"\n")
 plan_path=OUTPUT/"ensemble_plan_768_equal.json"
 plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+"\n")
 print("GPU released; frozen fold bests:",[(v["fold"],v["best_metrics"]["epoch"],v["best_metrics"]["validation"]["macro_auc"]) for v in snapshot],flush=True)
 sys.path.insert(0,str(PROJECT))
 import torch
 torch.set_num_threads(4)
 torch.backends.cudnn.benchmark=True
 torch.backends.cuda.matmul.allow_tf32=True
 torch.backends.cudnn.allow_tf32=True
 from inference_overnight import main as inference_main
 destination=OUTPUT/"inference"
 args=["--ensemble-plan",str(plan_path),"--output",str(destination),"--image-root",config["image_root"],"--work-root",str(PROJECT.parent),"--valid-manifest",config["official_valid_manifest"],"--test-manifest",config["test_manifest"],"--sample-submission",config["sample_submission"],"--batch-size","8","--num-workers","8","--prefetch-factor","2","--device","cuda"]
 (OUTPUT/"inference_execution.json").write_text(json.dumps({"started_hong_kong":datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),"arguments":args,"cuda_cudnn_benchmark":True,"TF32":True,"torch_num_threads":4,"training_steps":0},ensure_ascii=False,indent=2)+"\n")
 print("Starting inference only:",args,flush=True)
 inference_main(args)
 submission=destination/"submission_final.csv"
 assert submission.is_file()
 upload=destination/"submission.csv"
 shutil.copyfile(submission,upload)
 assert sha(upload)==sha(submission)
 (OUTPUT/"driver_result.json").write_text(json.dumps({"status":"complete","finished_hong_kong":datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),"submission_final":str(submission),"upload_filename_alias":str(upload),"sha256":sha(submission),"folds":[x["fold"] for x in checkpoints]},ensure_ascii=False,indent=2)+"\n")
 print("INFERENCE COMPLETE",submission,flush=True)

if __name__=="__main__":
 try:main()
 except BaseException as exc:
  (OUTPUT/"driver_error.json").write_text(json.dumps({"status":"failed","error":repr(exc)},ensure_ascii=False,indent=2)+"\n")
  traceback.print_exc()
  raise
