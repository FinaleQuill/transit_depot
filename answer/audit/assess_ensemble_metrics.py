"""Independent CPU assessment of saved official-valid probabilities."""
from pathlib import Path
import csv,json
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score

ROOT=Path(__file__).resolve().parent
DEST=ROOT/"inference"
manifest=Path("/home/featurize/work/cloud_project_v2/data/clean_v1_reviewed/valid_manifest.csv")

def main():
 labels={}
 with manifest.open(encoding="utf-8-sig",newline="") as f:
  for row in csv.DictReader(f):
   key=(row["subject_id"],row["study_id"]);target=tuple(json.loads(row["targets"]))
   if key in labels:assert labels[key]==target
   labels[key]=target
 with (DEST/"valid_study_probabilities.csv").open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
 keys=[(r["Subject_id"],r["Study_id"]) for r in rows]
 assert len(keys)==len(set(keys))==len(labels)==859 and set(keys)==set(labels)
 y=np.asarray([labels[k] for k in keys],dtype=np.int64)
 p=np.asarray([[float(r[f"class_{c}"]) for c in range(1,11)] for r in rows],dtype=np.float64)
 assert p.shape==y.shape==(859,10) and np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
 saved=json.loads((DEST/"valid_metrics.json").read_text())
 gates=np.asarray(json.loads((DEST/"thresholds.json").read_text())["thresholds"],dtype=np.float64)
 auc=float(roc_auc_score(y,p,average="macro"))
 fixed_f1=float(f1_score(y,p>=0.5,average="macro",zero_division=0))
 tuned_f1=float(f1_score(y,p>=gates[None,:],average="macro",zero_division=0))
 assert abs(auc-saved["macro_auc"])<1e-12
 assert abs(tuned_f1-saved["macro_f1"])<1e-12
 report={"scope":"original official valid, held out from all 5 fold training","studies":859,"macro_auc":auc,"macro_ap":float(average_precision_score(y,p,average="macro")),"macro_f1_at_0_5":fixed_f1,"macro_f1_at_saved_valid_fitted_thresholds":tuned_f1,"thresholds":gates.tolist(),"f1_tuned_on_same_valid":True,"probability_submission_uses_gate_column":False,"probability_submission_default_gate":0.5,"per_class_auc":roc_auc_score(y,p,average=None).tolist(),"same_protocol_as_test":"five equal fold EMA bests; native 768 identity plus horizontal-flip TTA"}
 (ROOT/"ensemble_assessment.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
 print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
