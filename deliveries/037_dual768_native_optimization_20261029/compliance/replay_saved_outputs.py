"""CPU consistency replay only; consumes saved v1 outputs, never runs a model."""
from pathlib import Path
import sys,json
sys.path.insert(0,"/home/featurize/work/cloud_project_v2")
import cost
from deploy_kd import discover_images,read_submission
r=Path("/home/featurize/work/deliveries/036_dual768_2080ti_diagnosis_20261029")
records=discover_images("/home/featurize/work/deliveries/034_dual768_proxy_timing_20261029/bundle/inputs/pictures_100")
reference=cost.read_image_probabilities(r/"benchmark_nfs_v1/round_1.image_predictions.csv")
ids=cost.source_image_ids(records,None)
image=cost.compare_image_output(records,r/"benchmark_local_v1/round_1.image_predictions.csv",reference,ids)
study=cost.compare_predictions(read_submission(r/"benchmark_local_v1/round_1.csv"),read_submission(r/"benchmark_nfs_v1/round_1.csv"))
report={"cpu_only":True,"models_executed":False,"saved_image_comparison":image,"saved_study_comparison":study}
assert image["match"] and image["compared_images"]==100 and study["match"]
Path("/home/featurize/work/deliveries/037_dual768_native_optimization_20261029/compliance/saved_output_replay.json").write_text(json.dumps(report,indent=2)+"\n")
print({"image_match":image["match"],"images":image["compared_images"],"study_match":study["match"],"studies":study["compared_studies"]})
