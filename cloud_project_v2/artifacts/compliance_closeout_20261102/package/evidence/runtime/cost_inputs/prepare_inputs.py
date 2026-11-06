from pathlib import Path
import csv,json,hashlib,shutil,os,datetime,importlib.util
R=Path('/home/featurize/work/cloud_project_v2');OUT=R/'artifacts/compliance_closeout_20261102/cost_inputs';OUT.mkdir(parents=True,exist_ok=True)
M=R/'data/clean_v1_reviewed/test_manifest.csv';SIDE=R.parent/'deliveries/041_tiny768_p2_A_submission_20261030/submission.image_predictions.csv';RAW=Path('/home/featurize/data/cxr_first')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
fields=['image_id','split','subject_id','study_id','study_key','relative_path','file_sha256']
# Discard all non-identity fields immediately. No labels/probabilities are used.
with M.open(encoding='utf-8-sig',newline='') as f:rows=[{k:r[k] for k in fields} for r in csv.DictReader(f)]
groups={}
for r in rows:groups.setdefault((r['subject_id'],r['study_id']),[]).append(r)
keys=sorted(groups)
def choose(target,exclude):
 picked=[];count=0;skipped=[]
 for key in keys:
  if key in exclude:continue
  if count==target:break
  if count+len(groups[key])<=target:picked.append(key);count+=len(groups[key])
  else:skipped.append(key)
 if count!=target:raise ValueError('deterministic complete-Study greedy selection could not hit exact count; no fallback permitted')
 return picked,skipped
measured,mskip=choose(100,set());warm,wskip=choose(5,set(measured))
# Sidecar identity audit only; never parse Probability or Predict_class values.
with SIDE.open(encoding='utf-8-sig',newline='') as f:
 ids={(r['Subject_id'],r['Study_id'],r['Image_id']) for r in csv.DictReader(f)}
selected={};records=[]
for label,picked in [('pictures_100',measured),('pictures_5',warm)]:
 root=OUT/label
 if root.exists():raise FileExistsError(root)
 root.mkdir()
 selected[label]=[]
 for key in picked:
  for r in sorted(groups[key],key=lambda x:(Path(x['relative_path']).name,x['image_id'])):
   assert (r['subject_id'],r['study_id'],r['image_id']) in ids
   src=RAW/r['relative_path'];dst=root/r['subject_id']/r['study_id']/Path(r['relative_path']).name;dst.parent.mkdir(parents=True,exist_ok=True)
   before=sha(src);assert before==r['file_sha256'],src
   # Source and target are on distinct devices. Hardlink not possible; symlink escapes rejected by frozen discover_images.
   shutil.copyfile(src,dst)
   copied=sha(dst);assert copied==before
   entry={**r,'subset':label,'target_relative_path':dst.relative_to(OUT).as_posix(),'size_bytes':dst.stat().st_size,'source_sha256':before,'copied_sha256':copied,'method':'byte_identical_copy_due_cross_device_and_frozen_symlink_guard'}
   selected[label].append(entry);records.append(entry)
# Minimal ORIGINAL FULL identity manifest proves complete Studies without carrying labels/quality attributes.
manifest=OUT/'original_test_identity_manifest.csv'
with manifest.open('w',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
for name in selected:
 p=OUT/(name+'_selected_manifest.csv')
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(selected[name])
# Validate through frozen deployment functions only, no Predictor construction, no image decode/model forward.
module_path=R/'artifacts/compliance_closeout_20261102/package/code/deploy_kd.py';spec=importlib.util.spec_from_file_location('cost_input_static_deploy',module_path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
coverage={}
for name,count in [('pictures_100',100),('pictures_5',5)]:
 found=mod.discover_images(OUT/name);assert len(found)==count
 coverage[name]=mod.match_timing_manifest(found,manifest);assert coverage[name]['complete_studies_proven']
receipt={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'User-authorized cost.py vs historical041 submission consistency only; no selection/optimization/training','official_sample':False,'sample_scope':'Self-selected fixed complete-Study identity subset; not official-designated100. Existing official sample knowledge is not used to claim certification.','selection_algorithm':'Sort all original manifest Study keys lexicographically by (Subject_id,Study_id), take whole Study if it fits remaining count until exactly100. For warmup repeat same order excluding measured Studies until exactly5. Sort image files by filename within Study. No probability, label, view, geometry, image quality, predicted class or score used.','selected_using_probabilities':False,'selected_using_labels':False,'model_forward_executed':False,'original_manifest_path':str(M),'original_manifest_sha256':sha(M),'identity_manifest_path':manifest.name,'identity_manifest_sha256':sha(manifest),'sidecar_path':str(SIDE),'sidecar_sha256':sha(SIDE),'sidecar_fields_used':['Image_id','Subject_id','Study_id'],'measured_images':100,'measured_studies':len(measured),'warmup_images':5,'warmup_studies':len(warm),'measured_warmup_study_overlap':0,'measured_skipped_for_exact_count':mskip,'warmup_skipped_for_exact_count':wskip,'source_device':os.stat(RAW).st_dev,'target_device':os.stat(OUT).st_dev,'link_limitation':'Cross-device hardlinks unavailable; frozen discover_images rejects symlinks escaping timing root. Used byte-identical local copies, never modifies original files.','full_study_coverage':coverage,'selected_files':records,'total_copied_bytes':sum(x['size_bytes'] for x in records),'note':'Only105 chosen image byte streams were read for SHA/copy; no image decoding or unselected image reads. This is numerical consistency preparation, not official timing Gate evidence.'}
(OUT/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
shutil.copy2(__file__,OUT/'prepare_inputs.py')
print(json.dumps({k:receipt[k] for k in ['measured_images','measured_studies','warmup_images','warmup_studies','total_copied_bytes','full_study_coverage']},ensure_ascii=False))
print(OUT)
