#!/usr/bin/env python3
"""Independent standard-library CPU audit; no GPU imports or source edits."""
from pathlib import Path
import csv, hashlib, itertools, json, math
from collections import defaultdict
from datetime import datetime, timezone

ROOT=Path('/home/featurize/work/dual768_proxy_timing_results')
BUNDLE=Path('/home/featurize/work/deliveries/034_dual768_proxy_timing_20261029/bundle')

def readj(p): return json.loads(p.read_text())
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
def rows(p):
 with p.open(newline='') as f: return list(csv.DictReader(f))
def predictions(p):
 rr=rows(p); d={}
 for r in rr:
  keys=['Image_id','Subject_id','Study_id'] if 'Image_id' in r else ['Subject_id','Study_id']
  k=tuple(r[x] for x in keys)+(int(r['Predict_class']),)
  assert k not in d
  d[k]=float(r['Probability']); assert math.isfinite(d[k]) and 0<=d[k]<=1
 samples={k[:-1] for k in d}
 assert all({k[-1] for k in d if k[:-1]==s}==set(range(1,11)) for s in samples)
 return d

def compare(a,b):
 a,b=predictions(a),predictions(b); assert a.keys()==b.keys()
 dd=[abs(a[k]-b[k]) for k in a]
 within=[]; across=[]; samples=sorted({k[:-1] for k in a})
 def sign(x): return (x>0)-(x<0)
 for s in samples:
  for c,d in itertools.combinations(range(1,11),2):
   if sign(a[s+(c,)]-a[s+(d,)])!=sign(b[s+(c,)]-b[s+(d,)]): within.append([list(s),c,d])
 if len(next(iter(a)))==3:
  for c in range(1,11):
   for s,t in itertools.combinations(samples,2):
    if sign(a[s+(c,)]-a[t+(c,)])!=sign(b[s+(c,)]-b[t+(c,)]):
     across.append({'class':c,'study_a':list(s),'study_b':list(t),'current_probabilities':[a[s+(c,)],a[t+(c,)]],'reference_probabilities':[b[s+(c,)],b[t+(c,)]]})
 return {'rows':len(a),'samples':len(samples),'max_abs':max(dd),'mean_abs':sum(dd)/len(dd),'gt_1e_minus_3':sum(v>1e-3 for v in dd),'binary_changes':sum((a[k]>=.5)!=(b[k]>=.5) for k in a),'within_sample_order_or_tie_changes':within,'across_study_order_or_tie_changes':across}

def pct(v,p):
 s=sorted(v); i=(len(s)-1)*p/100; a=math.floor(i); b=math.ceil(i)
 return s[a]+(s[b]-s[a])*(i-a)

main=readj(ROOT/'main_run.json'); native=Path(main['output_dir']); report=readj(native/'local_resource_report.json')
diag_info=readj(ROOT/'diagnostic_run.json'); diag=Path(diag_info['output_dir']); dr=readj(diag/'diagnostic_report.json')
checks={}; details={}
def check(name,condition): checks[name]=bool(condition); assert condition,name
check('main_exit_code_zero', (ROOT/(main['run_id']+'.exit_code.txt')).read_text().strip()=='0')
receipt=readj(ROOT/(main['run_id']+'.execution_receipt.json'))
check('receipt_exit_zero',receipt['exit_code']==0)
check('main_uninstrumented',main['instrumented'] is False)
check('protocol_original',report['protocol']=={'measured_images':100,'warmup_images':5,'rounds':3,'precision':'float16','batch_size':1,'execution':'serial','timing':'read_file_to_prediction_CSV_write_including_preprocess_postprocess','model_initialization_timed':False,'aggregation':'per_image_three_round_mean_then_100_image_mean'})
check('proxy_fields_false',report['T4_latency_certified'] is False and report['official_compliance_pass'] is False)
check('original_atol_1e_minus_7', all(x['absolute_tolerance']==1e-7 and x['match'] for x in report['comparison']['rounds']))
sha_matches=[]
for line in (BUNDLE/'SHA256SUMS').read_text().splitlines():
 expected,name=line.split(maxsplit=1); path=BUNDLE/name.lstrip('*')
 sha_matches.append({'path':str(path.relative_to(BUNDLE)),'sha256':digest(path),'expected':expected})
check('all_131_sha256sum_entries_match',len(sha_matches)==131 and all(x['sha256']==x['expected'] for x in sha_matches))
details['bundle_integrity']={'files':len(sha_matches),'all_match':True}
config=readj(BUNDLE/'model/033/config.json')
check('config_sha_frozen',digest(BUNDLE/'model/033/config.json')=='9363149e143dec2871ac5fb4fbfdee4885f9f62add6a9f2acf4704635311b590')
check('seeds_43_44', [m['seed'] for m in config['members']]==[43,44])
check('fixed_25_75',config['model_weights']==[[.25]*10,[.75]*10])
check('two_768_hflip2',len(config['members'])==2 and all(m['image_size']==768 and m['tta']=='hflip2' for m in config['members']))
check('gpu_sm68_allocator8',report['resources']['mode']=='native' and report['resources']['cuda_visible_sm_count']==68 and report['resources']['allocator']['requested_limit_gib']==8)
# Frozen source establishes sequential member/view loops, a batch dimension of one,
# FP16 input conversion, and benchmark 5-image warmup followed by three rounds.
source=(BUNDLE/'deploy_kd.py').read_text(); benchmark=(BUNDLE/'benchmark_local_resources.py').read_text()
check('frozen_source_serial_four_forward_structure','for member, model in zip(self.members, self.models):' in source and 'for view in views:' in source and 'logits = model(transformed)' in source and 'unsqueeze(0)' in source and 'dtype=torch.float16' in source and 'for index in range(3):' in benchmark and 'atol=1e-7' in benchmark)
csv_records=[]; reference_results=[]
for i in range(1,4):
 for image in [False,True]:
  name=f'round_{i}'+('.image_predictions.csv' if image else '.csv'); path=native/name
  values=predictions(path); expected=report['round_outputs'][i-1]['image_predictions_sha256' if image else 'sha256']
  check(f'{name}_hash_and_coverage',digest(path)==expected and len(values)==(1000 if image else 530))
  reference_results.append({'round':i,'level':'image' if image else 'study',**compare(path,BUNDLE/'reference_4090'/name)})
  csv_records.append({'file':name,'sha256':digest(path),'rows':len(values)})
check('three_study_csvs_identical',len({x['sha256'] for x in csv_records if '.image_' not in x['file']})==1)
check('three_image_csvs_identical',len({x['sha256'] for x in csv_records if '.image_' in x['file']})==1)
details['csvs']=csv_records; details['cross_4090_independently_recomputed']=reference_results
check('reported_4090_summary_accurate',all(x['max_abs']==0.0002449899911880493 and x['gt_1e_minus_3']==0 and x['binary_changes']==0 and not x['within_sample_order_or_tie_changes'] and x['mean_abs']==(2.4192805275561112e-05 if x['level']=='image' else 1.9770578710048648e-05) and len(x['across_study_order_or_tie_changes'])==(0 if x['level']=='image' else 1) and all(z['class']==3 for z in x['across_study_order_or_tie_changes']) for x in reference_results))
mat=report['per_image_round_ms']; order=report['image_order']
check('timing_100_by_3_finite',len(mat)==len(order)==len(set(order))==100 and all(len(row)==3 and all(math.isfinite(v) and v>=0 for v in row) for row in mat))
means=[sum(row)/3 for row in mat]; flat=[v for row in mat for v in row]
timing_rows=rows(ROOT/'comparison_with_4090/all_300_timing_observations.csv')
check('exported_300_observations_exactly_retained',len(timing_rows)==300 and all(r['Image_id']==order[n//3] and int(r['round'])==n%3+1 and float(r['milliseconds'])==flat[n] for n,r in enumerate(timing_rows)))
recomputed={'round_means':[sum(row[i] for row in mat)/100 for i in range(3)],'mean':sum(flat)/300,'per_image_mean_p95':pct(means,95),'per_image_mean_max':max(means),'all_300_p95':pct(flat,95),'all_300_max':max(flat),'raw_gt_100ms':sum(v>100 for v in flat)}
check('timing_summary_matches',abs(recomputed['mean']-report['mean_ms'])<1e-10 and abs(recomputed['per_image_mean_p95']-report['p95_ms'])<1e-10 and recomputed['per_image_mean_max']==report['max_ms'] and all(abs(a-b)<1e-10 for a,b in zip(recomputed['round_means'],report['round_mean_ms'])))
details['timing']=recomputed
check('diagnostic_separate_after_main',native!=diag and diag_info['diagnostic_only'] and diag_info['instrumented'] and not diag_info['replaces_primary_benchmark'] and diag_info['started_utc']>receipt['ended_utc'])
check('diagnostic_single_round_exit_zero',dr['protocol']['rounds']==1 and dr['exit_code']==0 and dr['protocol']['measured_images']==100)
segments=readj(diag/'per_image_segments.json')['images']
check('diagnostic_100_images_four_forward_spans',len(segments)==100 and all(len(s['forward_event_spans_ms'])==4 and s['load_calls']==1 and s['prediction_completed'] for s in segments))
details['diagnostic_vs_native_round1']={}
for image in [False,True]:
 name='round_1'+('.image_predictions.csv' if image else '.csv')
 result=compare(diag/name,native/name)
 details['diagnostic_vs_native_round1'][name]=result
 check('diagnostic_'+name+'_same_predictions',result['max_abs']==0 and digest(diag/name)==digest(native/name))
for file in [ROOT/(main['run_id']+'.console.log'),ROOT/(main['run_id']+'.execution_receipt.txt'),ROOT/'comparison_with_4090.console.log',ROOT/'comparison_with_4090.exit_code.txt']:
 check('preserved_'+file.name,file.exists())
result={'created_utc':datetime.now(timezone.utc).isoformat(),'cpu_only':True,'audit_pass':all(checks.values()),'checks':checks,'details':details,'scope_note':'Independent artifact audit only. Proxy measurements do not constitute T4 certification or official test consistency. The class-3 cross-Study inversion is retained and is not called prediction-semantic compliance.'}
out=ROOT/'analysis/final_independent_audit.json'
with out.open('x') as f: json.dump(result,f,indent=2); f.write('\n')
print(json.dumps({'audit_pass':result['audit_pass'],'checks':len(checks),'output':str(out),'timing':recomputed},indent=2))
