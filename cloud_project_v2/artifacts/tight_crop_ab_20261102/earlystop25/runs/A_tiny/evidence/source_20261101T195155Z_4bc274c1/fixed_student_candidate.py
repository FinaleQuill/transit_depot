"""Reproduce a fixed Tiny/Base probability ensemble on original official valid.

This experimental loader shares production per-image math. It does not train,
fit parameters, load teachers, or accept a test split. Source exports are copied
to a new local candidate; the frozen baseline and original exports stay intact.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil

import numpy as np

from deploy_kd import CLASS_IDS, MEAN, STD, MAX_MODEL_BYTES, Predictor, sha256_file, read_submission
from review_student_upper_bound import fixed_mean, original_labels, read_bound_predictions, write_new
from student_experiment_evaluation import ExperimentalPredictor, evaluate_predictor, score_probabilities


BACKBONES = ('convnext_tiny.fb_in22k_ft_in1k_384', 'convnext_base.fb_in22k_ft_in1k_384')


def make_candidate(models, output, *, sizes, receipt):
    sizes = list(sizes)
    if sizes not in ([768], [640,768,896]):
        raise ValueError('Use original 768 or the predeclared three-scale mean')
    if output.exists():
        raise FileExistsError('Never overwrite an existing candidate')
    inputs = json.loads(Path(receipt).read_text())['models']
    checked = []
    for name, root, expected, identity in zip(('A','B'), models, BACKBONES, inputs):
        root = Path(root).resolve(strict=True)
        config = json.loads((root/'config.json').read_text())
        if (identity['label'] != name or identity['backbone'] != expected
                or config['backbone_name'] != expected or config['image_size'] != 768
                or config['weights_sha256'] != identity['weights_sha256']
                or config['source_checkpoint_sha256'] != identity['source_checkpoint_sha256']
                or sha256_file(root/config['weights_file']) != identity['weights_sha256']):
            raise ValueError('Source A/B identity differs from audited inputs')
        checked.append((name, root, config))
    output.mkdir(parents=True)
    members = []
    for name, root, config in checked:
        destination = output/name
        destination.mkdir()
        for filename in ('config.json', config['weights_file']):
            shutil.copy2(root/filename, destination/filename)
            if sha256_file(root/filename) != sha256_file(destination/filename):
                raise RuntimeError('Candidate asset copy differs from source')
        members.append({'name':name, 'directory':name, 'probability_weight':0.5,
            'config_sha256':sha256_file(destination/'config.json'),
            'weights_sha256':config['weights_sha256'], 'source_epoch':config['source_epoch']})
    config = {'schema_version':1, 'artifact_type':'fixed_experimental_tiny_base_ensemble',
        'class_ids':CLASS_IDS, 'precision':'float16', 'image_sizes':sizes,
        'image_size':sizes[0] if len(sizes)==1 else None,
        'tta':'hflip2', 'ensemble':True, 'gates':None, 'source_members':members,
        'official_valid_manifest_sha256':checked[0][2]['official_valid_manifest_sha256'],
        'aggregation':'mean sigmoid(identity,hflip) per image/scale, equal scale/model means, original Study mean',
        'weights_fitted':False, 'formal_submission_ready':False,
        'teacher_models_loaded':False, 'source_receipt_sha256':sha256_file(receipt)}
    if len({c['official_valid_manifest_sha256'] for _,_,c in checked}) != 1:
        raise ValueError('Students must use the same original holdout')
    write_new(output/'config.json',config)
    asset_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    if asset_bytes>MAX_MODEL_BYTES:
        raise ValueError('Candidate exceeds the inference-asset size limit')
    write_new(output/'asset_report.json',{'actual_inference_bytes':asset_bytes,
        'limit_bytes':MAX_MODEL_BYTES,'size_pass':True,'unique_model_files':2,
        'note':'Each scale reuses the same two fixed on-disk FP16 weights. Code/runtime excluded.'})
    return config


class FixedStudentPredictor(Predictor):
    def __init__(self, model_dir):
        self.model_dir=Path(model_dir).resolve(strict=True)
        config=json.loads((self.model_dir/'config.json').read_text())
        if (config.get('artifact_type')!='fixed_experimental_tiny_base_ensemble'
                or config['class_ids']!=CLASS_IDS or config['tta']!='hflip2'
                or config['image_sizes'] not in ([768],[640,768,896])
                or len(config['source_members'])!=2):
            raise ValueError('Malformed fixed candidate')
        self.config=config
        self.models,self.members,self.model_weights=[],[],[]
        self.unique_models=[]
        for expected,entry in zip(BACKBONES,config['source_members']):
            source=(self.model_dir/entry['directory']).resolve(strict=True)
            if source.parent!=self.model_dir or sha256_file(source/'config.json')!=entry['config_sha256']:
                raise ValueError('Candidate source config changed')
            loaded=ExperimentalPredictor(source,tta='hflip2')
            if (loaded.config['backbone_name']!=expected
                    or loaded.config['weights_sha256']!=entry['weights_sha256']
                    or entry['probability_weight']!=0.5):
                raise ValueError('Candidate member identity or fixed weight changed')
            self.torch,self.device=loaded.torch,loaded.device
            self.unique_models.append(loaded.model)
            for size in config['image_sizes']:
                self.models.append(loaded.model)
                self.members.append({**loaded.members[0],'image_size':size})
                self.model_weights.append([entry['probability_weight']/len(config['image_sizes'])]*10)
            del loaded
        from inference_inputs import ExactU8Input
        self.model=self.models[0]
        self.input_lookup=ExactU8Input(self.torch,self.device,MEAN,STD,image_sizes=config['image_sizes'])
        # Parent Predictor._predict_image decodes each input once and shares
        # transformed tensors. Repeated scales reference the same model objects.
        self.config={**config,'members':self.members}


def development_latency(predictor, inputs, manifest, output, validation):
    from benchmark_local_resources import validate_inputs, validate_round_output
    from cost import time_round, summarize_timings, timing_environment, compare_predictions
    selection=Path(inputs)/'selection.portable.json'
    if not selection.exists(): selection=Path(inputs)/'selection.json'
    original=json.loads(selection.read_text())
    if original['manifest_sha256']!=sha256_file(manifest) or original['origin_split']!='valid':
        raise ValueError('Development timing requires the unchanged valid-only selection')
    output.mkdir()
    relocated=output/'selection.relocated.json'
    write_new(relocated,{**original,'manifest_path':str(Path(manifest).resolve())})
    measured,warmup,coverage=validate_inputs(Path(inputs)/'pictures_100',Path(inputs)/'pictures_5',relocated)
    for row in warmup: predictor.predict_image(row['path'])
    predictor.synchronize()
    durations,comparisons,round_outputs=[],[],[]
    reference=read_submission(validation/'valid_predictions.csv')
    for i in range(3):
        file=output/f'round_{i+1}.csv'
        times,values=time_round(predictor,measured,file)
        comparisons.append(compare_predictions(values,reference,atol=0,allow_reference_superset=True))
        round_outputs.append(validate_round_output(measured,file,values))
        durations.append(times)
    if not all(c['match'] for c in comparisons):
        raise RuntimeError('Timing predictions differ from full valid')
    rounds,per_image,mean=summarize_timings(durations)
    report={'mean_ms':mean,'round_mean_ms':rounds.mean(1).tolist(),
        'per_image_round_ms':rounds.T.tolist(),'p95_ms':float(np.percentile(per_image,95)),
        'coverage':coverage,'comparisons_with_full_valid':comparisons,'round_outputs':round_outputs,
        'environment':timing_environment(predictor),'T4_certified':False,
        'protocol':'100 original valid, 5 separate warmup, batch1 serial3 rounds; end-to-end actual CSV writes'}
    write_new(output/'latency_report.json',report)
    print('LATENCY_MS',mean,flush=True)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,nargs=2,required=True)
    parser.add_argument('--sizes',type=int,nargs='+',required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--image-root',type=Path,required=True)
    parser.add_argument('--references',type=Path,nargs='+',required=True)
    parser.add_argument('--latency-inputs',type=Path)
    args=parser.parse_args(argv)
    from inference_reproducibility import configure_inference
    records,labels,counts=original_labels(args.manifest)
    if len(records)!=1422 or len(labels)!=859 or len(args.references)!=2*len(args.sizes):
        raise ValueError('Require full original valid and one reference for each model/scale')
    sources=[read_bound_predictions(p,args.manifest,labels) for p in args.references]
    expected=fixed_mean(sources,[1]*len(sources))
    args.output.mkdir(parents=True,exist_ok=False)
    model=args.output/'model'
    make_candidate(args.models,model,sizes=args.sizes,receipt=args.receipt)
    configure_inference()
    predictor=FixedStudentPredictor(model)
    validation=args.output/'valid'
    result=evaluate_predictor(predictor,image_root=args.image_root,manifest=args.manifest,output_dir=validation)
    actual=read_submission(validation/'valid_predictions.csv')
    error=max(float(np.max(np.abs(actual[k]-expected[k]))) for k in expected)
    expected_scores=score_probabilities(labels,expected,counts)
    if error>2e-15 or result['per_class_auc']!=expected_scores['per_class_auc']:
        raise RuntimeError('Actual combined predictor differs from saved-probability AUC/reference')
    write_new(args.output/'direct_verification.json',{'max_abs_probability_error':error,
        'all_class_auc_equal':True,'macro_auc':result['macro_auc'],
        'evaluation_source_sha256':sha256_file(__file__), 'test_used':False,
        'references':[{'directory':str(p),'predictions_sha256':sha256_file(p/'valid_predictions.csv')} for p in args.references]})
    print('ACTUAL_MACRO_AUC',result['macro_auc'],flush=True)
    if args.latency_inputs:
        development_latency(predictor,args.latency_inputs,args.manifest,args.output/'latency',validation)
    del predictor
    gc.collect()


if __name__=='__main__':
    main()
