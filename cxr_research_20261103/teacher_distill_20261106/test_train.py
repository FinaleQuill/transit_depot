"""CPU lifecycle/coverage regressions; no public model construction or GPU work."""
from __future__ import annotations

from contextlib import ExitStack, nullcontext
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from teacher_distill_20261106 import train as trainer


class PlainScaler:
    def scale(self, loss): return loss
    def unscale_(self, optimizer): pass
    def get_scale(self): return 1.
    def step(self, optimizer): optimizer.step()
    def update(self): pass
    def state_dict(self): return {}


class SyntheticStudies:
    def __init__(self, count=5, *, duplicate=False):
        self.samples=[]
        self.groups=[]
        self.records=[]
        self.image_size=512
        for i in range(count):
            image_count=(1,2,1,3,2)[i%5]
            key=f'p{i}/s{i}'
            ids=[f'i{i}_{j}' for j in range(image_count)]
            start=len(self.records)
            target=tuple([i%2]*10)
            self.records.extend(dict(targets=target) for _ in ids)
            self.groups.append(list(range(start,start+image_count)))
            self.samples.append(dict(images=torch.ones(image_count,3,8,8),
                targets=torch.tensor(target,dtype=torch.float32),study_key=key,
                subject_id=f'p{i}',image_ids=ids))
        if duplicate and count>1:
            self.samples[-1]['study_key']=self.samples[0]['study_key']
            self.samples[-1]['image_ids']=self.samples[0]['image_ids']

    def __len__(self): return len(self.groups)
    def __getitem__(self,index): return self.samples[index]
    def set_epoch(self,epoch): self.epoch=epoch


class TinyTeacher(torch.nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.head=torch.nn.Sequential(torch.nn.Linear(1,10))
        self.image_size=kwargs.get('image_size',512)
        self.sizes=[]
        self.forward_sizes=[]

    def cuda(self): return self
    def set_image_size(self,size):
        self.image_size=size;self.sizes.append(size)

    def optimizer_parameter_groups(self,**kwargs):
        return [dict(params=list(self.parameters()),lr=.01,base_lr=.01)]

    def forward_studies(self,images):
        values=[]
        for image in images:
            if image.shape[-1]!=self.image_size:
                raise ValueError('Input resolution differs from set_image_size')
            self.forward_sizes.append(image.shape[-1])
            values.append(self.head(image.mean().reshape(1)).float())
        return values


def cpu_device_adapters(stack):
    """Adapt only CUDA transport/autocast; retain real loss/backward/optimizer."""
    original_to=torch.Tensor.to
    def tensor_to(tensor,*args,**kwargs):
        if args and isinstance(args[0],str) and args[0].startswith('cuda'):
            args=('cpu',)+args[1:]
        if str(kwargs.get('device','')).startswith('cuda'):
            kwargs=dict(kwargs,device='cpu')
        return original_to(tensor,*args,**kwargs)
    stack.enter_context(patch.object(torch.Tensor,'to',tensor_to))
    stack.enter_context(patch.object(torch,'autocast',lambda *a,**k:nullcontext()))
    stack.enter_context(patch.object(torch.cuda,'reset_peak_memory_stats',return_value=None))
    stack.enter_context(patch.object(torch.cuda,'max_memory_allocated',return_value=0))


class EpochCoverageTests(unittest.TestCase):
    def execute(self,dataset,loader=None):
        model=TinyTeacher(image_size=8);ema=deepcopy(model)
        optimizer=torch.optim.SGD([dict(params=list(model.parameters()),lr=.1,base_lr=.1)])
        progress=dict(attempted_updates=0,optimizer_updates=0,skipped_updates=0)
        config=dict(workers=0,seed=20261106,pack_images=1,effective_studies=4,
            clip_grad=1.,epochs=2,ema_decay=.999)
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            cpu_device_adapters(stack)
            if loader is not None:
                stack.enter_context(patch('breakthrough.train.make_loader',return_value=loader))
            metrics=trainer.train_epoch(model,ema,dataset,optimizer,PlainScaler(),
                torch.ones(10),config,1,progress,Path(tmp))
        return metrics,progress

    def test_epoch_covers_every_study_all_images_and_tail_update(self):
        dataset=SyntheticStudies()
        metrics,progress=self.execute(dataset)
        self.assertEqual(metrics['studies'],5)
        self.assertEqual(metrics['images'],9)
        self.assertEqual(progress['optimizer_updates'],2)
        self.assertEqual(progress['skipped_updates'],0)
        self.assertEqual(dataset.epoch,1)

    def test_repeated_study_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Repeated Study'):
            self.execute(SyntheticStudies(duplicate=True))

    def test_truncated_loader_is_rejected(self):
        dataset=SyntheticStudies()
        with self.assertRaisesRegex(ValueError,'Incomplete epoch coverage'):
            self.execute(dataset,loader=dataset.samples[:-1])


class OofTests(unittest.TestCase):
    def test_real_oof_function_switches_both_resolutions_and_keeps_all_views(self):
        class OofData:
            def __init__(self,rows,root,*,image_size,training):
                self.size=image_size
                self.groups=[[0],[1]]
            def __len__(self): return len(self.groups)
            def __getitem__(self,i):
                return dict(images=torch.ones(i+1,3,self.size,self.size),
                    targets=torch.tensor([i]*10,dtype=torch.float32),
                    study_key=f'p{i}/s{i}',subject_id=f'p{i}')
        model=TinyTeacher(image_size=768).eval()
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            cpu_device_adapters(stack)
            stack.enter_context(patch.object(trainer,'StudyDataset',OofData))
            stack.enter_context(patch.object(trainer,'make_loader',
                side_effect=lambda dataset,**kwargs:[dataset[i] for i in range(len(dataset))]))
            stack.enter_context(patch.object(trainer,'roc_auc_score',return_value=.5))
            config=dict(resolution_schedule=[dict(through_epoch=1,image_size=768)],
                oof_image_sizes=[512,768],workers=0,seed=20261106,image_root=tmp,
                checkpoint_dir=tmp)
            report=trainer.predict_oof(model,[{'unused':'metadata'}],config,Path(tmp))
            saved=np.load(Path(tmp)/'oof_predictions.npz',allow_pickle=False)
            self.assertEqual(saved['weighted_study_logits'].shape,(2,4,10))
            self.assertEqual(report['studies'],2)
            self.assertFalse(report['H_images_accessed'])
            self.assertEqual(model.sizes,[512,768])
            self.assertEqual(model.forward_sizes,[512]*4+[768]*4)


class FixedLifecycleTests(unittest.TestCase):
    def test_run_uses_all_fixed_epochs_final_ema_and_never_builds_H_dataset(self):
        rows=dict(train=[{'subject_id':'train_patient'}],
                  oof=[{'study_key':'oof_patient/oof'}],H=[{'study_key':'H_patient/H'}])
        dataset=SyntheticStudies(2)
        epochs=[];predicted=[];snapshots=[]
        def build_dataset(records,*args,**kwargs):
            self.assertIs(records,rows['train'])
            return dataset
        def do_epoch(model,ema,train,opt,scaler,weights,config,epoch,progress,output):
            epochs.append((epoch,train.image_size,model.image_size,ema.image_size))
            with torch.no_grad(): ema.head[-1].weight.fill_(epoch)
            progress['attempted_updates']+=1;progress['optimizer_updates']+=1
            # Worsening loss must not cause checkpoint/epoch selection.
            return dict(epoch=epoch,loss=float(epoch),studies=2,images=3,**progress)
        def final_prediction(model,records,config,output,**kwargs):
            self.assertIs(records,rows['oof'])
            predicted.append(float(model.head[-1].weight[0,0]))
            return dict(studies=1,H_images_accessed=False)
        def checkpoint(path,payload,epoch,role):
            snapshots.append((Path(path).name,epoch,role));return f'{epoch:064x}'
        original_tensor,original_as_tensor=torch.tensor,torch.as_tensor
        def on_cpu(function):
            def call(*args,**kwargs):
                if str(kwargs.get('device','')).startswith('cuda'):
                    kwargs=dict(kwargs,device='cpu')
                return function(*args,**kwargs)
            return call
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);output=root/'output';state=root/'state'
            config=dict(output=str(output),checkpoint_dir=str(state),family='dinov3_l',
                pretrained_path='not-opened.safetensors',image_root=tmp,seed=20261106,epochs=2,
                resolution_schedule=[dict(through_epoch=1,image_size=512),dict(through_epoch=2,image_size=768)],
                drop_rate=.2,drop_path=.1,view_chunk_size=1,backbone_lr=1e-5,head_lr=2e-4,
                weight_decay=.05,layer_decay=.9,train_manifest='synthetic',selection_policy='fixed_final_ema_no_H_selection')
            for name,value in dict(verify_inputs=lambda config:(rows,{}),
                source_evidence=lambda:([],{}),start_run=lambda *a,**k:None,
                StudyDataset=build_dataset,TeacherStudyModel=TinyTeacher,train_epoch=do_epoch,
                predict_oof=final_prediction,save_checkpoint=checkpoint,
                EVIDENCE=root).items():
                stack.enter_context(patch.object(trainer,name,value))
            stack.enter_context(patch.object(trainer.shutil,'disk_usage',return_value=SimpleNamespace(free=50_000_000_000)))
            stack.enter_context(patch.object(torch.amp,'GradScaler',return_value=PlainScaler(),create=True))
            stack.enter_context(patch.object(torch,'tensor',on_cpu(original_tensor)))
            stack.enter_context(patch.object(torch,'as_tensor',on_cpu(original_as_tensor)))
            result=trainer.run.__wrapped__(config)
        self.assertEqual(epochs,[(1,512,512,512),(2,768,768,768)])
        self.assertEqual(predicted,[2.])
        self.assertEqual(result['completed_epoch'],2)
        self.assertEqual(result['state'],'completed_fixed_schedule')
        self.assertEqual([e for name,e,role in snapshots if name=='last.pt'],[1,2])
        self.assertTrue(all('best' not in name for name,_,_ in snapshots))


@unittest.skipUnless(Path('/home/featurize/data/kd_rebuild_20261106/_SUCCESS.json').is_file(),
    'Requires the prepared original-data metadata; no images or models are loaded')
class RegisteredInputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.plan=Path('/home/featurize/data/kd_rebuild_20261106')
        weights=self.root/'public.safetensors';weights.write_bytes(b'synthetic public asset identity')
        provenance=self.root/'source.json'
        revision='a'*40
        provenance.write_text(json.dumps(dict(repo='timm/vit_large_patch16_dinov3.lvd1689m',
            gated=False,license='dinov3-license',revision=revision,sha256=trainer.digest(weights))))
        environment=self.root/'environment.json';environment.write_text('{"synthetic_cpu_test":true}')
        self.config=dict(family='dinov3_l',inner_fold=0,pretrained_revision=revision,
            source_manifest=str(self.plan/'source_train.csv'),R_manifest=str(self.plan/'R.csv'),
            H_manifest=str(self.plan/'H.csv'),train_manifest=str(self.plan/'inner_0/train.csv'),
            oof_manifest=str(self.plan/'inner_0/oof.csv'),fold_plan=str(self.plan/'plan.json'),
            pretrained_path=str(weights),provenance_path=str(provenance),epochs=2,
            resolution_schedule=[dict(through_epoch=1,image_size=512),dict(through_epoch=2,image_size=768)],
            selection_policy='fixed_final_ema_no_H_selection',automatic_submission=False,
            output=str(self.root/'evidence/run'),checkpoint_dir=str(self.root/'experiment/state/run'),
            environment_manifest=str(environment),environment_manifest_sha256=trainer.digest(environment),
            source_sha256=trainer.source_evidence()[1])
        keys=('source_manifest','R_manifest','H_manifest','train_manifest','oof_manifest',
              'fold_plan','pretrained_path','provenance_path')
        self.config['input_sha256']={k:trainer.digest(self.config[k]) for k in keys}
        self.patches=ExitStack();self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.object(trainer,'EVIDENCE',self.root/'evidence'))
        self.patches.enter_context(patch.object(trainer,'EXPERIMENT',self.root/'experiment'))

    def test_metadata_matches_loader_contract_and_H_images_are_not_opened(self):
        with patch.object(trainer,'StudyDataset',side_effect=AssertionError('Metadata verification opened images')):
            rows,_=trainer.verify_inputs(self.config)
        self.assertEqual(len(rows['source']),30344)
        self.assertEqual(len(rows['H']),6067)
        self.assertIsInstance(rows['train'][0]['targets'],tuple)

    def test_changed_input_is_rejected(self):
        self.config['input_sha256']['train_manifest']='0'*64
        with self.assertRaises(ValueError):trainer.verify_inputs(self.config)

    def test_inner_fold_cannot_be_relabelled_while_manifests_stay_fixed(self):
        self.config['inner_fold']=1
        with self.assertRaises(ValueError):trainer.verify_inputs(self.config)

    def test_changed_source_is_rejected_before_a_new_teacher(self):
        self.config['source_sha256']=dict(self.config['source_sha256'])
        self.config['source_sha256']['teacher_distill_20261106/train.py']='0'*64
        with self.assertRaises(ValueError):trainer.verify_inputs(self.config)

    def test_H_selection_and_old_run_outputs_are_rejected(self):
        with self.subTest('selection'):
            bad=dict(self.config,selection_policy='best_H_auc')
            with self.assertRaises(ValueError):trainer.verify_inputs(bad)
        with self.subTest('old output'):
            bad=dict(self.config,output='/home/featurize/work/reconstruction_runs_20261103/BREAKTHROUGH_20261105/RAW_REBUILD')
            with self.assertRaises(ValueError):trainer.verify_inputs(bad)


if __name__=='__main__':
    unittest.main()
