"""Independent CPU checks of split boundaries and actual optimizer transitions."""
from contextlib import ExitStack, nullcontext, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from breakthrough import train as trainer


class FakeDataset:
    def __init__(self, sizes=(1, 5, 2, 1, 3), dropout=False):
        self.groups, self.batches, self.records = [], [], []
        for i, views in enumerate(sizes):
            ids = [f'{i}-{v}' for v in range(views)]
            self.groups.append(ids)
            self.batches.append(dict(images=torch.randn(views, 3, 2, 2),
                targets=torch.tensor([(i + c) % 2 for c in range(10)], dtype=torch.float32),
                study_key=f'p{i}/s{i}', subject_id=f'p{i}', image_ids=ids))
        self.epoch = None
    def __len__(self):
        return len(self.groups)
    def set_epoch(self, epoch):
        self.epoch = epoch


class TinyStudy(nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(3, 10)
    def forward_studies(self, images):
        # Different view counts never change the target's loss weight.
        return [self.fc(self.drop(x.mean((0, 2, 3)))) for x in images]


class CPUScaler:
    def __init__(self, skips=()):
        self.value = 1024.
        self.calls = 0
        self.skips = set(skips)
    def scale(self, loss):
        return loss
    def unscale_(self, optimizer):
        pass
    def get_scale(self):
        return self.value
    def step(self, optimizer):
        self.calls += 1
        if self.calls not in self.skips:
            optimizer.step()
    def update(self):
        if self.calls in self.skips:
            self.value /= 2.


class TrainerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        torch.set_num_threads(1)

    def fixture(self, directory):
        rows, mapping = [], {}
        for fold in range(5):
            for positive in (0, 1):
                i = fold * 2 + positive
                row = dict(split='train', training_eligible='True', image_id=f'i{i}',
                    subject_id=f'p{i}', study_id=f's{i}', study_key=f'p{i}/s{i}',
                    targets=tuple([positive]*10), file_sha256=hashlib.sha256(f'f{i}'.encode()).hexdigest(),
                    pixel_sha256=hashlib.sha256(f'x{i}'.encode()).hexdigest())
                rows.append(row); mapping[row['study_key']] = fold
        source = directory/'source.csv'; source.write_text('synthetic-source')
        train = directory/'train.csv'; train.write_text('synthetic-train')
        dev = directory/'dev.csv'; dev.write_text('synthetic-dev')
        weight = directory/'model.safetensors'; weight.write_bytes(b'synthetic-public-encoder')
        provenance = directory/'provenance.json'
        provenance.write_text(json.dumps(dict(sha256=trainer.digest(weight),
            repo='timm/vit_base_patch14_reg4_dinov2.lvd142m', license='apache-2.0',
            gated=False, pretraining_dataset='LVD-142M')))
        fold_plan = directory/'folds.json'; fold_plan.write_text(json.dumps(dict(
            source_manifest_sha256=trainer.digest(source), study_to_fold=mapping)))
        config = dict(source_manifest=str(source), train_manifest=str(train), dev_manifest=str(dev),
            pretrained_path=str(weight), provenance_path=str(provenance), fold_plan=str(fold_plan))
        config['input_sha256'] = {key: trainer.digest(value) for key,value in config.items()}
        by_path = {str(source): rows, str(train): rows[:8], str(dev): rows[8:]}
        return config, by_path

    def test_real_parsed_targets_pass_strict_80_20_fold_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, by_path = self.fixture(Path(tmp))
            with patch.object(trainer, 'read_records', side_effect=lambda p: deepcopy(by_path[str(p)])):
                tr, dev, provenance = trainer.verify_inputs(config)
            self.assertEqual((len(tr), len(dev)), (8, 2))
            self.assertFalse(set(x['subject_id'] for x in tr) & set(x['subject_id'] for x in dev))

    def test_relabelled_missing_wrongfold_and_official_valid_rows_are_rejected(self):
        for violation in ('labels', 'missing', 'wrongfold', 'official_valid'):
            with self.subTest(violation=violation), tempfile.TemporaryDirectory() as tmp:
                config, by_path = self.fixture(Path(tmp))
                if violation == 'labels':
                    by_path[config['dev_manifest']][0] = dict(by_path[config['dev_manifest']][0], targets=tuple([1]*10))
                elif violation == 'missing':
                    by_path[config['train_manifest']] = by_path[config['train_manifest']][:-1]
                elif violation == 'wrongfold':
                    by_path[config['train_manifest']], by_path[config['dev_manifest']] = by_path[config['dev_manifest']], by_path[config['train_manifest']]
                else:
                    by_path[config['dev_manifest']][0] = dict(by_path[config['dev_manifest']][0], split='valid')
                with patch.object(trainer, 'read_records', side_effect=lambda p: deepcopy(by_path[str(p)])):
                    with self.assertRaises(ValueError):
                        trainer.verify_inputs(config)

    def test_changed_input_bytes_and_public_origin_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, by_path = self.fixture(Path(tmp))
            Path(config['pretrained_path']).write_bytes(b'changed')
            with patch.object(trainer, 'read_records', side_effect=lambda p: deepcopy(by_path[str(p)])):
                with self.assertRaises(ValueError):
                    trainer.verify_inputs(config)
        with tempfile.TemporaryDirectory() as tmp:
            config, by_path = self.fixture(Path(tmp))
            path = Path(config['provenance_path']); p = json.loads(path.read_text()); p['pretraining_dataset']='medical'
            path.write_text(json.dumps(p)); config['input_sha256']['provenance_path']=trainer.digest(path)
            with patch.object(trainer, 'read_records', side_effect=lambda p: deepcopy(by_path[str(p)])):
                with self.assertRaises(ValueError):
                    trainer.verify_inputs(config)

    def test_optimizer_layer_rates_cover_once_and_exclude_tokens_norm_bias_from_decay(self):
        model = nn.Module(); model.backbone = nn.Module()
        model.backbone.patch_embed = nn.Linear(3, 4)
        model.backbone.blocks = nn.ModuleList([nn.Linear(4,4) for _ in range(3)])
        model.backbone.norm = nn.LayerNorm(4)
        model.backbone.pos_embed = nn.Parameter(torch.ones(1,5,4))
        model.backbone.cls_token = nn.Parameter(torch.ones(1,1,4))
        model.backbone.reg_token = nn.Parameter(torch.ones(1,4,4))
        model.head = nn.Linear(4,10)
        groups = trainer.optimizer_groups(model, dict(backbone_lr=1e-5,head_lr=2e-4,layer_decay=.8,weight_decay=.05))
        parameters=[p for group in groups for p in group['params']]
        self.assertEqual(len(parameters), len({id(p) for p in parameters}))
        self.assertEqual({id(p) for p in parameters}, {id(p) for p in model.parameters()})
        lookup={name:group for group in groups for name in group['parameter_names']}
        self.assertAlmostEqual(lookup['backbone.patch_embed.weight']['lr'], 1e-5*.8**3)
        self.assertAlmostEqual(lookup['backbone.blocks.2.weight']['lr'], 1e-5)
        self.assertAlmostEqual(lookup['head.weight']['lr'], 2e-4)
        for name in ('backbone.pos_embed','backbone.cls_token','backbone.reg_token','backbone.norm.weight','head.bias'):
            self.assertEqual(lookup[name]['weight_decay'], 0.)
        self.assertEqual(lookup['head.weight']['weight_decay'], .05)

    def cpu_epoch(self, model, ema, dataset, optimizer, scaler, config, epoch, progress, output):
        original_to = torch.Tensor.to
        def cpu_to(tensor, *args, **kwargs):
            if args and args[0] == 'cuda':
                args = ('cpu',) + args[1:]
            return original_to(tensor, *args, **kwargs)
        with ExitStack() as stack:
            stack.enter_context(patch.object(trainer, 'make_loader', return_value=dataset.batches))
            stack.enter_context(patch.object(torch.Tensor, 'to', cpu_to))
            stack.enter_context(patch.object(torch, 'autocast', side_effect=lambda *a,**kw:nullcontext()))
            stack.enter_context(patch.object(torch.cuda, 'reset_peak_memory_stats'))
            stack.enter_context(patch.object(torch.cuda, 'max_memory_allocated', return_value=0))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return trainer.train_epoch(model, ema, dataset, optimizer, scaler, torch.ones(10),
                config, epoch, progress, output)

    def config(self):
        return dict(workers=0,seed=1,pack_images=3,effective_studies=4,epochs=3,clip_grad=100.,ema_decay=.999)

    def optimizer(self, model):
        return torch.optim.AdamW([dict(params=list(model.parameters()), lr=.002, base_lr=.002)],weight_decay=0.)

    def test_pack_tail_and_complete_study_gradients_match_direct_batch(self):
        model=TinyStudy(); reference=deepcopy(model); ema=deepcopy(model)
        optimizer=self.optimizer(model); reference_optimizer=self.optimizer(reference)
        dataset=FakeDataset(); config=self.config(); progress=dict(attempted_updates=0,optimizer_updates=0,skipped_updates=0)
        with tempfile.TemporaryDirectory() as tmp:
            metrics=self.cpu_epoch(model,ema,dataset,optimizer,CPUScaler(),config,1,progress,Path(tmp))
        for step,start in enumerate((0,4)):
            batches=dataset.batches[start:start+4]
            prediction=torch.stack(reference.forward_studies([x['images'] for x in batches]))
            target=torch.stack([x['targets'] for x in batches])
            F.binary_cross_entropy_with_logits(prediction,target).backward()
            for group in reference_optimizer.param_groups:
                group['lr']=group['base_lr']*trainer.learning_rate_factor(step,2,3)
            reference_optimizer.step(); reference_optimizer.zero_grad(set_to_none=True)
        for p,q in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(p,q,rtol=1e-5,atol=2e-7)
        self.assertEqual(metrics['studies'],5); self.assertEqual(metrics['images'],12)
        self.assertEqual(progress,dict(attempted_updates=2,optimizer_updates=2,skipped_updates=0))

    def test_amp_skipped_update_never_advances_ema(self):
        model=TinyStudy(); ema=deepcopy(model); optimizer=self.optimizer(model)
        progress=dict(attempted_updates=0,optimizer_updates=0,skipped_updates=0)
        with tempfile.TemporaryDirectory() as tmp, patch.object(trainer,'update_ema',wraps=trainer.update_ema) as spy:
            self.cpu_epoch(model,ema,FakeDataset(),optimizer,CPUScaler(skips={2}),self.config(),1,progress,Path(tmp))
            self.assertEqual(spy.call_count,1)
            self.assertEqual(spy.call_args.args[2],1)
        self.assertEqual(progress,dict(attempted_updates=2,optimizer_updates=1,skipped_updates=1))

    def test_saved_optimizer_rng_and_progress_reproduce_next_epoch(self):
        model=TinyStudy(dropout=.3); ema=deepcopy(model); optimizer=self.optimizer(model)
        dataset=FakeDataset(); progress=dict(attempted_updates=0,optimizer_updates=0,skipped_updates=0)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            self.cpu_epoch(model,ema,dataset,optimizer,CPUScaler(),self.config(),1,progress,output)
            saved=deepcopy(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
                progress=progress,rng=trainer.rng_state()))
            trainer.save_checkpoint(output/'last.pt',saved,1,'synthetic_full_state')
            self.cpu_epoch(model,ema,dataset,optimizer,CPUScaler(),self.config(),2,progress,output)
            restored=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
            second=TinyStudy(dropout=.3); second_ema=deepcopy(second); second_optimizer=self.optimizer(second)
            second.load_state_dict(restored['model']); second_ema.load_state_dict(restored['ema'])
            second_optimizer.load_state_dict(restored['optimizer']); trainer.restore_rng(restored['rng'])
            self.cpu_epoch(second,second_ema,dataset,second_optimizer,CPUScaler(),self.config(),2,restored['progress'],output)
            for left,right in ((model,second),(ema,second_ema)):
                for p,q in zip(left.parameters(),right.parameters()):
                    torch.testing.assert_close(p,q,rtol=0,atol=0)
            self.assertEqual(progress,restored['progress'])

    def test_complete_run_saves_epoch_status_without_duplicate_progress_keywords(self):
        # Run the real persistence/early-stop lifecycle while replacing CUDA
        # computation already covered by train_epoch tests above.
        class LifecycleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Sequential(nn.Linear(3,10))
            def cuda(self):
                return self
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root=Path(tmp); output=root/'output'; state=root/'state'
            config=dict(output=str(output),checkpoint_dir=str(state),seed=42,image_size=518,
                drop_rate=.2,drop_path=.1,view_chunk_size=4,pretrained_path='synthetic',image_root='synthetic',
                epochs=1,ema_decay=.999,min_delta=.0001,min_epochs=1,patience=5)
            records=[dict(subject_id='p0'),dict(subject_id='p1')]
            stack.enter_context(patch.object(trainer,'verify_inputs',return_value=(records,records,{})))
            stack.enter_context(patch.object(trainer,'start_run'))
            stack.enter_context(patch.object(trainer,'DinoStudyModel',side_effect=lambda *a,**kw:LifecycleModel()))
            stack.enter_context(patch.object(trainer,'StudyDataset',side_effect=lambda *a,**kw:FakeDataset()))
            stack.enter_context(patch.object(trainer,'training_weights',return_value=(np.array([[0]*10,[1]*10]),np.ones(10))))
            stack.enter_context(patch.object(trainer,'optimizer_groups',side_effect=lambda m,c:[dict(params=list(m.parameters()),lr=.01,base_lr=.01)]))
            progress=dict(attempted_updates=2,optimizer_updates=2,skipped_updates=0)
            stack.enter_context(patch.object(trainer,'train_epoch',return_value=dict(epoch=1,loss=.5,studies=5,images=12,**progress)))
            stack.enter_context(patch.object(trainer,'evaluate',return_value=dict(overall=dict(macro_auc=.7),patient_equal_macro_auc=.72)))
            original_tensor=torch.tensor
            def cpu_tensor(*args,**kwargs):
                if kwargs.get('device')=='cuda':
                    kwargs['device']='cpu'
                return original_tensor(*args,**kwargs)
            stack.enter_context(patch.object(torch,'tensor',side_effect=cpu_tensor))
            stack.enter_context(redirect_stdout(io.StringIO()))
            result=trainer.run(config,smoke=True)
            self.assertEqual(result['state'],'smoke_completed')
            self.assertTrue((state/'last.pt').is_file())
            self.assertTrue((output/'best_fp16.pt').is_file())
            history=[json.loads(x) for x in (output/'history.jsonl').read_text().splitlines()]
            self.assertEqual(len(history),1)
            self.assertEqual(history[0]['macro_auc'],.7)

    def test_positive_weights_count_studies_once_and_cap_rare_classes(self):
        dataset=FakeDataset(sizes=(1,1))
        dataset.groups=[[0,1],[2]]
        dataset.records=[dict(targets=[1]*10),dict(targets=[1]*10),dict(targets=[0]*10)]
        targets,weights=trainer.training_weights(dataset)
        self.assertEqual(targets.shape,(2,10)); np.testing.assert_array_equal(weights,np.ones(10))
        dataset.groups=[[i] for i in range(40)]
        dataset.records=[dict(targets=[int(i==0)]*10) for i in range(40)]
        _,weights=trainer.training_weights(dataset); np.testing.assert_array_equal(weights,np.full(10,4.))


if __name__=='__main__':
    unittest.main()
