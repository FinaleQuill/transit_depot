"""Small explicit GPU exception-path gate; no real models or weights are loaded."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'executed':False,'task':'packed_d2h_and_staging_exception_paths'}))
        return 0
    sys.path.insert(0,str(args.source_root.resolve()))
    import numpy as np
    import torch
    import deploy_kd as deploy
    from lut_prototype import CandidatePredictor
    assert torch.__version__.split('+')[0]=='2.10.0' and torch.version.cuda=='12.8'
    assert torch.cuda.is_available()
    torch.set_num_threads(1)

    class FakeModel:
        def __init__(self,outputs):
            self.outputs=outputs; self.calls=0
        def __call__(self,tensor):
            value=self.outputs[self.calls]; self.calls+=1
            return value

    def fixture(outputs):
        base=SimpleNamespace(torch=torch,device=torch.device('cuda'),
            members=[{'image_size':768,'tta':'hflip2'} for _ in range(2)],
            models=[FakeModel(outputs[:2]),FakeModel(outputs[2:])],
            model_weights=[[.25]*10,[.75]*10],config={},inventory={},sync_calls=0)
        def sync():
            base.sync_calls+=1; torch.cuda.synchronize()
        base.synchronize=sync
        state={'pixels':np.zeros((768,768),dtype=np.uint8)}
        candidate=CandidatePredictor(base,deploy,pixel_loader=lambda path,sizes:{768:state['pixels']})
        return base,candidate,state

    def valid():
        return torch.full((1,10),.37,dtype=torch.float16,device='cuda')

    checks=[]
    for shape in ((10,),(1,9),(2,10)):
        outputs=[torch.zeros(shape,dtype=torch.float16,device='cuda'),valid(),valid(),valid()]
        base,candidate,state=fixture(outputs)
        try:
            candidate.predict_image('synthetic_shape_error')
        except FloatingPointError as error:
            assert str(error)=='Invalid single-image student logits'
        else:
            raise AssertionError('Wrong logit shape was accepted')
        assert base.sync_calls==1
        assert [model.calls for model in base.models]==[1,0]
        base.models[:]=[FakeModel([valid(),valid()]),FakeModel([valid(),valid()])]
        state['pixels'].fill(255)
        actual=candidate.predict_image('synthetic_retry_after_shape_error')
        assert np.isfinite(actual).all() and base.sync_calls==1
        assert [model.calls for model in base.models]==[2,2]
        checks.append({'shape':list(shape),'error_fence_calls':1,'retry_ok':True})
    for bad in (float('inf'),float('-inf'),float('nan')):
        for index in range(4):
            outputs=[valid() for _ in range(4)]
            outputs[index][0,2]=bad
            base,candidate,state=fixture(outputs)
            try:
                candidate.predict_image('synthetic_nonfinite_logits')
            except FloatingPointError as error:
                assert str(error)=='Invalid single-image student logits'
            else:
                raise AssertionError('Nonfinite raw logits escaped the packed flag')
            assert [model.calls for model in base.models]==[2,2]
            assert base.sync_calls==1  # Exception cleanup only, after completed D2H.
            checks.append({'raw_logit':str(bad),'view_index':index,'rejected':True})
    report={'executed':True,'gpu_diagnostic_only':True,'real_model_loads':0,
            'training_runs':0,'status':'passed','checks':checks}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
