"""Explicit GPU validation for parent scheduling; no execution without --execute.

This is an isolated parity diagnostic, not a timing benchmark. It intentionally
compares the original CUDA FP32->FP16 input conversion and uses blocking output
copies for bit comparisons. Such checks stay outside production timing.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--inputs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"executed": False, "task": "LUT_input_and_packed_D2H_parity",
                          "gpu_work_requires_execute": True, "output_dir": str(args.output_dir)}))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"executed": True, "diagnostic_only": True, "T4_certified": False,
              "training": False, "production_source_changed": False,
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "argv": sys.argv, "input_checks": [], "prediction_checks": []}
    started = time.perf_counter(); code = 1
    try:
        assert sys.version_info[:2] == (3,12), sys.version
        sys.path.insert(0, str(args.source_root.resolve()))
        import numpy as np
        import torch
        import deploy_kd as deploy
        from lut_prototype import (CandidatePredictor, ExactU8Input,
                                   load_image_pixels, normalize_pixels)

        assert torch.__version__.split("+")[0] == "2.10.0"
        assert torch.version.cuda == "12.8" and torch.cuda.is_available()
        random.seed(43); np.random.seed(43); torch.manual_seed(43); torch.cuda.manual_seed_all(43)
        props = torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / props.total_memory)
        report["gpu"] = {"name": props.name, "sm": props.multi_processor_count,
                         "capability": [props.major,props.minor], "allocator_gib": 8}
        report["baseline_source_sha256"] = hashlib.sha256((args.source_root/'deploy_kd.py').read_bytes()).hexdigest()
        lookup = ExactU8Input(torch, torch.device("cuda"), deploy.MEAN, deploy.STD, image_sizes=[768])

        def check_input(pixels, name):
            reference = torch.from_numpy(normalize_pixels(pixels, deploy.MEAN, deploy.STD)).unsqueeze(0).to(
                "cuda", dtype=torch.float16, memory_format=torch.channels_last)
            actual = lookup.to_tensor(pixels)
            assert actual.dtype == torch.float16
            assert tuple(actual.shape) == tuple(reference.shape)
            assert actual.is_contiguous(memory_format=torch.channels_last)
            assert actual.cpu().numpy().tobytes() == reference.cpu().numpy().tobytes(), name
            report["input_checks"].append({"name": name, "shape": list(actual.shape),
                                          "fp16_bit_parity": True})

        rng = np.random.default_rng(20261029)
        gray = np.arange(256,dtype=np.uint8).reshape(16,16)
        check_input(gray,"all_256_gray_values")
        check_input(np.stack((gray, gray[::-1],255-gray),axis=-1),"all_3x256_RGB_fallback_values")
        for shape in ((37,51),(513,257),(768,768)):
            gray = rng.integers(0,256,shape,dtype=np.uint8)
            check_input(gray,"random_gray_"+str(shape))
        measured = deploy.discover_images(args.inputs_root/'pictures_100')
        warmup = deploy.discover_images(args.inputs_root/'pictures_5')
        assert len(measured)==100 and len(warmup)==5
        for record in [*warmup,*measured]:
            path=record['path']
            pixels=load_image_pixels(path,[768],deploy._letterbox_image)[768]
            actual=lookup.to_tensor(pixels)
            original=deploy.load_image_arrays(path,[768])[768]
            expected=torch.from_numpy(original).unsqueeze(0).to(
                'cuda',dtype=torch.float16,memory_format=torch.channels_last)
            assert actual.cpu().numpy().tobytes()==expected.cpu().numpy().tobytes(), path
            report['input_checks'].append({'name':record['image_id'],'fp16_bit_parity':True})

        base=deploy.Predictor(args.model_dir)
        candidates = {'lut_only': CandidatePredictor(base,deploy,use_lut=True,merge_d2h=False),
                      'merged_d2h_only': CandidatePredictor(base,deploy,use_lut=False,merge_d2h=True),
                      'lut_and_merged_d2h': CandidatePredictor(base,deploy,use_lut=True,merge_d2h=True)}
        for record in warmup:
            base.predict_image(record['path'])
            for candidate in candidates.values(): candidate.predict_image(record['path'])
        for name in ['baseline',*candidates]:
            with (args.output_dir/(name+'.image_predictions.csv')).open('x',newline='') as stream:
                csv.writer(stream).writerow(deploy.IMAGE_CSV_HEADER)
        for record in measured:
            expected = base.predict_image(record['path'])
            values={'baseline':expected}
            for name,candidate in candidates.items():
                actual=candidate.predict_image(record['path'])
                assert actual.dtype==expected.dtype and actual.tobytes()==expected.tobytes(),(record['image_id'],name)
                values[name]=actual
            for name,value in values.items():
                with (args.output_dir/(name+'.image_predictions.csv')).open('a',newline='') as stream:
                    deploy.write_image_prediction(csv.writer(stream),record,value)
            report['prediction_checks'].append({'image_id':record['image_id'],'all_3_candidate_modes_byte_parity':True})
        report['all_input_checks_passed']=True
        report['all_prediction_checks_passed']=True
        report['status']='passed'; code=0
    except BaseException:
        report.update(status='failed',exception=traceback.format_exc())
        traceback.print_exc()
    finally:
        report.update(ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                      wall_seconds=time.perf_counter()-started,exit_code=code)
        with (args.output_dir/'verification.json').open('x') as stream:
            json.dump(report,stream,indent=2); stream.write('\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ['input_checks','prediction_checks']},indent=2))
    return code


if __name__=='__main__':
    raise SystemExit(main())
