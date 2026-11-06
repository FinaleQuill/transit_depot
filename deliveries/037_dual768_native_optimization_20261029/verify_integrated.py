"""Validate integrated native/LUT/one-D2H code against frozen v1; not a timer."""
import csv
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/featurize/work/cloud_project_v2')
INPUTS = Path('/tmp/dual768_2080ti_diagnosis/inputs')
MODEL = ROOT.parent / '033_dual_tiny768_seed43_25_seed44_75/fp16_hflip2'

def read_vectors(path, image):
    import numpy as np
    rows = {}
    for row in csv.DictReader(path.open()):
        key = (row['Subject_id'], row['Image_id'] if image else row['Study_id'])
        values = rows.setdefault(key, np.zeros(10, dtype=np.float64))
        values[int(row['Predict_class']) - 1] = float(row['Probability'])
    return rows

def main():
    output = ROOT / 'integrated_avx2_v1'
    output.mkdir(exist_ok=False)
    report = {'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'diagnostic_only': True,
              'training': False, 'T4_certified': False, 'input_checks': [], 'probability_checks': []}
    started = time.perf_counter()
    code = 1
    try:
        sys.path.insert(0, str(REPO))
        import numpy as np
        import torch
        import deploy_kd as deploy
        import fast_lanczos
        from inference_inputs import load_image_pixels, normalize_pixels
        from inference_reproducibility import configure_inference, compare_probability_outputs
        from benchmark_local_resources import validate_inputs
        configure_inference(torch_module=torch)
        props = torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / props.total_memory)
        assert fast_lanczos.available()
        report['native_backend'] = fast_lanczos.implementation()
        spec = importlib.util.spec_from_file_location('frozen_v1_deploy', ROOT/'source_baseline_v1/deploy_kd.py')
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        measured, warmup, coverage = validate_inputs(INPUTS/'pictures_100', INPUTS/'pictures_5', INPUTS/'selection.portable.json')
        report['coverage'] = coverage
        predictor = deploy.Predictor(MODEL)
        for record in [*warmup, *measured]:
            reference = baseline.load_image_arrays(record['path'], [768])[768]
            current = deploy.load_image_arrays(record['path'], [768])[768]
            assert reference.tobytes() == current.tobytes(), record['image_id']
            pixels = load_image_pixels(record['path'], [768], deploy._letterbox_image)[768]
            actual = predictor.input_lookup.to_tensor(pixels)
            expected = torch.from_numpy(reference).unsqueeze(0).to('cuda', dtype=torch.float16, memory_format=torch.channels_last)
            assert actual.stride() == expected.stride()
            assert actual.cpu().numpy().tobytes() == expected.cpu().numpy().tobytes(), record['image_id']
            report['input_checks'].append({'image_id': record['image_id'], 'float32_and_fp16_byte_equal': True})
        pixels = np.arange(256, dtype=np.uint8).reshape(16,16)
        for fixture in [pixels, np.stack((pixels, pixels[::-1], 255-pixels), axis=-1)]:
            expected = torch.from_numpy(normalize_pixels(fixture, deploy.MEAN, deploy.STD)).unsqueeze(0).to('cuda', dtype=torch.float16, memory_format=torch.channels_last)
            actual = predictor.input_lookup.to_tensor(fixture)
            assert actual.cpu().numpy().tobytes() == expected.cpu().numpy().tobytes()
        reference_images = read_vectors(ROOT/'local_baseline_v1/round_1.image_predictions.csv', True)
        reference_studies = read_vectors(ROOT/'local_baseline_v1/round_1.csv', False)
        for record in warmup:
            predictor.predict_image(record['path'])
        actual_images, actual_studies = {}, {}
        accumulator = deploy.StudyAccumulator(measured)
        with (output/'image_predictions.csv').open('x', newline='') as image_file, (output/'study_predictions.csv').open('x', newline='') as study_file:
            image_writer, study_writer = csv.writer(image_file), csv.writer(study_file)
            image_writer.writerow(deploy.IMAGE_CSV_HEADER)
            study_writer.writerow(deploy.CSV_HEADER)
            for record in measured:
                actual = predictor.predict_image(record['path'])
                key = (record['subject_id'], record['image_id'])
                expected = reference_images[key]
                assert actual.tobytes() == expected.tobytes(), record['image_id']
                actual_images[key] = actual
                deploy.write_image_prediction(image_writer, record, actual)
                result = accumulator.add(record, actual)
                if result is not None:
                    subject, study, mean = result
                    actual_studies[(subject, study)] = mean
                    deploy.write_prediction(study_writer, subject, study, mean)
                report['probability_checks'].append({'image_id': record['image_id'], 'byte_equal': True})
        assert accumulator.complete()
        report['image_comparison'] = compare_probability_outputs(actual_images, reference_images)
        report['study_comparison'] = compare_probability_outputs(actual_studies, reference_studies)
        assert report['image_comparison']['match'] and report['study_comparison']['match']
        report['status'] = 'passed'
        code = 0
    except BaseException:
        report.update(status='failed', exception=traceback.format_exc())
        traceback.print_exc()
    finally:
        report.update(ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(), wall_seconds=time.perf_counter()-started, exit_code=code)
        report['source_sha256'] = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*REPO.glob('*.py'), *REPO.joinpath('native_lanczos').glob('*.so'), REPO/'native_lanczos/resample.c']}
        (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ['input_checks', 'probability_checks', 'source_sha256']}, indent=2))
    return code

if __name__ == '__main__':
    raise SystemExit(main())
