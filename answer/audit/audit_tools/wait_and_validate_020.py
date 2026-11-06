"""Wait for completed inference artifacts; independently audit CSV/protocol on CPU."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import time

from independent_submission_check import check

ROOT = Path('/home/featurize/work/deliveries/020_768_ensemble_20261020_124902')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    final = ROOT / 'inference/submission_final.csv'
    alias = ROOT / 'inference/submission.csv'
    driver_path = ROOT / 'driver_result.json'
    while not (final.is_file() and alias.is_file() and driver_path.is_file()):
        if (ROOT / 'driver_error.json').is_file():
            raise RuntimeError((ROOT / 'driver_error.json').read_text())
        # 每次最多等 15 秒；只检查文件存在性，不使用 GPU 或模型。
        time.sleep(15)
    result = check(final,
        Path('/home/featurize/work/cloud_project_v2/data/clean_v1_reviewed/test_manifest.csv'),
        Path('/home/featurize/work/submission.csv'),
        ROOT / 'inference/study_probabilities.csv')
    status = load(ROOT / 'inference/status.json')
    driver = load(driver_path)
    plan_path = ROOT / 'ensemble_plan_768_equal.json'
    plan = load(plan_path)
    provenance = load(ROOT / 'inference/provenance.json')
    protocol = provenance['protocol']
    entries = plan['checkpoints']
    models = provenance['models']
    expected_folds = list(range(5))
    checks = {
        'inference_status_complete': status.get('status') == 'complete',
        'driver_status_complete': driver.get('status') == 'complete',
        'final_path_matches_status': status.get('submission') == str(final),
        'final_path_matches_driver': driver.get('submission_final') == str(final),
        'upload_alias_path_matches_driver': driver.get('upload_filename_alias') == str(alias),
        'upload_alias_equal_bytes': alias.read_bytes() == final.read_bytes(),
        'driver_sha256_matches_csv': driver.get('sha256') == result['sha256'],
        'frozen_plan_sha256_matches_status': status.get('ensemble_plan_sha256') == digest(plan_path),
        'frozen_plan_sha256_matches_provenance': provenance.get('ensemble_plan_sha256') == digest(plan_path),
        'five_distinct_folds_0_to_4': sorted(e['fold'] for e in entries) == expected_folds,
        'driver_folds_match_plan': sorted(driver.get('folds', [])) == expected_folds,
        'plan_five_equal_weights': len(entries) == 5 and all(e['weight'] == 1 for e in entries),
        'plan_768_hflip': plan.get('tta') == {'scales': [768], 'hflip': True},
        'actual_protocol_768_hflip': protocol.get('tta') == {'scales': [768], 'hflip': True},
        'actual_protocol_aggregation': protocol.get('aggregation') == 'sigmoid -> original Study view mean -> TTA mean -> fold weighted mean',
        'actual_five_equal_weights': len(protocol['checkpoints']) == 5 and all(abs(e['normalized_weight'] - .2) <= 1e-15 for e in protocol['checkpoints']),
        'provenance_five_models': provenance.get('fold_count') == 5 and len(models) == 5,
        'actual_models_match_frozen_plan': {(e['fold'], e['sha256']) for e in models} == {(e['fold'], e['sha256']) for e in entries},
        'actual_weights_are_ema': all(e.get('weights_source') == 'ema' for e in models),
        'test_not_used_for_fitting': provenance.get('test_used_for_fitting') is False,
        'platform_upload_not_performed': provenance.get('platform_upload_performed') is False,
    }
    result.update({'checked_hong_kong': datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
        'upload_filename_alias': str(alias), 'alias_sha256': digest(alias),
        'frozen_plan_sha256': digest(plan_path), 'folds': expected_folds,
        'tta': plan['tta'], 'fold_normalized_weights': [.2] * 5,
        'protocol_and_artifact_checks': checks})
    result['errors'].extend(name for name, ok in checks.items() if not ok)
    result['passed'] = not result['errors']
    output = ROOT / 'independent_submission_validation.json'
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
