#!/usr/bin/env python3
"""Read-only CPU audit of the three uninstrumented 033 timing rounds.

This does not alter production tolerances, probabilities, or benchmark results.
Exact pairwise order/tie relations are checked without a tolerance. Fixed binary
predictions use probability >= 0.5. Outputs are created exclusively, not replaced.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

CLASSES = tuple(range(1, 11))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_predictions(path, image):
    identities = ['Image_id', 'Subject_id', 'Study_id'] if image else ['Subject_id', 'Study_id']
    expected = identities + ['Predict_class', 'Probability']
    result = {}
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected:
            raise ValueError(f'{path}: unexpected columns {reader.fieldnames!r}')
        for row in reader:
            sample = tuple(row[name] for name in identities)
            if not all(sample):
                raise ValueError(f'{path}: blank identity')
            cls = int(row['Predict_class'])
            probability = float(row['Probability'])
            if cls not in CLASSES or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f'{path}: invalid class or probability')
            key = sample + (cls,)
            if key in result:
                raise ValueError(f'{path}: duplicate key {key!r}')
            result[key] = probability
    groups = defaultdict(set)
    for key in result:
        groups[key[:-1]].add(key[-1])
    for sample, classes in groups.items():
        if classes != set(CLASSES):
            raise ValueError(f'{path}: incomplete classes for {sample!r}')
    return result, identities


def relation(a, b):
    return (a > b) - (a < b)


def compare_predictions(current_path, reference_path, image, tag, difference_rows, pair_rows):
    current, identities = read_predictions(current_path, image)
    reference, _ = read_predictions(reference_path, image)
    common = sorted(current.keys() & reference.keys())
    missing = sorted(reference.keys() - current.keys())
    extra = sorted(current.keys() - reference.keys())
    differences = [abs(current[k] - reference[k]) for k in common]
    binary_changes = 0
    for key, delta in zip(common, differences):
        changed_binary = (current[key] >= 0.5) != (reference[key] >= 0.5)
        binary_changes += changed_binary
        if delta != 0 or changed_binary:
            difference_rows.append([tag, 'image' if image else 'study', *(['|'.join(key[:-1])]), key[-1], reference[key], current[key], delta, int(delta > 1e-3), int(changed_binary)])
    current_samples = {k[:-1] for k in current}
    reference_samples = {k[:-1] for k in reference}
    common_samples = sorted(current_samples & reference_samples)
    within_changed_samples = []
    within_changed_pairs = 0
    within_tie_changes = 0
    for sample in common_samples:
        changed = False
        for ca, cb in itertools.combinations(CLASSES, 2):
            ref_rel = relation(reference[sample + (ca,)], reference[sample + (cb,)])
            cur_rel = relation(current[sample + (ca,)], current[sample + (cb,)])
            if ref_rel != cur_rel:
                changed = True
                within_changed_pairs += 1
                within_tie_changes += (ref_rel == 0 or cur_rel == 0)
                pair_rows.append([tag, 'within_image' if image else 'within_study', '|'.join(sample), ca, cb, ref_rel, cur_rel])
        if changed:
            within_changed_samples.append(list(sample))
    cross_study = None
    if not image:
        cross_study = {}
        for cls in CLASSES:
            changed_pairs = tie_changes = inversions = 0
            affected = set()
            for sample_a, sample_b in itertools.combinations(common_samples, 2):
                ref_rel = relation(reference[sample_a + (cls,)], reference[sample_b + (cls,)])
                cur_rel = relation(current[sample_a + (cls,)], current[sample_b + (cls,)])
                if ref_rel != cur_rel:
                    changed_pairs += 1
                    tie_changes += (ref_rel == 0 or cur_rel == 0)
                    inversions += (ref_rel * cur_rel == -1)
                    affected.update([sample_a, sample_b])
                    pair_rows.append([tag, 'across_study', str(cls), '|'.join(sample_a), '|'.join(sample_b), ref_rel, cur_rel])
            cross_study[str(cls)] = {'compared_pairs': math.comb(len(common_samples), 2), 'changed_pairs': changed_pairs,
                                     'strict_inversions': inversions, 'tie_relation_changes': tie_changes,
                                     'affected_studies': [list(x) for x in sorted(affected)]}
    return {
        'current_file': str(current_path), 'reference_file': str(reference_path),
        'current_sha256': sha256(current_path), 'reference_sha256': sha256(reference_path),
        'identity_columns': identities, 'current_rows': len(current), 'reference_rows': len(reference),
        'compared_rows': len(common), 'current_samples': len(current_samples), 'reference_samples': len(reference_samples),
        'missing_keys': [list(k) for k in missing], 'extra_keys': [list(k) for k in extra],
        'coverage_exact': not missing and not extra,
        'max_absolute_probability_difference': max(differences, default=None),
        'mean_absolute_probability_difference': sum(differences) / len(differences) if differences else None,
        'probability_differences_gt_1e_minus_3': sum(d > 1e-3 for d in differences),
        'nonzero_probability_differences': sum(d != 0 for d in differences),
        'binary_threshold': 0.5, 'binary_comparator': '>=', 'binary_changes': binary_changes,
        'sample_class_pair_comparisons': len(common_samples) * math.comb(10, 2),
        'sample_class_order_or_tie_changed_pairs': within_changed_pairs,
        'sample_class_tie_relation_changes': within_tie_changes,
        'sample_class_order_or_tie_changed_samples': within_changed_samples,
        'per_class_across_study_order': cross_study,
    }


def percentile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p / 100
    lower = math.floor(index)
    upper = math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def timing_summary(path, all_rows):
    report = json.loads(path.read_text(encoding='utf-8'))
    matrix = report['per_image_round_ms']
    order = report['image_order']
    if len(matrix) != 100 or len(order) != 100 or len(set(order)) != 100:
        raise ValueError(f'{path}: expected 100 unique images and timing rows')
    if any(len(row) != 3 for row in matrix):
        raise ValueError(f'{path}: expected three timing observations per image')
    flat = [v for row in matrix for v in row]
    if any(not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in flat):
        raise ValueError(f'{path}: invalid timing observation')
    means = [sum(row) / 3 for row in matrix]
    round_means = [sum(row[r] for row in matrix) / 100 for r in range(3)]
    report_delta = {
        'mean_ms': abs(sum(means) / 100 - report['mean_ms']),
        'p95_ms': abs(percentile(means, 95) - report['p95_ms']),
        'max_ms': abs(max(means) - report['max_ms']),
        'round_means_max_abs': max(abs(a-b) for a,b in zip(round_means, report['round_mean_ms'])),
        'per_image_means_max_abs': max(abs(a-b) for a,b in zip(means, report['per_image_mean_ms'])),
    }
    for image, row, avg in zip(order, matrix, means):
        for rnd, ms in enumerate(row, 1):
            all_rows.append([image, rnd, ms, avg, int(ms > 100)])
    return {
        'report': str(path), 'report_sha256': sha256(path), 'observations': len(flat),
        'images': len(matrix), 'rounds': 3, 'round_mean_ms': round_means,
        'overall_mean_ms': sum(flat) / len(flat), 'per_image_three_round_mean_p95_ms': percentile(means, 95),
        'per_image_three_round_mean_max_ms': max(means), 'all_300_observations_p95_ms': percentile(flat, 95),
        'all_300_observations_max_ms': max(flat), 'observations_gt_100ms': sum(v > 100 for v in flat),
        'per_image_means_gt_100ms': sum(v > 100 for v in means),
        'recomputed_vs_saved_absolute_differences': report_delta,
        'all_observations_retained': True,
        'slowest_observations': [dict(image_id=order[i], round=r+1, milliseconds=matrix[i][r])
                                 for i,r in sorted(itertools.product(range(100),range(3)),key=lambda x: matrix[x[0]][x[1]],reverse=True)[:10]],
        'protocol': report['protocol'], 'coverage': report['coverage'],
        'deployment': report['deployment'], 'requested_resources': report['requested_resources'],
        'environment': report['environment'], 'resources': report['resources'],
        'benchmark_comparison': report['comparison'],
        'T4_latency_certified': report['T4_latency_certified'],
        'official_compliance_pass': report['official_compliance_pass'],
    }


def write_csv(path, header, rows):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    current = args.current_dir.resolve(strict=True)
    reference = args.reference_dir.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    differences, ranking_changes, timings = [], [], []
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'cpu_only': True,
              'scope': 'valid proxy timing; no official T4 or official test consistency certification',
              'binary_rule': 'probability >= 0.5', 'ordering_rule': 'exact pairwise -1/0/+1, with no tolerance',
              'probability_threshold_only_reported': 1e-3, 'cross_hardware_rounds': [], 'same_machine_rounds': []}
    for rnd in range(1, 4):
        for image in (False, True):
            suffix = '.image_predictions.csv' if image else '.csv'
            name = f'round_{rnd}{suffix}'
            report['cross_hardware_rounds'].append(compare_predictions(current/name,reference/name,image,f'current_round_{rnd}_vs_4090_round_{rnd}',differences,ranking_changes))
            if rnd > 1:
                report['same_machine_rounds'].append(compare_predictions(current/name,current/f'round_1{suffix}',image,f'current_round_{rnd}_vs_current_round_1',differences,ranking_changes))
    report['timing'] = timing_summary(current/'local_resource_report.json',timings)
    reference_timings = []
    report['reference_timing'] = timing_summary(reference/'local_resource_report.json',reference_timings)
    report['latency_ratio_current_over_4090'] = report['timing']['overall_mean_ms'] / report['reference_timing']['overall_mean_ms']
    comparisons = report['cross_hardware_rounds'] + report['same_machine_rounds']
    report['all_coverage_exact'] = all(c['coverage_exact'] for c in comparisons)
    report['all_fixed_0_5_binary_equal'] = all(c['binary_changes'] == 0 for c in comparisons)
    report['all_sample_class_order_and_ties_equal'] = all(c['sample_class_order_or_tie_changed_pairs'] == 0 for c in comparisons)
    report['all_cross_study_class_order_and_ties_equal'] = all(all(v['changed_pairs'] == 0 for v in c['per_class_across_study_order'].values()) for c in comparisons if c['per_class_across_study_order'] is not None)
    write_csv(output/'all_probability_differences.csv',['comparison','level','sample_identity','class','reference_probability','current_probability','absolute_difference','gt_1e_minus_3','binary_changed'], differences)
    write_csv(output/'all_order_or_tie_changes.csv',['comparison','scope','sample_or_class','item_a','item_b','reference_relation','current_relation'],ranking_changes)
    write_csv(output/'all_300_timing_observations.csv',['Image_id','round','milliseconds','image_three_round_mean_ms','gt_100ms'],timings)
    with (output/'comparison_report.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output_dir':str(output),'all_coverage_exact':report['all_coverage_exact'],
                      'all_fixed_0_5_binary_equal':report['all_fixed_0_5_binary_equal'],
                      'all_sample_class_order_and_ties_equal':report['all_sample_class_order_and_ties_equal'],
                      'all_cross_study_class_order_and_ties_equal':report['all_cross_study_class_order_and_ties_equal'],
                      'timing':{k:v for k,v in report['timing'].items() if k in ['round_mean_ms','overall_mean_ms','per_image_three_round_mean_p95_ms','per_image_three_round_mean_max_ms','all_300_observations_p95_ms','all_300_observations_max_ms']}} ,indent=2))
    return 0 if report['all_coverage_exact'] else 2

if __name__ == '__main__':
    raise SystemExit(main())
