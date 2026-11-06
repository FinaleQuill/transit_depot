"""Shared fixed inference seed and conservative output-consistency checks.

The checks only compare already-produced outputs. They never fit a threshold or
change probabilities. Exact order and tie checks cover both classes within a
Study and each class across the compared Studies; the latter is a conservative
engineering interpretation of the supplement's unspecified sorting scope.
"""
from __future__ import annotations

import math
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

INFERENCE_SEED = 43


def configure_inference(*, torch_module: Any = None) -> dict[str, Any]:
    """Apply the same seed/precision settings before either CLI loads a model.

    Do not enable a different deterministic-algorithm policy here: changing the
    selected CUDA kernels could change the frozen FP16 deployment's outputs.
    """
    if torch_module is None:
        import torch as torch_module

    random.seed(INFERENCE_SEED)
    np.random.seed(INFERENCE_SEED)
    torch_module.manual_seed(INFERENCE_SEED)
    torch_module.cuda.manual_seed_all(INFERENCE_SEED)
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.backends.cudnn.allow_tf32 = False
    return {"seed": INFERENCE_SEED, "cudnn_benchmark": False, "allow_tf32": False,
            "model_eval_required": True, "random_inference_augmentation": False}


def _rank_and_tie_signature(values: np.ndarray) -> np.ndarray:
    """Each item's first/last occupied sorted position; O(n log n), no pairs."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    return np.stack((starts[inverse], (starts + counts - 1)[inverse]), axis=-1)


def compare_probability_outputs(
    results: Mapping, submitted: Mapping, *, atol: float = 1e-7,
    gates: Sequence[float] | None = None, expected_keys: Iterable | None = None,
    allow_reference_superset: bool = False,
) -> dict[str, Any]:
    """Validate coverage, probabilities, binary decisions, order and exact ties.

    A full submission may contain more Studies than a timing subset, but callers
    must explicitly permit it and supply the expected timing identities. By
    default both identity sets must be covered. Missing/invalid outputs fail the
    report rather than disappearing from the compared denominator.
    """
    if not isinstance(atol, (int, float)) or not math.isfinite(atol) or atol < 0:
        raise ValueError("Probability tolerance must be finite and nonnegative")
    thresholds = np.full(10, 0.5, dtype=np.float64) if gates is None else np.asarray(gates, dtype=np.float64)
    if (thresholds.shape != (10,) or not np.isfinite(thresholds).all()
            or (thresholds < 0).any() or (thresholds > 1).any()):
        raise ValueError("Fixed Gate thresholds must contain ten finite values in [0, 1]")
    if allow_reference_superset and expected_keys is None:
        raise ValueError("Reference supersets require explicit expected timing identities")

    def label(key):
        subject, study = key
        return f"{subject}/{study}"

    actual_keys = set(results)
    required = set(actual_keys if expected_keys is None else expected_keys)
    missing_results = set(required - actual_keys)
    unexpected_results = set(actual_keys - required)
    missing_reference, reused_reference, used_reference = set(), set(), set()
    invalid, probability, binary, gate, within, cross = (set() for _ in range(6))
    valid_keys, actual_rows, reference_rows = [], [], []
    max_error = 0.0
    for key in sorted(actual_keys, key=label):
        # Support the documented legacy three-column CSV with globally unique
        # Study_id, without silently matching two subjects to one reference row.
        reference_key = key if key in submitted else (None, key[1])
        if reference_key not in submitted:
            missing_reference.add(key)
            continue
        if reference_key in used_reference:
            reused_reference.add(key)
            continue
        used_reference.add(reference_key)
        try:
            actual = np.asarray(results[key], dtype=np.float64)
            reference = np.asarray(submitted[reference_key], dtype=np.float64)
            if (actual.shape != (10,) or reference.shape != (10,)
                    or not np.isfinite(actual).all() or not np.isfinite(reference).all()
                    or (actual < 0).any() or (actual > 1).any()
                    or (reference < 0).any() or (reference > 1).any()):
                raise ValueError("Invalid probability vector")
        except (TypeError, ValueError, OverflowError):
            invalid.add(key)
            continue
        error = float(np.max(np.abs(actual - reference)))
        max_error = max(max_error, error)
        if error > atol:
            probability.add(key)
        if np.any((actual >= 0.5) != (reference >= 0.5)):
            binary.add(key)
        if np.any((actual >= thresholds) != (reference >= thresholds)):
            gate.add(key)
        if not np.array_equal(_rank_and_tie_signature(actual), _rank_and_tie_signature(reference)):
            within.add(key)
        valid_keys.append(key)
        actual_rows.append(actual)
        reference_rows.append(reference)
    unused_reference = set(submitted) - used_reference
    if not allow_reference_superset:
        missing_results.update(unused_reference)
    if valid_keys:
        actual_matrix, reference_matrix = np.stack(actual_rows), np.stack(reference_rows)
        changed = np.zeros(len(valid_keys), dtype=bool)
        for column in range(10):
            changed |= np.any(
                _rank_and_tie_signature(actual_matrix[:, column])
                != _rank_and_tie_signature(reference_matrix[:, column]), axis=1)
        cross.update(key for key, differs in zip(valid_keys, changed) if differs)
    semantic = invalid | binary | gate | within | cross
    coverage = missing_results | unexpected_results | missing_reference | reused_reference
    mismatched = coverage | semantic | probability
    empty = not required or not actual_keys

    def names(keys):
        return sorted({label(key) for key in keys})

    return {
        "compared_studies": len(results), "valid_compared_studies": len(valid_keys),
        "expected_studies": len(required), "reference_studies": len(submitted),
        "mismatched_studies": names(mismatched), "mismatched_study_count": len(mismatched),
        "missing_result_studies": names(missing_results),
        "unexpected_result_studies": names(unexpected_results),
        "missing_reference_studies": names(missing_reference),
        "reused_reference_studies": names(reused_reference),
        "uncompared_reference_studies": len(unused_reference),
        "reference_superset_allowed": allow_reference_superset,
        "identity_coverage_match": not coverage and not empty,
        "invalid_probability_studies": names(invalid),
        "probability_mismatched_studies": names(probability),
        "binary_mismatched_studies": names(binary),
        "gate_mismatched_studies": names(gate),
        "sample_class_order_mismatched_studies": names(within),
        "cross_study_class_order_mismatched_studies": names(cross),
        "semantic_mismatched_study_count": len(semantic),
        "semantic_match": not semantic and not coverage and not empty,
        "absolute_tolerance": float(atol),
        "max_absolute_error": None if invalid else max_error,
        "binary_threshold": 0.5, "gate_thresholds": thresholds.tolist(),
        "ordering_policy": "exact class order/ties within each Study and exact per-class order/ties across compared Studies",
        "order_mismatch_count_definition": "Studies with changed sorted-position/tie-group signatures; not a count of pairwise inversions",
        "match": not mismatched and not empty,
    }
