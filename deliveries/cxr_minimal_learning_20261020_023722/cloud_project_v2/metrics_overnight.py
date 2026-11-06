"""最终五折集成需要的指标小工具（精简版）。

原先由旧训练大文件提供的五个函数，完整抽取到这里。
只计算指标和阈值，不创建模型、不训练；调用方传入Study级标签和概率。
"""
from __future__ import annotations

from typing import Any
import numpy as np
from dataset import NUM_CLASSES

# 延迟导入sklearn，先检查[N_Study,10]标签/概率，再逐类计算。
def _metrics_imports():
    # 延迟导入，单纯阅读/import模型代码不要求安装评估依赖。
    try:
        from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
    except ImportError as exc:
        raise ImportError("未来运行需准备scikit-learn；本文件不会自动安装") from exc
    return roc_auc_score, f1_score, precision_recall_curve, average_precision_score


def _check_hard_predictions(targets: np.ndarray, probabilities: np.ndarray) -> None:
    if targets.ndim != 2 or targets.shape != probabilities.shape or targets.shape[1] != NUM_CLASSES:
        raise ValueError("评估必须使用形状一致的[N,10]硬标签和概率")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("AUC/F1不可直接用Mixup/CutMix的软标签计算")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("评估输入必须是有限的[0,1]概率")
    # 不悄悄nanmean：单类没有正例/负例时宏AUC的口径已经发生变化，直接报错。
    if targets.shape[0] == 0 or any(np.unique(targets[:, c]).size != 2 for c in range(NUM_CLASSES)):
        raise ValueError("每类都必须有正负Study，不能伪造AUC或默填阈值0.5")


# 每类挑F1最大的概率断点；并列时选更高阈值。
def search_class_thresholds(targets: np.ndarray, probabilities: np.ndarray) -> list[float]:
    """遍历每类所有唯一概率断点，精确最大化该类验证F1，不用粗网格。

    precision_recall_curve第k点对应proba>=thresholds[k]；最后一点没有阈值。
    排除最后一点后计算F1；若并列，取更高阈值，规则固定、可复现。
    十类独立最大化F1等价于在此数据上最大化十类的Macro-F1。
    """
    _check_hard_predictions(targets, probabilities)
    _, _, precision_recall_curve, _ = _metrics_imports()
    selected: list[float] = []
    for c in range(NUM_CLASSES):
        # 把该类的所有不同预测概率作为决策断点；无需手写0.1、0.2等粗网格。
        # 最后一个precision/recall点没有对应阈值，因此后面用[:-1]对齐。
        precision, recall, cutoffs = precision_recall_curve(targets[:, c], probabilities[:, c])
        denominator = precision[:-1] + recall[:-1]
        scores = np.divide(2 * precision[:-1] * recall[:-1], denominator,
                           out=np.zeros_like(denominator), where=denominator > 0)
        # cutoffs按升序排列；若多个断点F1并列，[-1]固定选择最高阈值，保证复现。
        best = np.flatnonzero(scores == scores.max())[-1]
        selected.append(float(cutoffs[best]))
    return selected


# AUC/AP看连续概率；F1/precision/recall先用十类阈值转成0/1。
def calculate_metrics(targets: np.ndarray, probabilities: np.ndarray,
                      thresholds: list[float]) -> dict[str, Any]:
    _check_hard_predictions(targets, probabilities)
    gates = np.asarray(thresholds, dtype=np.float64)
    if gates.shape != (NUM_CLASSES,) or not np.isfinite(gates).all() or (gates < 0).any() or (gates > 1).any():
        raise ValueError("阈值必须为十维有限[0,1]数组")
    roc_auc_score, f1_score, _, average_precision_score = _metrics_imports()
    aucs = [float(roc_auc_score(targets[:, c], probabilities[:, c])) for c in range(NUM_CLASSES)]
    f1s = [float(f1_score(targets[:, c], probabilities[:, c] >= gates[c], zero_division=0)) for c in range(NUM_CLASSES)]
    aps = [float(average_precision_score(targets[:, c], probabilities[:, c])) for c in range(NUM_CLASSES)]
    predicted, positive = probabilities >= gates, targets == 1
    tp = (predicted & positive).sum(axis=0)
    fp = (predicted & ~positive).sum(axis=0)
    fn = (~predicted & positive).sum(axis=0)
    precision = np.divide(tp, tp + fp, out=np.zeros(NUM_CLASSES), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros(NUM_CLASSES), where=(tp + fn) > 0)
    # 下一轮可据此区分稀有类“漏报多”还是“误报多”，不再只根据低F1猜原因。
    return {"macro_auc": float(np.mean(aucs)), "macro_f1": float(np.mean(f1s)),
            "macro_ap": float(np.mean(aps)), "per_class_ap": aps,
            "per_class_auc": aucs, "per_class_f1": f1s,
            "per_class_precision": precision.tolist(), "per_class_recall": recall.tolist(),
            "per_class_positive_studies": positive.sum(axis=0).tolist(),
            "per_class_tp": tp.tolist(), "per_class_fp": fp.tolist(), "per_class_fn": fn.tolist()}
