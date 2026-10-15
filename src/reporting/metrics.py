"""共用指标：排序用连续分数，等级误差用等级尺度预测。"""
import numpy as np
from scipy.stats import kendalltau, spearmanr


def evaluate(y_true, grade_scores, ranking_scores=None):
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(grade_scores, dtype=float)
    rank = pred if ranking_scores is None else np.asarray(ranking_scores, dtype=float)
    if y.ndim != 1 or not len(y) or y.shape != pred.shape or y.shape != rank.shape:
        raise ValueError('Expected equally sized nonempty 1-D arrays')
    if not all(np.isfinite(x).all() for x in (y, pred, rank)):
        raise ValueError('Metrics reject NaN/Inf inputs')
    if not np.all((y >= 0) & (y <= 10) & (y == np.floor(y))):
        raise ValueError('Expected integer grades 0..10')
    correlations_defined = len(np.unique(y)) > 1 and len(np.unique(rank)) > 1
    tau = float(kendalltau(y, rank, variant='c').statistic) if correlations_defined else None
    rho = float(spearmanr(y, rank).statistic) if correlations_defined else None
    # 明确采用非负等级上的四舍五入，而非 np.round 的银行家舍入。
    grades = np.floor(np.clip(pred, 0, 10) + 0.5)
    return {
        'n': len(y),
        'kendall_tau_c': tau,
        'spearman': rho,
        'mae': float(np.mean(np.abs(pred-y))),
        'mse': float(np.mean((pred-y)**2)),
        'acc_pm1': float(np.mean(np.abs(grades-y) <= 1)),
        'correlation_note': None if correlations_defined else 'Undefined for constant truth or predictions',
    }
