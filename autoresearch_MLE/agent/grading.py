"""Local validation helpers for autoresearch_MLE.

The MLE-Bench harness only grades the final ``submission.csv`` at the end
of the run; during the inner loop the agent has to score itself against a
held-out fold of the *training* data. These helpers wrap the most common
metrics so the agent doesn't reinvent them in every solution.

The agent is free to score in whatever way makes sense for the
competition -- this module just covers the boring cases.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

# We deliberately avoid importing sklearn at module load: some MLE-Bench
# tasks restrict the base image and the loop should still import on a
# minimal Python interpreter. Each metric imports its own dependencies
# lazily.


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred have different lengths")
    n = len(y_true)
    if n == 0:
        return float("nan")
    s = 0.0
    for a, b in zip(y_true, y_pred):
        d = float(a) - float(b)
        s += d * d
    return math.sqrt(s / n)


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred have different lengths")
    n = len(y_true)
    if n == 0:
        return float("nan")
    return sum(abs(float(a) - float(b)) for a, b in zip(y_true, y_pred)) / n


def accuracy(y_true: Iterable, y_pred: Iterable) -> float:
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)
    if len(y_true_list) != len(y_pred_list):
        raise ValueError("y_true and y_pred have different lengths")
    if not y_true_list:
        return float("nan")
    correct = sum(1 for a, b in zip(y_true_list, y_pred_list) if a == b)
    return correct / len(y_true_list)


def log_loss(y_true: Sequence[int], y_prob: Sequence[float], eps: float = 1e-15) -> float:
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob have different lengths")
    n = len(y_true)
    if n == 0:
        return float("nan")
    s = 0.0
    for y, p in zip(y_true, y_prob):
        p = min(max(float(p), eps), 1.0 - eps)
        if int(y) == 1:
            s -= math.log(p)
        else:
            s -= math.log(1.0 - p)
    return s / n


def auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Binary ROC AUC computed from ranks. Lazy sklearn-free fallback.

    For exact correctness on large arrays, prefer
    ``sklearn.metrics.roc_auc_score``. This implementation is O(n log n)
    and tie-aware (uses average ranks).
    """
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score have different lengths")
    n = len(y_true)
    if n == 0:
        return float("nan")

    # Average-rank computation, tie-aware.
    order = sorted(range(n), key=lambda i: y_score[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # ranks are 1-indexed
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1

    pos = sum(1 for y in y_true if int(y) == 1)
    neg = n - pos
    if pos == 0 or neg == 0:
        return float("nan")
    sum_pos_ranks = sum(r for r, y in zip(ranks, y_true) if int(y) == 1)
    return (sum_pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg)


# Direction registry: which way is "better" for each metric we ship.
DIRECTIONS: dict[str, str] = {
    "rmse": "min",
    "mae": "min",
    "log_loss": "min",
    "accuracy": "max",
    "auc": "max",
}
