"""Classification metrics — pure, hand-implemented (no sklearn).

Implementing the metrics ourselves rather than wrapping sklearn lets the
unit tests serve as oracle assertions of the textbook formulae, which is
the strongest correctness signal available short of formal verification.

All public functions accept array-like inputs (list, ndarray, Series) and
require ``y_true`` and ``y_pred`` to be the same length and non-empty
(see TestPlan.md §3 policy P2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricsReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None  # None when scores aren't supplied

    def as_dict(self) -> dict[str, float | None]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
        }


ArrayLike = Sequence[int] | np.ndarray | pd.Series


def _validate(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=int).ravel()
    yp = np.asarray(y_pred, dtype=int).ravel()
    if yt.size == 0 or yp.size == 0:
        raise ValueError("metrics undefined for empty input")
    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length; got {yt.shape} vs {yp.shape}"
        )
    return yt, yp


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[int, int, int, int]:
    """Return (TP, FP, FN, TN) for binary {0,1} labels."""
    yt, yp = _validate(y_true, y_pred)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    return tp, fp, fn, tn


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    yt, yp = _validate(y_true, y_pred)
    return float(np.mean(yt == yp))


def precision(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    tp, fp, _fn, _tn = confusion_matrix(y_true, y_pred)
    denom = tp + fp
    return float(tp / denom) if denom else 0.0


def recall(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    tp, _fp, fn, _tn = confusion_matrix(y_true, y_pred)
    denom = tp + fn
    return float(tp / denom) if denom else 0.0


def f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return float(2 * p * r / (p + r)) if (p + r) else 0.0


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """ROC-AUC via Mann–Whitney pair counting.

    AUC equals the probability that a randomly chosen positive example is
    ranked above a randomly chosen negative example. Implemented as
    (#pos>neg + 0.5 * #pos==neg) / (#pos * #neg).
    """
    yt = np.asarray(y_true, dtype=int).ravel()
    ys = np.asarray(y_score, dtype=float).ravel()
    if yt.size == 0 or ys.size == 0:
        raise ValueError("metrics undefined for empty input")
    if yt.shape != ys.shape:
        raise ValueError(
            f"y_true and y_score must have the same length; got {yt.shape} vs {ys.shape}"
        )

    pos_scores = ys[yt == 1]
    neg_scores = ys[yt == 0]
    if pos_scores.size == 0 or neg_scores.size == 0:
        raise ValueError("ROC-AUC undefined when only one class is present")

    diff = pos_scores[:, None] - neg_scores[None, :]
    wins = float(np.sum(diff > 0))
    ties = float(np.sum(diff == 0))
    total = float(pos_scores.size * neg_scores.size)
    return (wins + 0.5 * ties) / total


def evaluate(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_score: ArrayLike | None = None,
) -> MetricsReport:
    """Bundle the four point metrics (and AUC, when scores are given) into a report."""
    return MetricsReport(
        accuracy=accuracy(y_true, y_pred),
        precision=precision(y_true, y_pred),
        recall=recall(y_true, y_pred),
        f1=f1(y_true, y_pred),
        roc_auc=roc_auc(y_true, y_score) if y_score is not None else None,
    )
