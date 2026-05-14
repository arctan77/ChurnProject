"""Deterministic k-fold cross-validation.

Splits are contiguous (no shuffling), so the same input always yields the
same folds — required by test U2-8 (determinism). Phase 2 may introduce a
seeded shuffle option; until then determinism is enforced by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from churn.evaluation.metrics import MetricsReport, evaluate
from churn.modeling.protocol import Model


ModelFactory = Callable[[], Model]


@dataclass(frozen=True)
class CVReport:
    folds: list[MetricsReport]

    def __len__(self) -> int:
        return len(self.folds)

    def mean(self, metric: str) -> float:
        values = [getattr(f, metric) for f in self.folds]
        return float(np.mean(values))


def _contiguous_folds(n: int, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return k (train_idx, test_idx) pairs covering [0, n)."""
    if k < 2:
        raise ValueError(f"k must be >= 2; got {k}")
    if n < k:
        raise ValueError(f"need at least k={k} rows for k-fold CV; got {n}")
    indices = np.arange(n)
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1
    out: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for size in fold_sizes:
        stop = start + size
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        out.append((train_idx, test_idx))
        start = stop
    return out


def cross_validate(
    factory: ModelFactory,
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 5,
) -> CVReport:
    """Train-and-evaluate the model from ``factory`` on ``k`` folds of (X, y)."""
    if len(X) != len(y):
        raise ValueError(f"X and y must have the same length; got {len(X)} vs {len(y)}")

    reports: list[MetricsReport] = []
    for train_idx, test_idx in _contiguous_folds(len(X), k):
        model = factory()
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]
        # ROC-AUC is undefined when a fold's test set has only one class.
        # That's a fact of CV, not an error — leave the fold's AUC as None.
        score_arg = y_score if y_test.nunique() > 1 else None
        reports.append(evaluate(y_test, y_pred, y_score=score_arg))

    return CVReport(folds=reports)
