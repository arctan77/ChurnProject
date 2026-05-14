"""Unit tests for ``churn.evaluation``.

Each test maps 1:1 to a numbered case in TestPlan.md §5. Test docstrings
restate the methodology so the unit-test report can be assembled directly
from this file's output.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from churn.evaluation import (
    accuracy,
    confusion_matrix,
    cross_validate,
    evaluate,
    f1,
    precision,
    recall,
    roc_auc,
)
from churn.modeling import FakeModel


# ---------------------------------------------------------------------------
# U2-1 — Oracle: perfect classifier
# ---------------------------------------------------------------------------
def test_U2_1_perfect_classifier_yields_all_ones():
    """Oracle: predictions exactly match labels → every metric = 1.0."""
    y_true = [1, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 1]

    report = evaluate(y_true, y_pred)

    assert report.accuracy == 1.0
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1 == 1.0


# ---------------------------------------------------------------------------
# U2-2 — Oracle: all-wrong classifier
# ---------------------------------------------------------------------------
def test_U2_2_all_wrong_classifier_yields_all_zeros():
    """Oracle: every prediction inverted → every metric = 0.0 (no NaN)."""
    y_true = [1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 1, 0]

    report = evaluate(y_true, y_pred)

    assert report.accuracy == 0.0
    assert report.precision == 0.0
    assert report.recall == 0.0
    assert report.f1 == 0.0


# ---------------------------------------------------------------------------
# U2-3 — Oracle: hand-computed mixed confusion matrix
# ---------------------------------------------------------------------------
def test_U2_3_mixed_predictor_matches_handcomputed_confusion_matrix():
    """Oracle: 5-row truth table → TP=2, FP=1, FN=1, TN=1.

    accuracy = (TP+TN)/5 = 3/5
    precision = TP/(TP+FP) = 2/3
    recall    = TP/(TP+FN) = 2/3
    F1        = 2*P*R/(P+R) = 2/3
    """
    y_true = [1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 1, 1]

    tp, fp, fn, tn = confusion_matrix(y_true, y_pred)
    assert (tp, fp, fn, tn) == (2, 1, 1, 1)

    assert accuracy(y_true, y_pred) == pytest.approx(3 / 5)
    assert precision(y_true, y_pred) == pytest.approx(2 / 3)
    assert recall(y_true, y_pred) == pytest.approx(2 / 3)
    assert f1(y_true, y_pred) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# U2-4 — Equivalence partitioning: degenerate predictor (all positive)
# ---------------------------------------------------------------------------
def test_U2_4_all_positive_predictor_no_division_by_zero():
    """Equivalence (degenerate): predictor outputs only 1s.

    TP=3, FP=2, FN=0, TN=0 → precision=3/5, recall=1.0, F1=0.75.
    Crucially, recall's denominator (TP+FN) and precision's (TP+FP) are
    both nonzero here, but F1's combined denom must also avoid div-by-zero.
    """
    y_true = [1, 0, 1, 0, 1]
    y_pred = [1, 1, 1, 1, 1]

    assert precision(y_true, y_pred) == pytest.approx(3 / 5)
    assert recall(y_true, y_pred) == 1.0
    assert f1(y_true, y_pred) == pytest.approx(0.75)
    assert accuracy(y_true, y_pred) == pytest.approx(3 / 5)


# ---------------------------------------------------------------------------
# U2-5 — Boundary-value (lower, defined): single-sample input
# ---------------------------------------------------------------------------
def test_U2_5_single_sample_input_is_valid():
    """BVA: lowest defined input size is n=1. All metrics defined."""
    y_true = [1]
    y_pred = [1]

    report = evaluate(y_true, y_pred)
    assert report.accuracy == 1.0
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1 == 1.0


# ---------------------------------------------------------------------------
# U2-6 — Boundary-value (lower, undefined): empty input must raise (policy P2)
# ---------------------------------------------------------------------------
def test_U2_6_empty_input_raises_value_error():
    """BVA: n=0 is below the defined region; metrics must raise per P2."""
    with pytest.raises(ValueError, match="empty"):
        evaluate([], [])


# ---------------------------------------------------------------------------
# U2-7 — Oracle: ROC-AUC with hand-computed pair count
# ---------------------------------------------------------------------------
def test_U2_7_roc_auc_matches_handcomputed_pair_ranking():
    """Oracle: AUC = (#pos>neg + 0.5*#ties) / (#pos*#neg).

    y_true = [0, 0, 1, 1]
    y_score= [0.1, 0.4, 0.35, 0.8]
    Positives: {0.35, 0.8}, Negatives: {0.1, 0.4}.
    Pairs:  (0.35,0.1) win, (0.35,0.4) loss,
            (0.8,0.1)  win, (0.8,0.4)  win.
    Wins = 3, ties = 0, total pairs = 4 → AUC = 3/4 = 0.75.
    """
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]

    assert roc_auc(y_true, y_score) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# U2-8 — Structural: k-fold CV produces k results and is deterministic
# ---------------------------------------------------------------------------
def test_U2_8_cross_validate_is_kfold_and_deterministic():
    """Structural: cross_validate(k=5) returns 5 reports; identical re-run."""
    X = pd.DataFrame({"x": np.arange(10, dtype=float)})
    y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    def factory() -> FakeModel:
        return FakeModel(feature="x")

    first = cross_validate(factory, X, y, k=5)
    second = cross_validate(factory, X, y, k=5)

    assert len(first) == 5
    assert len(second) == 5
    for a, b in zip(first.folds, second.folds):
        assert a.as_dict() == b.as_dict()
    assert all(
        not math.isnan(r.accuracy) and not math.isnan(r.precision) for r in first.folds
    )
