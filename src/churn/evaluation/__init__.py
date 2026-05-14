"""Evaluation layer — pure metric functions and cross-validation."""

from churn.evaluation.metrics import (
    MetricsReport,
    accuracy,
    confusion_matrix,
    evaluate,
    f1,
    precision,
    recall,
    roc_auc,
)
from churn.evaluation.cross_validate import CVReport, cross_validate

__all__ = [
    "MetricsReport",
    "CVReport",
    "accuracy",
    "confusion_matrix",
    "cross_validate",
    "evaluate",
    "f1",
    "precision",
    "recall",
    "roc_auc",
]
