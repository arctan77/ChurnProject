"""Preprocessing layer — pure transforms over a Telco-shaped DataFrame."""

from churn.preprocessing.parse import parse_total_charges
from churn.preprocessing.encode import (
    SEMANTIC_NO_VALUES,
    collapse_semantic_no,
    fit_categorical_encoder,
    transform_categorical,
)
from churn.preprocessing.features import charges_per_tenure
from churn.preprocessing.preprocessor import Preprocessor

__all__ = [
    "Preprocessor",
    "SEMANTIC_NO_VALUES",
    "charges_per_tenure",
    "collapse_semantic_no",
    "fit_categorical_encoder",
    "parse_total_charges",
    "transform_categorical",
]
