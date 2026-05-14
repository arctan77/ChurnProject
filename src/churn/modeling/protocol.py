"""The ``Model`` protocol all classifiers in this project must satisfy."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Model(Protocol):
    """Minimal classifier interface.

    Tests assert against this protocol, never against a concrete class, so
    Phase 2 can introduce sklearn-backed implementations without breaking
    any existing test (see ArchitectureProposal.md §9, "Phase 2 substitution
    rule").
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Model": ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...
