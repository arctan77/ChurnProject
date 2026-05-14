"""FakeModel — deterministic threshold classifier (Phase 1 only)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FakeModel:
    """Threshold-based deterministic binary classifier.

    At fit time it records the threshold (either explicitly configured or
    learned as the median of the chosen feature column). At predict time
    it returns 1 where the feature exceeds the threshold and 0 otherwise.

    No randomness, no sklearn dependency, no numerical fudge — predictions
    are exactly reproducible across machines and runs.
    """

    feature: str
    threshold: float | None = None
    _learned_threshold: float | None = field(default=None, init=False, repr=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FakeModel":
        if self.feature not in X.columns:
            raise KeyError(f"feature column not in X: {self.feature!r}")
        self._learned_threshold = (
            float(self.threshold)
            if self.threshold is not None
            else float(X[self.feature].median())
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        t = self._require_fitted()
        return (X[self.feature].to_numpy() > t).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.predict(X)
        return np.column_stack([1 - preds, preds]).astype(float)

    def _require_fitted(self) -> float:
        if self._learned_threshold is None:
            raise RuntimeError("FakeModel.predict called before fit")
        return self._learned_threshold
