"""``Preprocessor`` — sklearn-style fit/transform wrapper over the pure helpers.

Tests assert on the public ``fit`` / ``transform`` contract (idempotency,
out-of-domain raise) so swapping the internal implementation later does
not break the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from churn.preprocessing.encode import fit_categorical_encoder, transform_categorical
from churn.preprocessing.parse import parse_total_charges


DEFAULT_CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
)


@dataclass
class Preprocessor:
    """Fit/transform pipeline over a Telco-shaped DataFrame.

    Stateless until ``fit`` is called. ``transform`` is pure with respect
    to its input and idempotent against itself (U1-8).
    """

    categorical_columns: tuple[str, ...] = DEFAULT_CATEGORICAL_COLUMNS
    _fitted_state: dict[str, list[str]] | None = field(default=None, init=False, repr=False)
    _is_fitted: bool = field(default=False, init=False, repr=False)

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        cols = [c for c in self.categorical_columns if c in X.columns]
        self._fitted_state = fit_categorical_encoder(X, cols)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted or self._fitted_state is None:
            raise RuntimeError("Preprocessor.transform called before fit")

        out = X.copy()

        if "TotalCharges" in out.columns:
            tenure_series = out.get("tenure")
            parsed: list[float] = []
            for i, raw in enumerate(out["TotalCharges"].tolist()):
                t = int(tenure_series.iloc[i]) if tenure_series is not None else None
                parsed.append(parse_total_charges(raw, tenure=t))
            out["TotalCharges"] = parsed

        out = transform_categorical(out, self._fitted_state)
        return out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
