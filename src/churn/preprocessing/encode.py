"""Categorical encoding with semantic-No collapse and out-of-domain raise.

Two pieces of Telco-specific logic live here:

1. ``"No internet service"`` and ``"No phone service"`` are *semantically*
   the same as plain ``"No"`` for downstream modeling — they signal "the
   feature does not apply". We collapse them at fit time so encoders see
   a stable level set.

2. Per TestPlan.md §3 policy P1, encountering a category at transform
   time that wasn't present at fit time raises ``ValueError``. Silent
   routing to an "unknown" bucket would mask data drift.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd


SEMANTIC_NO_VALUES: tuple[str, ...] = ("No internet service", "No phone service")


def collapse_semantic_no(series: pd.Series) -> pd.Series:
    """Replace ``"No internet service"`` / ``"No phone service"`` with ``"No"``.

    Pure: returns a new Series, never mutates the input.
    """
    return series.replace({v: "No" for v in SEMANTIC_NO_VALUES})


def fit_categorical_encoder(df: pd.DataFrame, columns: list[str]) -> dict[str, list[str]]:
    """Record the sorted unique values of each column after the semantic-No
    collapse. The returned dict is the encoder's "fitted state"."""
    state: dict[str, list[str]] = {}
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"column missing from fit input: {col!r}")
        collapsed = collapse_semantic_no(df[col].astype("string"))
        state[col] = sorted(collapsed.dropna().unique().tolist())
    return state


def transform_categorical(
    df: pd.DataFrame, fitted_state: Mapping[str, list[str]]
) -> pd.DataFrame:
    """Apply the fitted encoder. Unseen categories raise (policy P1)."""
    out = df.copy()
    for col, known in fitted_state.items():
        if col not in out.columns:
            raise KeyError(f"column missing from transform input: {col!r}")
        collapsed = collapse_semantic_no(out[col].astype("string"))
        unseen = set(collapsed.dropna().unique()) - set(known)
        if unseen:
            sample = sorted(unseen)[0]
            raise ValueError(
                f"unseen category in column {col!r}: {sample!r} "
                f"(known categories: {known})"
            )
        out[col] = collapsed
    return out
