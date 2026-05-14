"""Derived features. Pure functions over a DataFrame."""

from __future__ import annotations

import pandas as pd


def charges_per_tenure(df: pd.DataFrame) -> pd.Series:
    """``MonthlyCharges / tenure``, with tenure=0 mapped to 0.0.

    A naive division would emit ``inf`` for new customers (tenure=0,
    boundary case U1-4 in TestPlan.md §4). We map those rows to 0.0
    instead — semantically, "no per-month-of-tenure history yet."
    """
    if "tenure" not in df.columns or "MonthlyCharges" not in df.columns:
        missing = [c for c in ("tenure", "MonthlyCharges") if c not in df.columns]
        raise KeyError(f"required columns missing: {missing}")

    tenure = df["tenure"].astype(float)
    monthly = df["MonthlyCharges"].astype(float)
    safe_tenure = tenure.where(tenure != 0, other=1.0)
    ratio = monthly / safe_tenure
    return ratio.where(tenure != 0, other=0.0)
