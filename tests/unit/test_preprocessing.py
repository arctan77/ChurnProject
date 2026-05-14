"""Unit tests for ``churn.preprocessing``.

Each test maps 1:1 to a numbered case in TestPlan.md §4. Test docstrings
restate the methodology and the partition / boundary the case covers, so
the unit-test report can be assembled directly from this file's output.
"""

from __future__ import annotations

import pandas as pd
import pytest

from churn.preprocessing import (
    Preprocessor,
    charges_per_tenure,
    parse_total_charges,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _minimal_fit_frame() -> pd.DataFrame:
    """Smallest frame that fits the encoder with all categorical levels we test."""
    return pd.DataFrame(
        {
            "tenure": [1, 12, 24],
            "MonthlyCharges": [29.85, 56.95, 108.15],
            "TotalCharges": ["29.85", "683.40", "2595.60"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "OnlineSecurity": ["No", "Yes", "No internet service"],
            "InternetService": ["DSL", "Fiber optic", "No"],
        }
    )


# ---------------------------------------------------------------------------
# U1-1 — Equivalence partitioning: valid numeric string
# ---------------------------------------------------------------------------
def test_U1_1_parse_total_charges_valid_numeric_string():
    """Equivalence partition: well-formed numeric string → float."""
    assert parse_total_charges("29.85") == 29.85


# ---------------------------------------------------------------------------
# U1-2 — Equivalence partitioning: blank-for-tenure-0 (policy P3)
# ---------------------------------------------------------------------------
def test_U1_2_parse_total_charges_blank_with_tenure_zero_imputes_to_zero():
    """Equivalence partition: blank string + tenure=0 → 0.0 (policy P3)."""
    assert parse_total_charges(" ", tenure=0) == 0.0
    assert parse_total_charges("", tenure=0) == 0.0


# ---------------------------------------------------------------------------
# U1-3 — Equivalence partitioning: invalid (non-numeric) string
# ---------------------------------------------------------------------------
def test_U1_3_parse_total_charges_non_numeric_string_raises():
    """Equivalence partition: non-numeric content → ValueError naming the value."""
    with pytest.raises(ValueError, match="abc"):
        parse_total_charges("abc")


# ---------------------------------------------------------------------------
# U1-4 — Boundary-value (lower): tenure=0 must not divide-by-zero
# ---------------------------------------------------------------------------
def test_U1_4_charges_per_tenure_handles_tenure_zero_boundary():
    """BVA lower bound: tenure=0 with non-zero MonthlyCharges → 0.0 (no inf)."""
    df = pd.DataFrame({"tenure": [0], "MonthlyCharges": [29.85]})
    result = charges_per_tenure(df)
    assert result.iloc[0] == 0.0
    assert not pd.isna(result.iloc[0])
    assert result.iloc[0] != float("inf")


# ---------------------------------------------------------------------------
# U1-5 — Boundary-value (observed upper): tenure=72 passes through unmodified
# ---------------------------------------------------------------------------
def test_U1_5_charges_per_tenure_at_observed_upper_boundary():
    """BVA upper bound: tenure=72 → MonthlyCharges/72, no clamping."""
    df = pd.DataFrame({"tenure": [72], "MonthlyCharges": [144.0]})
    result = charges_per_tenure(df)
    assert result.iloc[0] == pytest.approx(144.0 / 72.0)


# ---------------------------------------------------------------------------
# U1-6 — Equivalence partitioning: semantic-class collapse
# ---------------------------------------------------------------------------
def test_U1_6_no_internet_service_encoded_identically_to_no():
    """Equivalence partition: 'No internet service' ≡ 'No' after transform."""
    fit_df = _minimal_fit_frame()
    pre = Preprocessor().fit(fit_df)

    transformed = pre.transform(fit_df)
    online_security = transformed["OnlineSecurity"].tolist()

    assert "No internet service" not in online_security
    assert online_security[2] == "No"
    assert online_security[0] == "No"


# ---------------------------------------------------------------------------
# U1-7 — Equivalence partitioning: out-of-domain category (policy P1)
# ---------------------------------------------------------------------------
def test_U1_7_unseen_category_at_transform_raises():
    """Equivalence partition: category not seen at fit → ValueError (policy P1)."""
    fit_df = _minimal_fit_frame()
    pre = Preprocessor().fit(fit_df)

    bad = fit_df.copy()
    bad.loc[0, "Contract"] = "Maybe"

    with pytest.raises(ValueError, match="unseen category"):
        pre.transform(bad)


# ---------------------------------------------------------------------------
# U1-8 — Property-based: idempotency of transform
# ---------------------------------------------------------------------------
def test_U1_8_transform_is_idempotent():
    """Property: transform(transform(X)) == transform(X) for every fit input.

    Tested at example level here (the TestPlan calls this 'property-style');
    a Hypothesis-based generator would be the natural Phase-2 upgrade.
    """
    fit_df = _minimal_fit_frame()
    pre = Preprocessor().fit(fit_df)

    once = pre.transform(fit_df)
    twice = pre.transform(once)

    pd.testing.assert_frame_equal(once, twice)
