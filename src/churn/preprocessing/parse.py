"""Parse raw CSV string fields into typed values.

The Telco dataset's ``TotalCharges`` column is the canonical case: it
ships as strings, with blanks for tenure=0 customers and the occasional
non-numeric junk. Per TestPlan.md §3 policy P3, blanks for tenure=0 are
imputed to 0.0; non-numeric strings still raise.
"""

from __future__ import annotations


def parse_total_charges(value: str | float | int, tenure: int | None = None) -> float:
    """Convert a raw ``TotalCharges`` field to a float.

    - Numeric strings parse normally.
    - Blank/whitespace strings impute to 0.0 *only when* ``tenure == 0``;
      otherwise raise (a blank for a non-zero-tenure row is corrupt data).
    - Non-numeric strings always raise ``ValueError``.
    """
    if isinstance(value, (int, float)):
        return float(value)

    text = value.strip() if isinstance(value, str) else ""

    if text == "":
        if tenure == 0:
            return 0.0
        raise ValueError(
            f"blank TotalCharges with tenure={tenure!r}; "
            "blanks are only imputable when tenure == 0 (policy P3)"
        )

    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"could not parse TotalCharges value: {value!r}") from exc
