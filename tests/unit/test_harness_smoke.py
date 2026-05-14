"""Smoke test: pytest discovers tests, the package imports, version is set.

Exists solely to verify the harness is wired up — delete or replace once
real unit tests land.
"""

import churn


def test_package_imports():
    assert churn.__version__ == "0.0.1"
