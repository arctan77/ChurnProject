"""Modeling layer.

Phase 1 ships only ``FakeModel`` — a deterministic, threshold-based
classifier that lets the test suite exercise the pipeline without sklearn's
randomness. Real sklearn-backed models slot in behind the same ``Model``
protocol in Phase 2 (see ArchitectureProposal.md §9).
"""

from churn.modeling.protocol import Model
from churn.modeling.fake import FakeModel
from churn.modeling.factory import build_model

__all__ = ["Model", "FakeModel", "build_model"]
