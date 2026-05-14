"""Model factory — Phase 1 only registers ``"fake"``."""

from __future__ import annotations

from typing import Any

from churn.modeling.fake import FakeModel
from churn.modeling.protocol import Model


def build_model(name: str, **hyperparams: Any) -> Model:
    if name == "fake":
        return FakeModel(**hyperparams)
    raise ValueError(f"unknown model name: {name!r}")
