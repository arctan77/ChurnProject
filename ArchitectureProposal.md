# Architecture Proposal — Customer Churn Prediction System
**CISC 593 P1 · Companion to `PropjectProposal.md`**

## 1. Guiding Principles

This is a *testing & verification* course. Every architectural choice below is justified primarily by how it affects testability — not by what would be most idiomatic in a production ML system.

1. **Pure functions wherever possible.** Preprocessing, feature engineering, and metric computation should take inputs and return outputs without touching disk, network, or global state. Pure functions are trivially unit-testable.
2. **I/O at the edges.** Reading datasets, writing reports, and rendering UI live in thin "adapter" modules. Core logic never imports `pandas.read_csv` or `open()` directly — it accepts already-loaded data.
3. **Dependency injection over imports.** A trainer takes a model factory; an evaluator takes a metrics function set; a predictor takes a fitted model. This lets tests swap in fakes/stubs without monkey-patching.
4. **Deterministic by default.** A single `random_state` is threaded through every stochastic call (model init, train/test split, CV fold). Reproducibility is a testing prerequisite.
5. **Explicit contracts.** Module boundaries use typed dataclasses (`@dataclass`) or `TypedDict`s rather than free-form dicts, so tests can assert against a stable shape.
6. **Mock-first ML.** Real sklearn estimators are deferred until the testing apparatus is solid. Phase 1 uses a deterministic `FakeModel` so tests exercise the *system*, not sklearn's behavior. See §9.

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         UI / CLI Layer                          │
│              (Flask or Dash — thin, mostly untested)            │
└──────────────────────────────┬──────────────────────────────────┘
                               │ calls
┌──────────────────────────────▼──────────────────────────────────┐
│                      Orchestration Layer                        │
│   Pipeline objects that compose Core modules end-to-end         │
│   (system-test target)                                          │
└──┬────────────┬────────────┬────────────┬───────────┬───────────┘
   │            │            │            │           │
   ▼            ▼            ▼            ▼           ▼
┌──────┐   ┌────────┐   ┌────────┐   ┌────────┐  ┌─────────┐
│ Data │   │Prepro- │   │Feature │   │Modeling│  │Reporting│
│ I/O  │   │cessing │   │  Eng.  │   │  +     │  │         │
│      │   │        │   │        │   │ Eval   │  │         │
└──────┘   └────────┘   └────────┘   └────────┘  └─────────┘
  ▲           ▲            ▲             ▲            ▲
  │           │            │             │            │
  └───────────┴────────────┴─────────────┴────────────┘
                  Each box = a testable unit
                  (pure, except Data I/O and Reporting)
```

Vertical separation matters: a unit test should be able to import any **Core module** without spinning up Flask, opening a file, or hitting the network.

## 3. Module Breakdown

### 3.1 `data_io/` — Boundary, light tests
- **Responsibility:** load UCI dataset(s) into a `pandas.DataFrame`; nothing else.
- **Public API:** `load_dataset(name: str) -> RawDataset` (a dataclass holding the frame + metadata).
- **Testing:** smoke tests that a known-good fixture loads with the expected schema. Network access is mocked or fixtures are committed.

### 3.2 `preprocessing/` — Pure, heavily tested
- **Responsibility:** missing-value imputation, scaling, encoding, train/test split.
- **Public API:** stateless functions like `impute_numeric(df, strategy) -> df`, `encode_categorical(df, cols, scheme) -> df`, plus a fittable `Preprocessor` class with `.fit(X) / .transform(X)` matching the sklearn idiom (so tests can assert behavioral parity).
- **Testing:** the bulk of unit tests live here — boundary-value analysis on imputation, equivalence partitioning over encoding schemes, idempotency of repeated transforms.

### 3.3 `features/` — Pure, heavily tested
- **Responsibility:** any derived feature (e.g., tenure buckets, ratios, interaction terms).
- **Public API:** a registry of named feature functions: `def add_feature_X(df) -> df`.
- **Testing:** parametrized unit tests covering normal, edge, and degenerate inputs.

### 3.4 `modeling/` — Mostly pure, mid-weight tests
- **Responsibility:** define a uniform `Model` protocol — `fit(X, y)`, `predict(X)`, `predict_proba(X)` — and provide implementations.
- **Phase 1 implementation:** `FakeModel` only. A deterministic, threshold-based classifier: at `fit`-time it records a chosen feature column and a threshold (either configured or learned trivially as the median); at `predict`-time it returns `1` where `feature > threshold`, else `0`. `predict_proba` returns hard 0/1 columns. No randomness, no sklearn dependency in the Core path.
- **Phase 2 implementation:** add real sklearn-backed wrappers (`LogRegModel`, `RFModel`, `GBMModel`) that satisfy the same protocol. The factory grows a name argument — but the protocol and every test against it stay valid.
- **Public API:** factory `build_model(name: Literal["fake", "logreg", "rf", "gbm"], **hyperparams) -> Model`. Phase 1 only registers `"fake"`.
- **Testing:** verify the factory returns an object honoring the protocol; verify `FakeModel` is fully deterministic (same input → identical predictions, no seed needed); verify hyperparameters (threshold, feature column) propagate; verify it integrates with `evaluation/` to produce hand-computable metrics.

### 3.5 `evaluation/` — Pure, heavily tested
- **Responsibility:** compute accuracy, precision, recall, F1, ROC-AUC; run k-fold CV.
- **Public API:** `evaluate(model, X, y) -> MetricsReport`, `cross_validate(model_factory, X, y, k) -> CVReport`.
- **Testing:** assertions against scikit-learn's own metrics on tiny hand-computed inputs (oracle testing); structural tests that CV produces `k` results.

### 3.6 `prediction/` — Pure, heavily tested
- **Responsibility:** apply a fitted pipeline to new data, single-row or batch.
- **Public API:** `predict_one(pipeline, record: dict) -> Prediction`, `predict_batch(pipeline, df) -> df`.
- **Testing:** edge cases — missing fields, unseen categorical levels, empty batch, single-row batch, type coercion.

### 3.7 `reporting/` — Boundary, light tests
- **Responsibility:** render a `MetricsReport` + predictions into PDF/CSV.
- **Public API:** `export_csv(report, path)`, `export_pdf(report, path)`.
- **Testing:** assert files are created with non-zero size and contain expected anchors; visual fidelity is not tested.

### 3.8 `pipelines/` — System-test target
- **Responsibility:** compose the modules above into named end-to-end flows (e.g., `train_and_evaluate`, `predict_from_saved_model`).
- **Testing:** these *are* the system tests. Run on a fixture dataset, assert metrics fall in a sane band, assert all expected artifacts exist.

### 3.9 `ui/` — Thin, minimally tested
- **Responsibility:** Flask routes (or Dash callbacks) that translate HTTP/UI events into pipeline calls.
- **Testing:** a couple of route-level tests using Flask's test client; deeper logic is exercised through the pipeline tests, not the UI.

## 4. Project Structure

```
TestingAndVerification/
├── PropjectProposal.md
├── ArchitectureProposal.md
├── pyproject.toml          # pytest + tooling config
├── README.md
├── src/
│   └── churn/
│       ├── __init__.py
│       ├── data_io/
│       ├── preprocessing/
│       ├── features/
│       ├── modeling/
│       ├── evaluation/
│       ├── prediction/
│       ├── reporting/
│       ├── pipelines/
│       └── ui/
├── tests/
│   ├── unit/               # mirrors src/churn/ structure
│   ├── system/             # end-to-end pipeline tests
│   └── fixtures/           # tiny committed datasets
└── data/                   # gitignored — full UCI downloads
```

The `src/`-layout is chosen specifically so tests cannot accidentally import un-installed code; this catches packaging bugs early.

## 5. Tech Stack & Tooling

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Required by proposal |
| Core libs | pandas, scikit-learn, numpy | Required by proposal |
| Plotting | matplotlib, seaborn | Required by proposal |
| Reporting | reportlab (PDF), pandas (CSV) | Lightweight, scriptable |
| UI | Flask | Simpler than Dash; less to mock in tests |
| **Test framework** | **pytest** | Fixtures, parametrization, plugins |
| Coverage | coverage.py + pytest-cov | Required for any honest test report |
| Property tests | hypothesis (optional) | Great for preprocessing invariants |
| Lint/format | ruff + black | Fast feedback |
| Type checking | mypy (optional) | Catches contract drift |

## 6. Testing Strategy (the actual deliverable)

A two-tier strategy mirroring the proposal:

**Tier 1 — Unit tests** (`tests/unit/`)
- One test module per source module.
- Cover preprocessing, feature engineering, modeling factory, evaluation, prediction.
- Techniques to demonstrate: AAA structure, fixtures, parametrization, equivalence partitioning, boundary-value analysis, oracle testing against sklearn, mock objects for the I/O layer.
- Target: ≥90% line coverage on Core modules.

**Tier 2 — System tests** (`tests/system/`)
- Drive `pipelines.train_and_evaluate` end-to-end on a small committed fixture.
- Assert: file artifacts exist, metrics are within an expected band, no warnings raised, deterministic re-run produces identical output.
- A second test exercises the prediction pipeline on held-out rows.

**Out of scope for testing** (intentional, document in report)
- The Flask UI's HTML rendering.
- The numerical accuracy of sklearn itself (we trust upstream).
- PDF visual layout.

## 7. Open Questions / Decisions to Revisit

1. **Dataset choice** — the proposal says "multiple UCI churn-related datasets." Recommend starting with the **Telco Customer Churn** dataset (well-known, ~7K rows, mixed types) and only adding a second after the testing pipeline is stable.
2. **UI breadth** — Flask routes for (a) trigger training, (b) submit a single record for prediction, (c) download report. Dash adds value only if interactive plots are graded; otherwise Flask is cheaper to test.
3. **Persistence** — do we save fitted models (`joblib`)? If yes, a `model_store/` module is added with its own boundary tests.
4. **CI** — running tests locally is fine for grading, but a GitHub Actions workflow is a few lines and demonstrates discipline; recommend doing it.

## 8. Suggested Build Order

**Phase 1 — Mock-first (the testing scaffolding):**
1. Skeleton + `pyproject.toml` + `pytest` running with one trivial test.
2. `data_io` + a committed fixture → first real unit test.
3. `preprocessing` (largest test surface) → bulk of unit tests.
4. `modeling` with **`FakeModel` only** + `evaluation` → first metrics flowing on a deterministic predictor.
5. `pipelines.train_and_evaluate` driven by `FakeModel` → first system test green end-to-end.
6. `prediction` + `reporting` (also against `FakeModel`).
7. Flask UI — by this point everything it calls is already covered.

**Phase 2 — Real ML (substitution, not rewrite):**

8. Add real sklearn-backed `Model` implementations (`LogRegModel`, `RFModel`, `GBMModel`) behind the existing protocol.
9. Add a thin smoke-test layer: each real model fits a fixture without errors, produces well-formed probabilities, and respects `random_state`.
10. Parametrize the existing system test over `["fake", "logreg", "rf", "gbm"]` to confirm the pipeline is model-agnostic.

## 9. Mock-First ML Strategy

The ML model is the *least testable* part of the system: it's stochastic, slow to fit, and its outputs depend on numerical details we don't want to assert against. So we defer it. Phase 1 ships a fully working pipeline with a `FakeModel` that has none of those problems; Phase 2 substitutes real estimators behind the same interface.

### Why mock first
- **Faster feedback loop.** Unit tests on a `FakeModel` run in milliseconds, not seconds.
- **Deterministic oracles.** When the model is a known threshold rule, we can hand-compute expected metrics for a tiny dataset and assert them exactly. No "metrics within 0.05 of expected" fudging.
- **Forces a clean protocol.** Writing tests against `FakeModel` first means the `Model` interface is shaped by *test ergonomics*, not by sklearn's API quirks. Real models then have to conform — which is the right direction of pressure.
- **Decouples failures.** When a system test fails in Phase 2, we know it's the real model (or its integration), because the same test passes with `FakeModel`.

### The `Model` protocol
```python
from typing import Protocol
import numpy as np
import pandas as pd

class Model(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Model": ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...        # shape (n,)
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...  # shape (n, 2)
```

### `FakeModel` (Phase 1)
A threshold-based deterministic classifier:

```python
@dataclass
class FakeModel:
    feature: str                       # column to threshold on
    threshold: float | None = None     # if None, learned as median at fit-time
    _learned_threshold: float | None = field(default=None, init=False)

    def fit(self, X, y):
        self._learned_threshold = (
            self.threshold if self.threshold is not None else float(X[self.feature].median())
        )
        return self

    def predict(self, X):
        t = self._learned_threshold
        return (X[self.feature].to_numpy() > t).astype(int)

    def predict_proba(self, X):
        preds = self.predict(X)
        return np.column_stack([1 - preds, preds])
```

Properties this gives the test suite:
- **No randomness.** Same input → identical output, every run, every machine.
- **Hand-computable.** A 5-row test dataset has predictions you can write down on paper, which means `evaluation/` tests can use the model's outputs as exact oracles.
- **No sklearn dependency in Phase 1 Core.** Tests fail loudly if anyone smuggles `sklearn` into a Core module before Phase 2.

### What stays mocked vs. real in Phase 1

| Module | Phase 1 | Notes |
|---|---|---|
| `data_io` | **real** | Loads a committed CSV fixture. No network. |
| `preprocessing` | **real** | Fully implemented; this is the largest unit-test surface. |
| `features` | **real** | Same as above. |
| `modeling` | **`FakeModel` only** | Real estimators arrive in Phase 2. |
| `evaluation` | **real** | Metrics are real; oracle inputs come from `FakeModel`. |
| `prediction` | **real** | Wires a fitted `FakeModel` to new inputs. |
| `pipelines` | **real** | End-to-end with `FakeModel`. |
| `reporting` | **real** | Writes real CSVs/PDFs from real `MetricsReport`s. |
| `ui` | **stub** | Routes return placeholder responses until pipelines are stable. |

### Phase 2 substitution rule
Adding real models must not require changing any test that already exists. If a test breaks when we add `LogRegModel`, the test was over-fitted to `FakeModel`'s implementation rather than to the protocol — that's a test-quality bug to fix in Phase 1 before moving on.
