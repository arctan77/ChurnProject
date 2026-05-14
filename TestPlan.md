# Test Plan — Unit Testing Assignment
**CISC 593 P1 · Companion to `PropjectProposal.md` and `ArchitectureProposal.md`**

This document is the binding plan for the **Unit Test (1)** group assignment. It records *which* units are being tested, *why* those units were chosen, *what* methodologies will be applied, and *which specific cases* satisfy each methodology. The unit test reports submitted for grading will be derived directly from §4 and §5 below.

## 1. Scope & Rationale

Two units have been selected: **`preprocessing`** and **`evaluation`**.

They were chosen against three criteria:

1. **Pure functions, no I/O.** Both modules accept in-memory data and return in-memory data. The reports can focus on *test technique* rather than on I/O mocking, which is what the rubric rewards.
2. **Rich, naturally enumerable input space.** Both modules' inputs partition cleanly — typed CSV fields with explicit categorical levels for `preprocessing`, and tabular `(y_true, y_pred)` arrays for `evaluation` — so equivalence partitioning and boundary-value analysis apply non-trivially.
3. **Methodology variety across the pair.** `preprocessing` showcases equivalence partitioning + boundary-value analysis + an idempotency property; `evaluation` showcases oracle testing against hand-computed truth tables + degenerate-input equivalence classes. The two reports therefore demonstrate *different* techniques rather than reusing the same playbook.

The pair also tells one coherent story end-to-end: a single committed fixture flows through `preprocessing` into a `FakeModel` (deterministic, see `ArchitectureProposal.md` §9) and out through `evaluation`, so the unit reports later compose cleanly into the system-test report.

## 2. Shared Fixture

A single hand-crafted CSV (`tests/fixtures/telco_mini.csv`, 8–12 rows) is the canonical input for both units. It is engineered so every textbook edge case appears at least once:

- A normal numeric/categorical row.
- A `TotalCharges = " "` row (raw-CSV blank-string parser case).
- A `tenure = 0` row (lower-boundary, divide-by-zero trap for derived features).
- A row with every "No internet service" / "No phone service" implicit-missing categorical.
- A balanced split of `Churn = Yes` / `No` so the metric oracle examples are computable.
- Both senior and non-senior, both `Contract = Month-to-month` and longer.

Reusing one fixture across units is deliberate: it keeps the report consistent and gives the system-test report later a single artifact to cite.

## 3. Policy Decisions Locked In

These are choices the test plan must commit to, since both the implementation *and* the tests need to agree on the contract.

| # | Question | Decision | Rationale |
|---|---|---|---|
| P1 | Out-of-domain categorical at `transform` time (e.g., fit on `{Yes,No}`, see `Maybe`) | **Raise `ValueError("unseen category: <value>")`** | Loud failure beats silent miscoding. A model can't recover from a category it never saw at fit time, and silently routing to an "unknown" bucket would mask data drift. Tests can assert the exception message, which is cleaner than asserting on a sentinel. |
| P2 | Empty input to evaluation metrics (`y_true=[]`, `y_pred=[]`) | **Raise `ValueError("metrics undefined for empty input")`** | F1 and ROC-AUC are mathematically undefined on empty input. A sentinel `MetricsReport` would propagate as zero or NaN through downstream reporting. Raising is the testable contract; the UI layer wraps and presents it. |
| P3 | `TotalCharges` blank string for `tenure = 0` rows | **Impute to `0.0`** | A customer with zero months tenure has logically paid nothing. This is the documented Telco-dataset convention and avoids bleeding NaN into downstream features. Non-numeric strings (`"abc"`) still raise — only the blank-for-tenure-0 case is imputed. |

## 4. Unit 1 — `preprocessing`

**Source files (anticipated):** `src/churn/preprocessing/__init__.py`, `src/churn/preprocessing/parse.py`, `src/churn/preprocessing/encode.py`, `src/churn/preprocessing/impute.py`, `src/churn/preprocessing/preprocessor.py`.

**Methodology:** *Equivalence partitioning + boundary-value analysis*, with one *property-based* idempotency check.

**Why this methodology:** the input space is a typed CSV row whose fields decompose into discrete partitions: numeric fields split into {valid, blank, non-numeric, negative}; categorical fields split into {known levels, "implicit-missing" levels, unseen levels at transform time}. Each partition has a clear contract, so equivalence partitioning gives high coverage with few cases. `tenure` has meaningful boundaries (0, 72), so BVA picks up the divide-by-zero risk in derived features. Idempotency (`transform(transform(X)) == transform(X)`) is an invariant the design *should* satisfy and is cheap to test as a property.

### 4.1 Test cases

| # | Methodology | Input | Expected output |
|---|---|---|---|
| **U1-1** | Equiv. partitioning — valid numeric | `parse_total_charges("29.85")` | returns `29.85` (float) |
| **U1-2** | Equiv. partitioning — blank-for-tenure-0 | `parse_total_charges(" ", tenure=0)` | returns `0.0` (per P3) |
| **U1-3** | Equiv. partitioning — invalid string | `parse_total_charges("abc")` | raises `ValueError`, message contains `"abc"` |
| **U1-4** | Boundary-value analysis (lower) | full row with `tenure=0`, `MonthlyCharges=29.85`; compute `charges_per_tenure` derived feature | feature returns `0.0` (no `ZeroDivisionError`, no `inf`) |
| **U1-5** | Boundary-value analysis (observed upper) | full row with `tenure=72` | passes through unmodified; no clamping, no warning |
| **U1-6** | Equiv. partitioning — semantic-class collapse | `OnlineSecurity = "No internet service"` | encodes identically to `OnlineSecurity = "No"` after `Preprocessor.transform` |
| **U1-7** | Equiv. partitioning — out-of-domain (P1) | `Preprocessor.fit(X with Contract ∈ {Yes,No})`, then `.transform(X' with Contract = "Maybe")` | raises `ValueError`, message contains `"unseen category"` and `"Maybe"` |
| **U1-8** | Property-based — idempotency | fitted `Preprocessor`; assert `transform(transform(X)).equals(transform(X))` on the full fixture | DataFrames equal |

8 cases, 3 distinct methodologies. Every case is a separate `pytest` function with `parametrize` reserved for U1-1/U1-2/U1-3 (same callable, different partitions).

## 5. Unit 2 — `evaluation`

**Source files (anticipated):** `src/churn/evaluation/__init__.py`, `src/churn/evaluation/metrics.py`, `src/churn/evaluation/cross_validate.py`.

**Methodology:** *Oracle testing* (hand-computed expected values from a 4–5 row truth table) supplemented by *equivalence partitioning* over degenerate predictor behavior and *boundary-value analysis* on input size.

**Why this methodology:** classification metrics have closed-form definitions over a confusion matrix. On a 5-row dataset the confusion matrix can be written down on paper, so each metric's expected value is computed exactly — no `pytest.approx` floating-point fudge, no "within 0.05 of expected" hand-waving. Oracle testing here is the *strongest possible* signal of correctness because the test asserts the textbook formula directly. Equivalence partitioning over predictor behavior (perfect / all-wrong / single-class / mixed) covers the regions where divide-by-zero risks live. Boundary-value analysis on input size (empty, single-row) catches contract-violation bugs cheaply.

### 5.1 Test cases

For cases U2-1 through U2-3, the truth table is:

| Row | y_true | y_pred (case U2-3) |
|-----|--------|--------------------|
| 1   | 1      | 1                  |
| 2   | 1      | 0                  |
| 3   | 0      | 0                  |
| 4   | 0      | 1                  |
| 5   | 1      | 1                  |

For U2-3 this gives TP=2, FP=1, FN=1, TN=1 → precision=2/3, recall=2/3, F1=2/3, accuracy=3/5.

| # | Methodology | Input | Expected output |
|---|---|---|---|
| **U2-1** | Oracle — perfect classifier | `y_true=[1,0,1,0,1]`, `y_pred=[1,0,1,0,1]` | accuracy=1.0, precision=1.0, recall=1.0, F1=1.0 |
| **U2-2** | Oracle — all-wrong classifier | `y_true=[1,0,1,0,1]`, `y_pred=[0,1,0,1,0]` | accuracy=0.0, precision=0.0, recall=0.0, F1=0.0 |
| **U2-3** | Oracle — mixed (hand-computed confusion matrix) | `y_true=[1,1,0,0,1]`, `y_pred=[1,0,0,1,1]` | accuracy=3/5, precision=2/3, recall=2/3, F1=2/3 |
| **U2-4** | Equiv. partitioning — degenerate predictor (all-positive) | `y_true=[1,0,1,0,1]`, `y_pred=[1,1,1,1,1]` | recall=1.0; precision=3/5; F1 defined; *no* `ZeroDivisionError` |
| **U2-5** | Boundary-value analysis (lower, defined) | `y_true=[1]`, `y_pred=[1]` | accuracy=1.0; precision=1.0; recall=1.0; F1=1.0 |
| **U2-6** | Boundary-value analysis (lower, undefined) — per P2 | `y_true=[]`, `y_pred=[]` | raises `ValueError`, message contains `"empty"` |
| **U2-7** | Oracle — ROC-AUC with `predict_proba` | `y_true=[0,0,1,1]`, `y_score=[0.1, 0.4, 0.35, 0.8]` | AUC = 0.75 (hand-computed: 3 of 4 ranking pairs correct) |
| **U2-8** | Structural — k-fold CV shape & determinism | `cross_validate(FakeModel factory, fixture, k=5)`, run twice with the same seed | both runs return `len() == 5`; results are bitwise identical |

8 cases, 4 distinct methodologies. U2-3 and U2-7 are the showpiece oracle cases that will appear in the report with full math worked out.

## 6. Coverage & Methodology Justification

The rubric demands *"reasoning why the methodology used achieves good test coverage."* The argument the reports will make:

- **Equivalence partitioning** covers the input space by partition rather than by enumeration. Every partition has at least one representative case; together they exhaust the contract.
- **Boundary-value analysis** complements partitioning by deliberately probing the *edges* of partitions, where defects historically cluster (off-by-one, divide-by-zero, NaN propagation).
- **Oracle testing** removes test-result ambiguity entirely: the expected value is computed by hand from a textbook formula on a small dataset, and the test asserts equality. This is the highest-confidence form of correctness check available to us short of formal verification.
- **Property-based / structural checks** assert *invariants* that should hold across the entire input space (idempotency, CV-output shape, determinism), catching bug classes that example-based tests miss.

Together these techniques are complementary, not redundant: each catches a different bug class. The reports will state this explicitly.

## 7. Engineer Assignments

The rubric requires each group member's contribution to be attributable. Four engineers are split into two pairs, one pair per unit. Within each pair, one engineer is **primary author** (writes the cases, runs them, drafts the report) and the other is **reviewer** (independently re-runs the cases, verifies the actual outputs match, signs off on the methodology section). This satisfies the "features unit tested by each group member" requirement for all four members and gives each unit a second pair of eyes before submission.

| Unit | Primary author | Reviewer | Date of testing |
|---|---|---|---|
| `preprocessing` | Emmanuel Ndone Suum    | Mirriam Chemutai Ronoh | _TBD_ |
| `evaluation`    | Janus Thor Kristjansson | Zeyu Wang              | _TBD_ |

Both authors will be listed under the **Engineers** field of their unit's report; the reviewer's sign-off appears in a "Reviewed by" line at the bottom of the same report. Date of testing is filled in on the day each unit's automated suite is executed and its actual outputs captured.

## 8. What This Plan Does *Not* Cover (and Why)

The other Phase-1 modules (`data_io`, `features`, `modeling`, `prediction`, `pipelines`, `reporting`, `ui`) are out of scope for the **Unit Test (1)** assignment but will be covered later:

- `data_io` and `reporting` are I/O boundaries; their tests are smoke-level and don't exhibit interesting methodology variety. They belong in the system-test deliverable, not here.
- `modeling` (Phase 1) is just `FakeModel` — too thin to fill a unit-test report.
- `pipelines` is the system-test target by design (see `ArchitectureProposal.md` §3.8).
- `features`, `prediction`, `ui` are valid unit-test candidates but adding them dilutes the methodology-quality bar; the rubric weighs depth over breadth (80 points for test-case quality vs. 20 for environment docs).

Two units, deeply tested, with explicit methodology rationale, is a stronger submission than seven units tested superficially.

## 9. Next Step

Scaffold the project skeleton (Task #1) so the test cases above have a concrete codebase to land in. The `tests/fixtures/telco_mini.csv` file is the first artifact to commit after the skeleton itself.
