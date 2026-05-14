# Customer Churn Prediction — CISC 593 P1

Coursework for CISC 593 (Testing & Verification). The application is a Python machine-learning system that predicts customer churn using UCI Machine Learning Repository data; the *grading target* is the unit and system testing built around it.

See:
- [`PropjectProposal.md`](PropjectProposal.md) — original project proposal.
- [`ArchitectureProposal.md`](ArchitectureProposal.md) — module layout and mock-first ML strategy.
- [`TestPlan.md`](TestPlan.md) — units under test, methodologies, test matrix, engineer assignments.

## Repository Layout

```
src/churn/         # application code (one subpackage per module)
tests/unit/        # unit-level tests (mirror of src/churn/)
tests/system/      # end-to-end pipeline tests
tests/fixtures/    # committed CSV fixtures used by tests
data/              # raw downloads — gitignored, not required for tests
```

## Setup

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --only-binary=:all: -e ".[dev]"
```

The `--only-binary=:all:` flag forces wheels and avoids needing a recent C compiler; numpy/pandas builds from source require GCC ≥ 9.3.

## Running Tests

```bash
# Full suite
pytest

# Just unit tests
pytest tests/unit -v

# With coverage report
pytest --cov=churn --cov-report=term-missing
```

## Engineers

| Module | Primary author | Reviewer |
|---|---|---|
| `preprocessing` | Emmanuel Ndone Suum     | Mirriam Chemutai Ronoh |
| `evaluation`    | Janus Thor Kristjansson | Zeyu Wang              |
