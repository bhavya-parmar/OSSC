# One-Shot Strategic Classification Under Unknown Costs (ICML 2024)

A Python implementation of the ICML 2024 paper:
> **"One-Shot Strategic Classification Under Unknown Costs"**  
> *Nir Rosenfeld, Lee Cohen, and Nir Rosenfeld (ICML 2024)*

CS620: Incentives & ML - Use-case Implementation\
Submitted by: Bhavya Parmar\
Roll No.: 23110059

---

## Table of Contents
- [Overview](#overview)
- [Repository Layout](#repository-layout)
- [Installation & Quickstart](#-installation--quickstart)
- [Generated Reports & Visualizations](#-generated-reports--visualizations)
- [Automated Unit Test Suite](#-automated-unit-test-suite)

---

## Overview

Standard supervised machine learning assumes test data is drawn from a stationary distribution. In high-stakes socio-technical domains (credit lending, college admissions, fraud detection), users are **rational economic agents** who actively manipulate their features to receive favorable predictions:
$$\Delta^*(x) = \arg\max_{x'} \left\{ \mathbb{I}\{\beta^\top x' + b \ge 0\} \cdot u - c(x, x') \right\}$$

Prior strategic classification literature typically assumed the institution knows the agents' exact cost function $c(x, x')$. This paper explores the realistic **one-shot setting**, where:
1. The institution must commit to a single classifier model $\beta$ at deployment.
2. The true cost matrix $\Sigma^*$ is **unknown** and cannot be learned through iterative rounds.
3. The cost is known only to belong to a continuous uncertainty set $\mathcal{C}$:
   $$\mathcal{C} = \left\{ \Sigma = \text{diag}(\sigma_1^2, \dots, \sigma_d^2) : \sigma_i \in [\sigma_{i,\min}, \sigma_{i,\max}] \right\}$$

This repository provides an end-to-end, object-oriented pipeline featuring:
- **`MAXLOSSCOST` (Algorithm 1)**: An exact $\mathcal{O}(nd + n\ln n)$ piecewise-linear breakpoint sweep that finds the worst-case cost matrix for any classifier $\beta$.
- **Full-Batch Subgradient Method (Algorithm 3)**: A minimax optimization loop minimizing the worst-case strategic hinge loss with structure-aware **Dual Norm Regularization** ($\lambda u^* \|\beta\|_{*,\Sigma}$), which provably restores convexity (Proposition 4.3).
- **Realistic Loan Approval Evaluation**: Evaluates robustness against a standard, naive Logistic Regression baseline on a real-world dataset (`loan_data.csv`).

---

## Repository Layout


```
OSSC/
├── data/
│   └── loan_data.csv                  # Real-world loan approval dataset (45,000 samples)
│
├── src/                               # Core Python package
│   ├── __init__.py                    # Public package API exports
│   │
│   ├── data/                          # Data loading & preprocessing
│   │   ├── __init__.py
│   │   └── loader.py                  # Cleaning, one-hot encoding, StandardScaler, label mapping
│   │
│   ├── costs/                         # Cost uncertainty models & bounds
│   │   ├── __init__.py
│   │   ├── uncertainty_set.py         # CostUncertaintySet class, dual norm, true cost sampling
│   │   └── domain_bounds.py           # Domain-grounded cost bounds for credit lending features
│   │
│   ├── algorithms/                    # Inner maximization solvers
│   │   ├── __init__.py
│   │   └── max_loss_cost.py           # Fast O(nd + n ln n) piecewise-linear breakpoint sweep
│   │
│   ├── models/                        # Supervised & strategic classifiers
│   │   ├── __init__.py
│   │   ├── baseline.py                # Naive Logistic Regression baseline wrapper
│   │   └── strategic_classifier.py    # Minimax Strategic Classifier (Algorithm 3, subgradient)
│   │
│   ├── simulation/                    # Strategic behavior simulator
│   │   ├── __init__.py
│   │   └── strategic_response.py      # Rational applicant utility-maximizing manipulation
│   │
│   └── evaluation/                    # Metrics & reporting
│       ├── __init__.py
│       └── metrics.py                 # Accuracy, FPR, TPR, Precision, Markdown & JSON reports
│
├── experiments/                       # Experiment pipeline runners
│   ├── __init__.py
│   ├── run_master_pipeline.py         # Unified 4-pillar experimental engine
│   └── run_experiments.py             # Single-run prototype (Exp 1 vs Exp 2)
│
├── reports/                           # Output artifacts
│   ├── figures/                       # High-resolution visualization charts
│   │   ├── master_summary_dashboard.png # Comprehensive 6-panel executive dashboard
│   │   ├── multiseed_cv_comparison.png# 3-Panel Mean +/- Std Cross-Validation Comparison
│   │   ├── social_cost_gaming_tax.png # Gaming Tax FNR curve & Pareto Trade-off Frontier
│   │   ├── utility_sweep_curves.png   # FPR, Accuracy, and Gaming vs utility u
│   │   ├── cost_scenarios_comparison.png # 4-Panel comparison across Nature cost scenarios
│   │   └── diameter_sweep_tradeoff.png# Gaming Tax escalation & accuracy vs Diameter D(C)
│   ├── master_experiment_results.json # Comprehensive JSON metrics across all pillars
│   └── master_experiment_report.md    # Master academic markdown report
│
├── tests/                             # Comprehensive unit test suite
│   ├── __init__.py
│   ├── test_data_loader.py            # Tests for data preprocessing & scaling
│   ├── test_uncertainty_set.py        # Tests for uncertainty bounds, dual norms & scenarios
│   ├── test_max_loss_cost.py          # Verifies sweep matches brute-force grid search (<1e-4)
│   ├── test_strategic_classifier.py   # Tests scikit-learn compliance & convergence
│   └── test_strategic_simulator.py    # Tests rational agent movement & boundary crossing
│
├── pyproject.toml                     # Modern package build configuration
├── requirements.txt                   # Production dependencies
├── run_all.py                         # Master unified CLI entrypoint (all 4 experimental pillars)
├── run.py                             # Quick CLI entrypoint (single prototype run)
├── strategic_classification.py        # Backwards-compatible import wrapper
└── README.md                          # Comprehensive project documentation
```

---

## Installation & Quickstart

### 1. Requirements
Ensure you have Python 3.9+ installed:
```bash
pip install -r requirements.txt
```

Alternatively, install in editable mode:
```bash
pip install -e .
```

### 2. Master Unified Pipeline (Recommended)
Run all 4 experimental pillars end-to-end across multiple seeds and utilities in one command:
```bash
python run_all.py
```
Or customize parameters:
```bash
python run_all.py --utility 0.5 --iters 100 --seeds 42 101 2024 7 99
```

You can also run specific experimental stages using the `--stage` flag:
- **Only Cross-Validation & Statistical Significance (Pillar 1)**:
  ```bash
  python run_all.py --stage cv
  ```
- **Only Utility Sweep Stress-Test (Pillar 2)**:
  ```bash
  python run_all.py --stage utility
  ```
- **Only Nature Cost Scenarios Across Seeds (Pillar 3)**:
  ```bash
  python run_all.py --stage scenarios
  ```
- **Only Uncertainty Set Diameter Sweep (Pillar 4)**:
  ```bash
  python run_all.py --stage diameter
  ```

### 3. Quick Single-Run Prototype
Run a fast 1-second sanity check on a single seed:
```bash
python run.py
```

### 4. Running Automated Tests
Execute the entire test suite using `pytest`:
```bash
python -m pytest tests -v
```

---

## Generated Reports & Visualizations

Running the pipeline automatically saves evaluation artifacts in `reports/`:
- **`reports/master_experiment_report.md`**: Full Markdown report with paired $t$-tests, scenario comparisons, and diameter scaling.
- **`reports/master_experiment_results.json`**: Machine-readable JSON metrics across all 4 experimental pillars.
- **`reports/figures/master_summary_dashboard.png`**: 6-panel executive dashboard summarizing all experimental findings.
- **`reports/figures/multiseed_cv_comparison.png`**: Cross-validation error bars for Accuracy, FPR surge, and FNR Gaming Tax.
- **`reports/figures/utility_sweep_curves.png`**: 3-panel incentive stress-test across $u \in [0.1, 1.0]$.
- **`reports/figures/cost_scenarios_comparison.png`**: Multi-panel evaluation across Nature's cost scenarios.
- **`reports/figures/diameter_sweep_tradeoff.png`**: Theoretical scaling and gaming tax escalation vs. diameter $D(\mathcal{C})$.
- **`reports/figures/social_cost_gaming_tax.png`**: Gaming Tax FNR curve and Pareto Trade-off Frontier.

---

## Automated Unit Test Suite

The test suite in `tests/` tests all components:
- `test_data_loader.py`: Verifies null checks, categorical encoding, label mapping to $\{-1, +1\}$, and feature scaling.
- `test_uncertainty_set.py`: Verifies monotonicity of dual norms, positive definite constraints, and true cost sampling.
- `test_max_loss_cost.py`: Validates that the fast $\mathcal{O}(nd + n\ln n)$ piecewise sweep matches brute-force grid search within $10^{-4}$ tolerance.
- `test_strategic_classifier.py`: Validates scikit-learn estimator compliance (`fit`, `predict`, `decision_function`) and monotonic loss descent.
- `test_strategic_simulator.py`: Verifies applicant gaming conditions, optimal displacement vectors, and strict boundary crossing.
