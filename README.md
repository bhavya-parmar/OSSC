# One-Shot Strategic Classification Under Unknown Costs (ICML 2024)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Clean & Modular](https://img.shields.io/badge/code%20style-modular-green.svg)](src/)

A production-level, modular Python implementation of the ICML 2024 paper:
> **"One-Shot Strategic Classification Under Unknown Costs"**  
> *Nir Rosenfeld, Lee Cohen, and Nir Rosenfeld (ICML 2024)*

---

## 📑 Table of Contents
- [Executive Overview](#-executive-overview)
- [Modular Architecture & Repository Layout](#-modular-architecture--repository-layout)
- [Installation & Quickstart](#-installation--quickstart)
- [Mathematical Formulation & Theory](#-mathematical-formulation--theory)
- [Experimental Results on Loan Approval Dataset](#-experimental-results-on-loan-approval-dataset)
- [Generated Reports & Visualizations](#-generated-reports--visualizations)
- [Automated Unit Test Suite](#-automated-unit-test-suite)

---

## 📌 Executive Overview

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

## 🏛 Modular Architecture & Repository Layout

The codebase has been refactored into a production-level, object-oriented package structure under the `OSSC/` repository:

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
│   └── run_experiments.py             # CLI runner orchestrating Exp 1 & Exp 2 + artifact generation
│
├── reports/                           # Output artifacts
│   ├── figures/                       # High-resolution visualization charts
│   │   ├── accuracy_comparison.png    # Truthful vs Manipulated Accuracy bar chart
│   │   ├── fpr_comparison.png         # Truthful vs Manipulated FPR surge bar chart
│   │   └── training_loss.png          # Subgradient optimization convergence curve
│   ├── experiment_results.json        # Machine-readable JSON metrics
│   └── evaluation_report.md           # Formatted markdown summary report
│
├── tests/                             # Comprehensive unit test suite
│   ├── __init__.py
│   ├── test_data_loader.py            # Tests for data preprocessing & scaling
│   ├── test_uncertainty_set.py        # Tests for uncertainty bounds & dual norm calculations
│   ├── test_max_loss_cost.py          # Verifies sweep matches brute-force grid search (<1e-4)
│   ├── test_strategic_classifier.py   # Tests scikit-learn compliance & convergence
│   └── test_strategic_simulator.py    # Tests rational agent movement & boundary crossing
│
├── pyproject.toml                     # Modern package build configuration
├── requirements.txt                   # Production dependencies
├── run.py                             # Root CLI entrypoint
├── strategic_classification.py        # Backwards-compatible import wrapper
└── README.md                          # Comprehensive project documentation
```

---

## ⚡ Installation & Quickstart

### 1. Requirements
Ensure you have Python 3.9+ installed:
```bash
pip install -r requirements.txt
```

Alternatively, install in editable mode:
```bash
pip install -e .
```

### 2. Running the Complete Experiment Pipeline
From inside the `OSSC/` directory:
```bash
python run.py
```
Or with custom hyperparameters:
```bash
python experiments/run_experiments.py --utility 0.5 --iters 100 --lr 0.4
```

### 3. Running Automated Tests
Execute the entire test suite using `pytest`:
```bash
python -m pytest tests -v
```

---

## 📐 Mathematical Formulation & Theory

### 1. Strategic Hinge Loss & Dual Norm
For cost function $c(x, x') = \|\Sigma^{1/2}(x' - x)\|_2$, Lemma 2.3 proves that the maximal increase in an applicant's logit score is:
$$\Delta \text{score}_{\max} = u^* \|\beta\|_{*,\Sigma} = u^* \|\Sigma^{-1/2} \beta\|_2 = u^* \sqrt{\sum_{i=1}^d \left(\frac{\beta_i}{\sigma_i}\right)^2}$$

The cost-dependent strategic hinge loss (Definition 2.2) is:
$$\ell_{\text{s-hinge}}^\Sigma(\beta^\top x + b, y) = \max\left\{0, 1 - y\left(\beta^\top x + b + u^* \|\beta\|_{*,\Sigma}\right)\right\}$$

### 2. Dual Norm Regularization (Ensuring Convexity)
Standard $\ell_2$ regularization fails to prevent non-convexity unless $\lambda \ge \|\Sigma^{-1}\|_2$ (Theorem 4.2). Instead, adding the cost-dependent **Dual Norm Regularizer** guarantees convexity for any cost in $\mathcal{C}$ whenever $\lambda \ge q(y = +1)$ (Proposition 4.3):
$$\min_{\beta, b} \max_{\Sigma \in \mathcal{C}} \left[ \frac{1}{n} \sum_{i=1}^n \max\left\{0, 1 - y_i\left(\beta^\top x_i + b + u^* \|\Sigma^{-1/2}\beta\|_2\right)\right\} + \lambda u^* \|\Sigma^{-1/2}\beta\|_2 \right]$$

### 3. Fast Inner Maximization (`MAXLOSSCOST`) in $\mathcal{O}(nd + n\ln n)$
Evaluating baseline scores $v_i = y_i(1 - y_i(\beta^\top x_i + b))$ and sorting gives the breakpoints where individual training points enter or exit positive hinge loss. Sweeping scalar $k \in [\|\beta\|_{\min}^*, \|\beta\|_{\max}^*]$ enables exact piecewise-linear evaluation in $\mathcal{O}(n\ln n)$ time. The maximizing matrix $\hat{\Sigma}$ is recovered analytically coordinate-by-coordinate:
$$\hat{\sigma}_i = \frac{|\beta_i|}{\left(k_{\max}^2 - \|\hat{\Sigma}_{i=0}^{-1/2} \beta\|_2^2\right)^{1/2}}$$

### 4. Rational Applicant Response Simulation (Test Time)
An applicant receiving a negative score $\beta^\top x + b < 0$ incurs minimal cost to reach the decision boundary:
$$\text{Cost}^*(x) = \frac{-(\beta^\top x + b)}{\|\beta\|_{*,\Sigma^*}}$$
If $\text{Net Utility} = u - \text{Cost}^*(x) > 0$, the applicant moves by optimal displacement:
$$\Delta^*(x) = \frac{-(\beta^\top x + b) + \epsilon}{\|\beta\|_{*,\Sigma^*}^2} \Sigma^{* -1} \beta$$

---

## 📊 Experimental Results on Loan Approval Dataset

Evaluated on $11,250$ holdout loan applications with utility payoff $u = 0.5$:

| Metric | Naive Baseline (Logistic Regression) | Strategic Classifier (Algorithm 3) | Performance / Robustness Impact |
| :--- | :---: | :---: | :--- |
| **Truthful Accuracy (Exp 1)** | 89.97% | 87.90% | Baseline +2.07% on clean data |
| **Manipulated Accuracy (Exp 2)** | 87.61% | **88.39%** | **Strategic outperforms Baseline by +0.78%** |
| **Accuracy Degradation** | **-2.36%** | **-0.49% (Robust)** | **Strategic suffers 79% less degradation** |
| **Truthful FPR (Exp 1)** | 5.62% | 3.73% | Strategic initiates with lower FPR |
| **Manipulated FPR (Exp 2)** | **12.80%** | **5.14%** | **Baseline FPR surges dramatically!** |
| **FPR Surge ($\Delta$ FPR)** | **+7.18%** | **+1.42%** | **Strategic is 5x more resistant to false approvals** |
| **Gaming Applicants** | **990 applicants** | **303 applicants** | **69.4% reduction in gaming** |

---

## 📈 Generated Reports & Visualizations

Running the pipeline automatically saves evaluation artifacts in `reports/`:
- **`reports/evaluation_report.md`**: Detailed quantitative markdown report.
- **`reports/experiment_results.json`**: Machine-readable JSON metrics for benchmarking.
- **`reports/figures/fpr_comparison.png`**: High-resolution chart showing FPR surge under manipulation.
- **`reports/figures/accuracy_comparison.png`**: Chart showing accuracy preservation.
- **`reports/figures/training_loss.png`**: Subgradient optimization loss convergence curve.

---

## 🧪 Automated Unit Test Suite

The test suite in `tests/` tests all components:
- `test_data_loader.py`: Verifies null checks, categorical encoding, label mapping to $\{-1, +1\}$, and feature scaling.
- `test_uncertainty_set.py`: Verifies monotonicity of dual norms, positive definite constraints, and true cost sampling.
- `test_max_loss_cost.py`: Validates that the fast $\mathcal{O}(nd + n\ln n)$ piecewise sweep matches brute-force grid search within $10^{-4}$ tolerance.
- `test_strategic_classifier.py`: Validates scikit-learn estimator compliance (`fit`, `predict`, `decision_function`) and monotonic loss descent.
- `test_strategic_simulator.py`: Verifies applicant gaming conditions, optimal displacement vectors, and strict boundary crossing.
