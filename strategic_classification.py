"""
Backwards-compatible wrapper module for OSSC.

Redirects imports to the modular `src` package:
- `load_and_preprocess_loan_data` from `src.data`
- `CostUncertaintySet`, `construct_domain_cost_bounds` from `src.costs`
- `max_loss_cost` from `src.algorithms`
- `StrategicClassifier`, `BaselineClassifier` from `src.models`
- `StrategicSimulator` from `src.simulation`
- `calculate_metrics` from `src.evaluation`
- `run_experiments` from `experiments.run_experiments`
"""

import os
import sys

# Ensure repository root is on sys.path
_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.data.loader import load_and_preprocess_loan_data
from src.costs.uncertainty_set import CostUncertaintySet
from src.costs.domain_bounds import construct_domain_cost_bounds
from src.algorithms.max_loss_cost import max_loss_cost
from src.models.strategic_classifier import StrategicClassifier
from src.models.baseline import BaselineClassifier
from src.simulation.strategic_response import StrategicSimulator
from src.evaluation.metrics import calculate_metrics, save_results_json, generate_markdown_report
from experiments.run_experiments import run_pipeline


def run_experiments(
    csv_path: str = "data/loan_data.csv",
    utility_u: float = 0.5,
    n_iter: int = 100,
    learning_rate: float = 0.4
):
    """Backwards-compatible run_experiments function."""
    reports_dir = os.path.join(_repo_root, "reports")
    return run_pipeline(
        csv_path=csv_path,
        reports_dir=reports_dir,
        utility_u=utility_u,
        n_iter=n_iter,
        learning_rate=learning_rate
    )


if __name__ == "__main__":
    csv_file = os.path.join(_repo_root, "data", "loan_data.csv")
    run_experiments(csv_path=csv_file, utility_u=0.5, n_iter=100, learning_rate=0.4)
