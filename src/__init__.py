"""
One-Shot Strategic Classification Under Unknown Costs (OSSC).

Python implementation of the ICML 2024 paper by Nir Rosenfeld et al.
"""

from .data.loader import load_and_preprocess_loan_data
from .costs.uncertainty_set import CostUncertaintySet
from .costs.domain_bounds import construct_domain_cost_bounds
from .algorithms.max_loss_cost import max_loss_cost
from .models.strategic_classifier import StrategicClassifier
from .models.baseline import BaselineClassifier
from .simulation.strategic_response import StrategicSimulator
from .evaluation.metrics import calculate_metrics, save_results_json, generate_markdown_report

__version__ = "1.0.0"

__all__ = [
    "load_and_preprocess_loan_data",
    "CostUncertaintySet",
    "construct_domain_cost_bounds",
    "max_loss_cost",
    "StrategicClassifier",
    "BaselineClassifier",
    "StrategicSimulator",
    "calculate_metrics",
    "save_results_json",
    "generate_markdown_report",
]

