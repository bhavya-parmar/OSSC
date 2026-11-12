"""
Test suite for MAXLOSSCOST inner-maximization algorithm.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.costs.uncertainty_set import CostUncertaintySet
from src.algorithms.max_loss_cost import max_loss_cost


def test_max_loss_cost_against_grid_search():
    """
    Verifies that the O(nd + n ln n) sweep matches an exhaustive grid search to < 1e-4.
    """
    np.random.seed(42)
    n, d = 100, 4
    X = np.random.randn(n, d)
    y = np.random.choice([-1.0, 1.0], size=n)
    beta = np.random.randn(d)
    b = 0.35
    bounds = np.array([[1.0, 3.0] for _ in range(d)])
    c_set = CostUncertaintySet(bounds)
    u_star = 0.75
    lam = float(np.mean(y == 1.0))

    Sigma_worst, sigma_vec, k_max, r_max = max_loss_cost(
        X, y, beta, b, c_set, u_star, lam
    )

    # Brute-force dense grid search ground truth
    k_min_val = c_set.min_dual_norm(beta)
    k_max_val = c_set.max_dual_norm(beta)
    ks = np.linspace(k_min_val, k_max_val, 3000)
    grid_risks = [
        np.mean(np.maximum(0.0, 1.0 - y * (X @ beta + b + u_star * kv))) + lam * u_star * kv
        for kv in ks
    ]
    grid_max_r = float(np.max(grid_risks))

    assert abs(r_max - grid_max_r) < 1e-4

    # Check that achieved dual norm matches k_max
    achieved_norm = c_set.dual_norm(beta, sigma_vec)
    assert abs(achieved_norm - k_max) < 1e-4
    assert c_set.is_valid_cost(sigma_vec)

