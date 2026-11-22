"""
Test suite for CostUncertaintySet and domain cost bounds.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.costs.uncertainty_set import CostUncertaintySet
from src.costs.domain_bounds import construct_domain_cost_bounds


def test_uncertainty_set_initialization():
    bounds = np.array([[1.0, 3.0], [2.0, 5.0], [0.5, 1.5]])
    c_set = CostUncertaintySet(bounds)
    assert c_set.d == 3
    assert np.allclose(c_set.sigma_min, [1.0, 2.0, 0.5])
    assert np.allclose(c_set.sigma_max, [3.0, 5.0, 1.5])

    # Error cases
    with pytest.raises(ValueError):
        CostUncertaintySet(np.array([[3.0, 1.0]]))  # min > max
    with pytest.raises(ValueError):
        CostUncertaintySet(np.array([[-1.0, 2.0]]))  # non-positive


def test_dual_norm_monotonicity():
    bounds = np.array([[1.0, 3.0], [2.0, 4.0]])
    c_set = CostUncertaintySet(bounds)
    beta = np.array([2.0, -1.5])

    k_min = c_set.min_dual_norm(beta)
    k_max = c_set.max_dual_norm(beta)
    assert k_min < k_max


def test_true_cost_sampling():
    bounds = np.array([[1.0, 3.0], [2.0, 6.0], [0.8, 2.4]])
    c_set = CostUncertaintySet(bounds)
    Sigma_star, sigma_star = c_set.sample_hidden_true_cost(random_state=99)

    assert Sigma_star.shape == (3, 3)
    assert c_set.is_valid_cost(sigma_star)
    assert np.all(sigma_star > c_set.sigma_min)
    assert np.all(sigma_star < c_set.sigma_max)


def test_cost_scenarios_and_diameter():
    bounds = np.array([[1.0, 3.0], [2.0, 6.0], [0.8, 2.4]])
    c_set = CostUncertaintySet(bounds)

    # 1. Worst-case cost
    Sigma_worst, sigma_worst = c_set.get_worst_case_cost()
    assert np.allclose(sigma_worst, c_set.sigma_min)
    assert np.allclose(np.diag(Sigma_worst), c_set.sigma_min ** 2)

    # 2. Best-case cost
    Sigma_best, sigma_best = c_set.get_best_case_cost()
    assert np.allclose(sigma_best, c_set.sigma_max)

    # 3. Skewed cost (feature 0 cheap, others expensive)
    Sigma_skew, sigma_skew = c_set.get_skewed_cost(cheap_indices=[0])
    assert np.isclose(sigma_skew[0], c_set.sigma_min[0])
    assert np.isclose(sigma_skew[1], c_set.sigma_max[1])
    assert np.isclose(sigma_skew[2], c_set.sigma_max[2])

    # 4. Multiple interior costs
    multiple = c_set.sample_multiple_interior_costs(n_samples=5, random_state=42)
    assert len(multiple) == 5
    for S_mat, s_vec in multiple:
        assert c_set.is_valid_cost(s_vec)

    # 5. Theoretical Diameter D(C)
    d_val = c_set.get_diameter()
    # Expected max over i of (1/sigma_min^2 - 1/sigma_max^2)
    # i=0: 1/1 - 1/9 = 8/9 = 0.8888
    # i=1: 1/4 - 1/36 = 8/36 = 0.2222
    # i=2: 1/0.64 - 1/5.76 = 1.5625 - 0.1736 = 1.3888
    assert np.isclose(d_val, (1.0 / 0.8**2) - (1.0 / 2.4**2))
    assert np.isclose(c_set.diameter, d_val)


